import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 从前向导入
from indexer_loss_fwd_kernel import (
    pytorch_reference,
    generate_index_mask_from_score,
    _indexer_loss_fused_kernel,
    BLOCK_D,
    BLOCK_H,
    NUM_STAGES as FWD_NUM_STAGES,
    NUM_WARPS as FWD_NUM_WARPS,
)

# 固定配置常量
BLOCK_TOPK = 256  # topk 分块大小
NUM_STAGES = 3
NUM_WARPS = 8


@triton.jit
def _indexer_loss_bwd_kernel(
    # 输入
    IndexScore_ptr,  # [batch, chunk_size, topk]
    Indices_ptr,     # [batch, chunk_size, topk]
    AttnSum_ptr,     # [batch, chunk_size, topk] - 前向保存的中间结果
    # 输出
    dIndexScore_ptr, # [batch, chunk_size, topk]
    # 标量参数
    batch_size, chunk_size, chunk_offset,
    topk: tl.constexpr,
    eps: tl.constexpr,
    # strides
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
    stride_asb, stride_ass, stride_ask,
    stride_disb, stride_diss, stride_disk,
    BLOCK_TOPK: tl.constexpr,
):
    """
    反向传播 kernel：计算 index_score 的梯度
    
    数学推导：
    Loss = KL(attn_dist || index_prob) = sum_k attn_dist_k * log(attn_dist_k / index_prob_k)
    
    由于 attn_dist 不依赖于 index_score（只依赖于 Q, K），所以：
    d Loss / d index_score_j = d/d(index_score_j) [ -sum_k attn_dist_k * log(index_prob_k) ]
    
    使用 softmax 的梯度公式：
    d Loss / d index_score_j = index_prob_j - attn_dist_j
    """
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    NEG_INF = -1e9
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    
    # 基地址
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    attn_sum_base = AttnSum_ptr + pid_batch * stride_asb + pid_row * stride_ass
    dis_base = dIndexScore_ptr + pid_batch * stride_disb + pid_row * stride_diss
    
    # =========================================================================
    # Step 1: 计算 attn_total (attn_sum 的总和，用于归一化成 attn_dist)
    # =========================================================================
    attn_total = 0.0
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_total += tl.sum(attn_sum_block)
    
    attn_total = tl.where(attn_total < eps, 1.0, attn_total)
    
    # =========================================================================
    # Step 2: 对 index_score 做 Online Softmax 得到 index_prob
    # =========================================================================
    is_m_global = NEG_INF
    is_l_global = 0.0
    
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF)
        is_val = tl.where(causal_mask_block, NEG_INF, is_val)
        
        is_m_block = tl.max(is_val)
        is_m_new = tl.maximum(is_m_global, is_m_block)
        is_exp_val = tl.exp(is_val - is_m_new)
        is_exp_val = tl.where(causal_mask_block, 0.0, is_exp_val)
        is_l_global = is_l_global * tl.exp(is_m_global - is_m_new) + tl.sum(is_exp_val)
        is_m_global = is_m_new
    
    is_m_global = tl.where(is_m_global == NEG_INF, 0.0, is_m_global)
    is_l_global = tl.where(is_l_global < 1e-9, 1.0, is_l_global)
    
    # =========================================================================
    # Step 3: 计算梯度 grad_index_score = index_prob - attn_dist
    # =========================================================================
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        # 计算 attn_dist = attn_sum / attn_total
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_dist = attn_sum_block / attn_total
        
        # 计算 index_prob = softmax(index_score)
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF)
        is_val = tl.where(causal_mask_block, NEG_INF, is_val)
        index_prob = tl.exp(is_val - is_m_global) / is_l_global
        
        # 梯度 = index_prob - attn_dist
        grad = index_prob - attn_dist
        grad = tl.where(causal_mask_block, 0.0, grad)
        
        # 写出梯度
        dis_ptrs = dis_base + offs_tk * stride_disk
        tl.store(dis_ptrs, grad, mask=tk_mask)


# ============================================================================
# Wrapper函数
# ============================================================================

def compute_index_loss_bwd(index_score, indices, attn_sum, chunk_offset=0, eps=1e-10, block_topk=None):
    """
    计算 index_score 的梯度
    
    Args:
        index_score: [batch, chunk_size, topk] - sparse版本的index分数
        indices: [batch, chunk_size, topk] - 每个query选择的topk个key索引
        attn_sum: [batch, chunk_size, topk] - 前向传播保存的中间结果
        chunk_offset: 当前chunk在完整序列中的起始位置 (用于causal mask)
        eps: 数值稳定epsilon
        block_topk: topk 分块大小，默认使用全局 BLOCK_TOPK
    
    Returns:
        grad_index_score: [batch, chunk_size, topk] - index_score的梯度
    """
    batch_size, chunk_size, topk = index_score.shape
    
    # 选择合适的 block_topk
    if block_topk is None:
        block_topk = min(BLOCK_TOPK, topk)
    if block_topk < 16:
        block_topk = 16
    
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    attn_sum = attn_sum.contiguous()
    
    # 输出：index_score的梯度
    grad_index_score = torch.zeros_like(index_score, dtype=torch.float32)
    
    # 每个program处理一个(batch, query_row)
    grid = (batch_size * chunk_size,)
    
    _indexer_loss_bwd_kernel[grid](
        index_score, indices, attn_sum,
        grad_index_score,
        batch_size, chunk_size, chunk_offset, topk, eps,
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),
        grad_index_score.stride(0), grad_index_score.stride(1), grad_index_score.stride(2),
        BLOCK_TOPK=block_topk,
        num_stages=NUM_STAGES, num_warps=NUM_WARPS,
    )
    
    # 除以 batch_size（对应前向 loss = sum / batch_size）
    return grad_index_score / batch_size


# ============================================================================
# 前向 Wrapper (返回 loss 和 attn_sum)
# ============================================================================

def compute_index_loss_fwd_with_attn_sum(query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10, block_topk=None):
    """
    前向计算，返回 loss 和 attn_sum (用于反向传播)
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim]
        key: [batch, kv_len, head_dim]
        index_score: [batch, chunk_size, topk]
        indices: [batch, chunk_size, topk]
        scaling: attention scaling factor
        chunk_offset: 当前chunk在完整序列中的起始位置
        eps: 数值稳定epsilon
        block_topk: topk 分块大小
    
    Returns:
        loss: 标量loss值
        attn_sum: [batch, chunk_size, topk] - 中间结果，用于反向传播
    """
    batch_size, num_heads, chunk_size, head_dim = query.shape
    topk = indices.shape[-1]
    
    if block_topk is None:
        block_topk = min(BLOCK_TOPK, topk)
    if block_topk < 16:
        block_topk = 16
    
    query = query.contiguous()
    key = key.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    loss_per_row = torch.zeros(batch_size, chunk_size, device=query.device, dtype=torch.float32)
    attn_sum = torch.zeros(batch_size, chunk_size, topk, device=query.device, dtype=torch.float32)
    
    grid = (batch_size * chunk_size,)
    
    block_h = min(BLOCK_H, num_heads)
    if block_h < 16:
        block_h = 16
    
    _indexer_loss_fused_kernel[grid](
        query, key, index_score, indices, loss_per_row,
        attn_sum,
        batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, eps,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),
        BLOCK_D=BLOCK_D,
        BLOCK_TOPK=block_topk,
        BLOCK_H=block_h,
        num_stages=FWD_NUM_STAGES, num_warps=FWD_NUM_WARPS,
    )
    
    loss = loss_per_row.sum() / batch_size
    return loss, attn_sum


# ============================================================================
# PyTorch参考实现 (使用 autograd)
# ============================================================================

def pytorch_reference_bwd_autograd(query, key, index_score_full, index_mask, topk_indices, scaling):
    """
    PyTorch参考实现：使用 loss.backward() 计算 index_score 的梯度
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim]
        key: [batch, kv_len, head_dim]
        index_score_full: [batch, chunk_size, kv_len] - 需要 requires_grad=True
        index_mask: [batch, 1, chunk_size, kv_len] - True表示需要mask
        topk_indices: [batch, chunk_size, topk] - topk索引，用于gather梯度
        scaling: attention scaling factor
    
    Returns:
        grad_index_score_sparse: [batch, chunk_size, topk] - topk位置的梯度
    """
    # 确保 index_score_full 需要梯度
    if not index_score_full.requires_grad:
        index_score_full = index_score_full.detach().requires_grad_(True)
    
    # 前向计算 loss
    loss = pytorch_reference(query, key, index_score_full, index_mask, scaling)
    
    # 反向传播
    loss.backward()
    
    # 从 full 梯度中 gather 出 topk 位置的梯度
    grad_full = index_score_full.grad  # [batch, chunk_size, kv_len]
    grad_sparse = torch.gather(grad_full, dim=-1, index=topk_indices)  # [batch, chunk_size, topk]
    
    return grad_sparse


# ============================================================================
# 测试配置和函数
# ============================================================================

@dataclass
class TestConfig:
    """测试配置"""
    name: str
    batch_size: int = 1
    num_heads: int = 8
    chunk_size: int = 256
    seq_len: int = 256
    head_dim: int = 64
    topk: int = 32
    seed: int = 42
    
    def __str__(self):
        return (f"batch={self.batch_size}, heads={self.num_heads}, "
                f"chunk={self.chunk_size}, seq={self.seq_len}, "
                f"dim={self.head_dim}, topk={self.topk}")


def run_single_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """运行单个精度测试
    
    测试流程：
    1. PyTorch: pytorch_reference 前向 -> loss.backward() -> gather topk 梯度
    2. Triton: 前向 kernel 得到 attn_sum -> 反向 kernel 计算梯度
    3. 比较两者梯度
    """
    torch.manual_seed(config.seed)
    scaling = 1.0 / (config.head_dim ** 0.5)
    
    # 生成测试数据
    query = torch.randn(config.batch_size, config.num_heads, config.chunk_size, 
                        config.head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(config.batch_size, config.seq_len, config.head_dim, 
                      device=device, dtype=torch.bfloat16)
    
    # Full 版本的 index_score (需要梯度)
    index_score_full = torch.randn(config.batch_size, config.chunk_size, config.seq_len, 
                                   device=device, dtype=torch.float32, requires_grad=True)
    
    # 生成 mask 和 indices
    chunk_offset = config.seq_len - config.chunk_size
    index_mask, topk_indices = generate_index_mask_from_score(
        index_score_full.detach(), config.topk, device, chunk_offset=chunk_offset)
    
    # Sparse 版本的 index_score
    index_score_sparse = torch.gather(index_score_full.detach(), dim=-1, index=topk_indices)
    
    # =========================================================================
    # PyTorch 参考实现：使用 autograd
    # =========================================================================
    ref_grad = pytorch_reference_bwd_autograd(
        query, key, index_score_full, index_mask, topk_indices, scaling
    )
    
    # =========================================================================
    # Triton 实现：前向 kernel + 反向 kernel
    # =========================================================================
    # 前向获取 attn_sum
    _, attn_sum = compute_index_loss_fwd_with_attn_sum(
        query, key, index_score_sparse.to(torch.bfloat16), topk_indices, 
        scaling, chunk_offset=chunk_offset
    )
    
    # 反向计算梯度
    tri_grad = compute_index_loss_bwd(
        index_score_sparse, topk_indices, attn_sum, 
        chunk_offset=chunk_offset
    )
    
    # 比较结果
    abs_diff = (ref_grad - tri_grad).abs().max().item()


    def print_diff(a, b, msg):
        abs_diff = torch.abs(a - b)
        print(f"  {msg}"
              f"    Max diff: {abs_diff.max().item(): .4f}"
              f"\tRel diff: {(abs_diff / (1e-4 + torch.abs(a))).mean().item() * 100: .4f}%")
    print_diff(ref_grad, tri_grad, "===============\n")

    rel_diff = abs_diff / (ref_grad.abs().max().item() + 1e-10)
    passed = rel_diff < 1e-2  # 由于精度差异，放宽到 1e-2
    
    return {
        'config': config,
        'ref_grad_max': ref_grad.abs().max().item(),
        'tri_grad_max': tri_grad.abs().max().item(),
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'passed': passed
    }


def test_full_accuracy(configs: List[TestConfig]):
    """批量运行精度测试"""
    print("\n" + "=" * 100)
    print("反向传播精度测试 (PyTorch autograd vs Triton kernel)")
    print("=" * 100)
    
    results = []
    for config in configs:
        result = run_single_accuracy_test(config)
        results.append(result)
    
    print(f"\n{'Name':<12} {'Config':<55} {'AbsDiff':<12} {'RelDiff':<12} {'Pass':<6}")
    print("-" * 97)
    for r in results:
        print(f"{r['config'].name:<12} {str(r['config']):<55} "
              f"{r['abs_diff']:<12.2e} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 97)
    print(f"总计: {passed_count}/{len(results)} 通过")
    
    return results


def test_performance(
    batch_size: int = 1,
    chunk_size: int = 8 * 1024,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 10,
    num_benchmark: int = 50,
):
    """性能测试"""
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    
    print("=" * 70)
    print("反向传播性能测试")
    print("=" * 70)
    print(f"参数: batch={batch_size}, chunk={chunk_size}, topk={topk}")
    print("=" * 70)
    
    # 生成测试数据
    index_score = torch.randn(batch_size, chunk_size, topk, device=device, dtype=torch.bfloat16)
    max_seq_len = chunk_size * 2
    chunk_offset = max_seq_len - chunk_size
    indices = torch.randint(0, max_seq_len, (batch_size, chunk_size, topk),
                           device=device, dtype=torch.int64)
    attn_sum = torch.rand(batch_size, chunk_size, topk, device=device, dtype=torch.float32).abs()
    
    results = {}
    
    # Triton 性能测试
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = compute_index_loss_bwd(index_score, indices, attn_sum, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_bwd(index_score, indices, attn_sum, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    triton_peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    results['triton'] = triton_time
    
    # PyTorch 性能测试
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = pytorch_reference_bwd(index_score, indices, attn_sum, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = pytorch_reference_bwd(index_score, indices, attn_sum, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_benchmark * 1000
    pytorch_peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    results['pytorch'] = pytorch_time
    
    print(f"\n>>> 性能结果 (warmup={num_warmup}, iters={num_benchmark})")
    print(f"  PyTorch:    {pytorch_time:.3f} ms")
    print(f"  Triton:     {triton_time:.3f} ms (加速: {pytorch_time/triton_time:.2f}x)")
    
    print(f"\n>>> 显存峰值")
    print(f"  Triton:   {triton_peak_memory:.2f} GB")
    print(f"  PyTorch:  {pytorch_peak_memory:.2f} GB")
    
    return results


if __name__ == "__main__":
    # 精度测试配置
    accuracy_configs = [
        TestConfig(name="小规模", batch_size=1, num_heads=4, chunk_size=32, seq_len=64, head_dim=32, topk=16),
        TestConfig(name="中等规模", batch_size=1, num_heads=8, chunk_size=128, seq_len=256, head_dim=64, topk=64),
        TestConfig(name="大规模", batch_size=1, num_heads=16, chunk_size=512, seq_len=1024, head_dim=128, topk=256),
        TestConfig(name="多batch", batch_size=4, num_heads=8, chunk_size=64, seq_len=128, head_dim=64, topk=32),
        TestConfig(name="大head_dim", batch_size=1, num_heads=8, chunk_size=128, seq_len=256, head_dim=256, topk=64),
        TestConfig(name="大规模", batch_size=1, num_heads=128, chunk_size=4096, seq_len=4096, head_dim=576, topk=2048),
        TestConfig(name="大规模", batch_size=1, num_heads=128, chunk_size=4096, seq_len=8192, head_dim=576, topk=2048),
        TestConfig(name="大规模", batch_size=1, num_heads=128, chunk_size=8192, seq_len=8192, head_dim=576, topk=2048),
        TestConfig(name="大规模", batch_size=1, num_heads=128, chunk_size=8192, seq_len=16 * 1024, head_dim=576, topk=2048),
    ]
    
    # 运行精度测试
    test_full_accuracy(accuracy_configs)
