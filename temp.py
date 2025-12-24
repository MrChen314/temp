"""
Triton Fused Optimized - Sparse Attention Loss (H20 GPU优化版本)

针对H20 (Hopper架构, sm_90) 的优化:
1. 固定配置: BLOCK_D=128, num_stages=3, num_warps=8
2. 单个kernel完成 attention softmax + head sum + KL loss 计算
3. 无需中间tensor，减少显存使用
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# 固定配置常量
BLOCK_D = 128
NUM_STAGES = 3
NUM_WARPS = 8


# ============================================================================
# Fused Kernel: Sparse Attention + Loss (优化编译时间版本)
# ============================================================================

@triton.jit
def _sparse_attn_loss_fused_kernel(
    Q_ptr, K_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    batch_size, num_heads, seq_len,  # num_heads 改为非 constexpr，避免循环展开
    head_dim: tl.constexpr,
    topk: tl.constexpr,  # topk 保持 constexpr 以保证正确性
    scaling,
    eps: tl.constexpr,
    # Strides for Q: [batch, num_heads, seq_len, head_dim]
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K: [batch, seq_len, head_dim]
    stride_kb, stride_ks, stride_kd,
    # Strides for IndexScore: [batch, seq_len, topk]
    stride_isb, stride_iss, stride_isk,
    # Strides for Indices: [batch, seq_len, topk]
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
):
    """
    完全融合的 Sparse Attention + Loss Kernel (优化编译时间版本)
    
    优化策略:
    1. num_heads 使用动态 range，避免 16 次循环展开
    2. topk 保持 constexpr 以保证 softmax 归一化正确性
    
    编译时间优化效果:
    - 原版: num_heads=16 时循环完全展开，代码膨胀 16 倍
    - 优化后: 循环不展开，编译时间大幅减少
    
    输出: 每个(batch, row)位置的loss值
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    NEG_INF = -1e9
    
    # 预计算偏移量
    offs_topk = tl.arange(0, topk)
    
    # 基地址计算
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    
    # 加载 indices [topk]
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    
    # Causal mask: indices > current_row 的位置需要 mask
    causal_mask = indices > pid_row
    
    # =========================================================================
    # Part 1: 累加所有 heads 的 attention scores
    # =========================================================================
    attn_sum = tl.zeros([topk], dtype=tl.float32)
    
    # 对每个 head 循环 (使用动态 range，不展开循环)
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        # 计算 QK^T - 分块处理 head_dim
        qk = tl.zeros([topk], dtype=tl.float32)
        
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            # 加载 Q chunk: [BLOCK_D]
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            
            # 批量 load K: [topk, BLOCK_D]
            k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
            
            # 向量化点积: q[d] * k_gathered[topk, d] -> sum over d -> [topk]
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        # 应用 scaling 和 causal mask
        qk = qk * scaling
        qk = tl.where(causal_mask, NEG_INF, qk)
        
        # Softmax (数值稳定版本)
        m = tl.max(qk)
        m = tl.where(m == NEG_INF, 0.0, m)
        p = tl.exp(qk - m)
        l = tl.sum(p)
        l = tl.where(l < 1e-9, 1.0, l)
        p = p / l
        p = tl.where(causal_mask, 0.0, p)
        
        # 累加到 attn_sum
        attn_sum += p
    
    # =========================================================================
    # Part 2: 归一化 attention 分布
    # =========================================================================
    attn_total = tl.sum(attn_sum)
    attn_total = tl.where(attn_total < eps, 1.0, attn_total)
    attn_dist = attn_sum / attn_total + eps
    
    # =========================================================================
    # Part 3: 计算 index_score 的 softmax
    # =========================================================================
    is_val = tl.load(is_base + offs_topk * stride_isk)
    is_val = tl.where(causal_mask, NEG_INF, is_val)
    
    m_is = tl.max(is_val)
    m_is = tl.where(m_is == NEG_INF, 0.0, m_is)
    p_is = tl.exp(is_val - m_is)
    s_is = tl.sum(p_is)
    s_is = tl.where(s_is < 1e-9, 1.0, s_is)
    index_prob = p_is / s_is + eps
    
    # =========================================================================
    # Part 4: 计算 KL 散度
    # =========================================================================
    kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
    kl = tl.where(causal_mask, 0.0, kl)
    kl_sum = tl.sum(kl)
    
    # 写出 loss
    tl.store(Loss_ptr + pid_batch * seq_len + pid_row, kl_sum)


# ============================================================================
# Wrapper函数
# ============================================================================

def compute_index_loss_sparse(query, key, index_score, indices, scaling, eps=1e-10):
    """
    Sparse版本的完整loss计算 (H20优化, 完全融合版本)
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, seq_len, head_dim]
        index_score: [batch, seq_len, topk]
        indices: [batch, seq_len, topk]
        scaling: attention scaling factor
        eps: 数值稳定epsilon
    
    Returns:
        loss: 标量loss值
    
    编译时间优化:
        - num_heads 使用动态循环，不展开，大幅减少编译时间
        - topk 保持 constexpr 以保证正确性
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    
    query = query.contiguous()
    key = key.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    # 输出: 每行的loss
    loss_per_row = torch.zeros(batch_size, seq_len, device=query.device, dtype=torch.float32)
    
    # 每个program处理一个(batch, row)
    grid = (batch_size * seq_len,)
    
    _sparse_attn_loss_fused_kernel[grid](
        query, key, index_score, indices, loss_per_row,
        batch_size, num_heads, seq_len, head_dim, topk, scaling, eps,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        BLOCK_D=BLOCK_D,
        num_stages=NUM_STAGES, num_warps=NUM_WARPS,
    )
    
    return loss_per_row.sum() / batch_size


# ============================================================================
# PyTorch参考实现 (Full版本)
# ============================================================================

def pytorch_reference(query, key, index_score, index_mask, scaling):
    """完整的PyTorch参考实现 (Full版本)
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, seq_len, head_dim]
        index_score: [batch, seq_len, seq_len]
        index_mask: [batch, 1, seq_len, seq_len] - True表示需要mask的位置
        scaling: attention scaling factor
    
    Returns:
        kl_loss: 标量loss值
    """
    eps = 1e-10
    
    # 计算attention: [batch, num_heads, seq_len, seq_len]
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(index_mask, -1e9)
    attn = torch.softmax(attn, dim=-1)
    
    # Head sum + normalize
    attn_sum = attn.sum(dim=1)  # [batch, seq_len, seq_len]
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    # Index score softmax (使用相同的mask，去掉head维度)
    index_mask_2d = index_mask.squeeze(1)  # [batch, seq_len, seq_len]
    index_score_masked = index_score.masked_fill(index_mask_2d, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    # KL散度
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


# ============================================================================
# 测试辅助函数
# ============================================================================

def generate_index_mask_from_score(index_score, topk, device='cuda'):
    """
    从index_score生成index_mask和topk_indices
    
    Args:
        index_score: [batch, seq_len, seq_len]
        topk: topk的数量
        device: 设备
    
    Returns:
        index_mask: [batch, 1, seq_len, seq_len] - True表示需要mask的位置
        topk_indices: [batch, seq_len, topk]
    """
    batch_size, seq_len, _ = index_score.shape
    
    # 创建causal mask (上三角为True)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device), 
        diagonal=1
    ).bool()
    
    # 对index_score应用causal mask
    causal_index_score = index_score.masked_fill(causal_mask, float('-inf'))
    
    # 取topk得到topk_indices
    topk_indices = causal_index_score.topk(topk, dim=-1)[1]
    
    # 创建index_mask（只在topk位置为False，其他为True）
    index_mask = torch.full(
        causal_index_score.shape, 
        True, 
        device=device
    ).scatter_(-1, topk_indices, False)
    
    # 与causal_mask合并
    index_mask = torch.logical_or(index_mask, causal_mask)
    
    # 添加head维度: [batch, 1, seq_len, seq_len]
    index_mask = index_mask.unsqueeze(1)
    
    return index_mask, topk_indices


# ============================================================================
# 测试函数
# ============================================================================

def test_full_accuracy(batch_size=1, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试完整流程精度 (Full版本)"""
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Full版本: index_score是 [batch, seq_len, seq_len]
    index_score_full = torch.randn(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    
    # 从index_score生成mask和indices
    index_mask, topk_indices = generate_index_mask_from_score(index_score_full, topk, device)
    
    # 从full index_score中gather出sparse index_score给Triton kernel使用
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    # PyTorch Full版本参考
    ref = pytorch_reference(query, key, index_score_full, index_mask, scaling)
    
    # Triton Sparse版本
    tri = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling)
    
    diff = abs(ref.item() - tri.item())
    passed = diff < 1e-3
    print(f"Accuracy - PyTorch(Full): {ref.item():.6f}, Triton(Sparse): {tri.item():.6f}, Diff: {diff:.6e}, Pass: {passed}")
    return passed


def test_performance(
    batch_size: int = 1,
    num_heads: int = 16,
    seq_len: int = 4096,
    head_dim: int = 256,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 10,
    num_benchmark: int = 50,
):
    """性能测试"""
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("=" * 70)
    print("Triton Sparse vs PyTorch Full 性能测试")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print(f"Sparse复杂度: O(seq * topk * head_dim * num_heads) = O({seq_len * topk * head_dim * num_heads:,})")
    print(f"Full复杂度:   O(seq * seq * head_dim * num_heads) = O({seq_len * seq_len * head_dim * num_heads:,})")
    print("=" * 70)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Full版本数据
    index_score_full = torch.randn(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    index_mask, topk_indices = generate_index_mask_from_score(index_score_full, topk, device)
    
    # Sparse版本数据
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    results = {}
    
    # Test 1: Triton fused kernel (Sparse)
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    results['triton_sparse'] = triton_time
    
    # Test 2: PyTorch reference (Full)
    for _ in range(num_warmup):
        _ = pytorch_reference(query, key, index_score_full, index_mask, scaling)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = pytorch_reference(query, key, index_score_full, index_mask, scaling)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_benchmark * 1000
    results['pytorch_full'] = pytorch_time
    
    print(f"\n>>> 性能结果 (warmup={num_warmup}, iters={num_benchmark})")
    print(f"  PyTorch Full ref:      {pytorch_time:.3f} ms")
    print(f"  Triton Sparse fused:   {triton_time:.3f} ms (加速: {pytorch_time/triton_time:.2f}x)")
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("精度测试 (PyTorch Full vs Triton Sparse)")
    print("=" * 70)
    
    print("\n[小规模测试]")
    test_full_accuracy(batch_size=1, num_heads=4, seq_len=64, head_dim=32, topk=16)
    
    print("\n[中等规模测试]")
    test_full_accuracy(batch_size=1, num_heads=8, seq_len=256, head_dim=64, topk=64)
    
    print("\n[大规模测试]")
    test_full_accuracy(batch_size=1, num_heads=16, seq_len=1024, head_dim=128, topk=256)
    
    print("\n")
    test_performance(
        batch_size=1,
        num_heads=16,
        seq_len=8 * 1024,
        head_dim=512,
        topk=2048,
        num_warmup=2,
        num_benchmark=3,
    )
