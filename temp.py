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
    batch_size, num_heads, chunk_size,  # num_heads 改为非 constexpr，避免循环展开
    chunk_offset,  # 当前 chunk 在完整序列中的起始位置
    head_dim: tl.constexpr,
    topk: tl.constexpr,  # topk 保持 constexpr 以保证正确性
    scaling,
    eps: tl.constexpr,
    # Strides for Q: [batch, num_heads, chunk_size, head_dim]
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K: [batch, kv_len, head_dim] (kv_len 通过 indices 隐式处理)
    stride_kb, stride_ks, stride_kd,
    # Strides for IndexScore: [batch, chunk_size, topk]
    stride_isb, stride_iss, stride_isk,
    # Strides for Indices: [batch, chunk_size, topk]
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
):
    """
    完全融合的 Sparse Attention + Loss Kernel (优化编译时间版本)
    
    支持分块注意力场景:
    - chunk_size: 当前chunk的query数量
    - kv_len: key的完整长度 (通过indices间接访问，无需显式传入)
    
    优化策略:
    1. num_heads 使用动态 range，避免 16 次循环展开
    2. topk 保持 constexpr 以保证 softmax 归一化正确性
    
    编译时间优化效果:
    - 原版: num_heads=16 时循环完全展开，代码膨胀 16 倍
    - 优化后: 循环不展开，编译时间大幅减少
    
    输出: 每个(batch, query_row)位置的loss值
    """
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size  # 当前处理的query行
    
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
    
    # 计算当前 query 的全局位置
    global_query_pos = chunk_offset + pid_row
    
    # Causal mask: indices > global_query_pos 的位置需要 mask
    # 即：只能 attend 到位置 <= 当前全局位置的 key
    causal_mask = indices > global_query_pos
    
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
    tl.store(Loss_ptr + pid_batch * chunk_size + pid_row, kl_sum)


# ============================================================================
# Wrapper函数
# ============================================================================

def compute_index_loss_sparse(query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10):
    """
    Sparse版本的完整loss计算 (H20优化, 完全融合版本)
    
    支持分块注意力场景: chunk_size (query长度) 可以不等于 kv_len (key长度)
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim] - 当前chunk的query
        key: [batch, kv_len, head_dim] - 完整的key (KV cache)
        index_score: [batch, chunk_size, topk] - sparse版本的index分数
        indices: [batch, chunk_size, topk] - 每个query选择的topk个key索引
        scaling: attention scaling factor
        chunk_offset: 当前chunk在完整序列中的起始位置 (用于causal mask)
        eps: 数值稳定epsilon
    
    Returns:
        loss: 标量loss值
    
    编译时间优化:
        - num_heads 使用动态循环，不展开，大幅减少编译时间
        - topk 保持 constexpr 以保证正确性
    """
    batch_size, num_heads, chunk_size, head_dim = query.shape
    topk = indices.shape[-1]
    
    query = query.contiguous()
    key = key.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    # 输出: 每行(每个query位置)的loss
    loss_per_row = torch.zeros(batch_size, chunk_size, device=query.device, dtype=torch.float32)
    
    # 每个program处理一个(batch, query_row)
    grid = (batch_size * chunk_size,)
    
    _sparse_attn_loss_fused_kernel[grid](
        query, key, index_score, indices, loss_per_row,
        batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, eps,
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
    
    支持分块注意力场景: chunk_size (query长度) 可以不等于 kv_len (key长度)
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim] - 当前chunk的query
        key: [batch, kv_len, head_dim] - 完整的key (KV cache)
        index_score: [batch, chunk_size, kv_len] - 每个query对所有key的分数
        index_mask: [batch, 1, chunk_size, kv_len] - True表示需要mask的位置
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

def generate_index_mask_from_score(index_score, topk, device='cuda', chunk_offset=0):
    """
    从index_score生成index_mask和topk_indices
    
    用于分块注意力计算场景:
    - chunk_size: 当前chunk的query长度
    - seq_len: 完整序列的key长度 (KV cache长度)
    
    Args:
        index_score: [batch, chunk_size, seq_len] - 每个query位置对所有key位置的分数
        topk: 每个query位置选择的top-k个key
        device: 设备
        chunk_offset: 当前chunk在完整序列中的起始位置 (用于causal mask)
    
    Returns:
        index_mask: [batch, 1, chunk_size, seq_len] - True表示需要mask的位置
        topk_indices: [batch, chunk_size, topk] - 每个query选择的topk个key的索引
    """
    batch_size, chunk_size, seq_len = index_score.shape
    
    # 创建causal mask: query位置 i 只能看到 key位置 <= chunk_offset + i
    # 对于 chunk 内的第 i 个 query，其全局位置是 chunk_offset + i
    # 它只能 attend 到 key 位置 j，其中 j <= chunk_offset + i
    query_positions = chunk_offset + torch.arange(chunk_size, device=device).view(-1, 1)
    key_positions = torch.arange(seq_len, device=device).view(1, -1)
    causal_mask = key_positions > query_positions  # [chunk_size, seq_len]
    
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
    
    # 添加head维度: [batch, 1, chunk_size, seq_len]
    index_mask = index_mask.unsqueeze(1)
    
    return index_mask, topk_indices


# ============================================================================
# 测试函数
# ============================================================================

def test_full_accuracy(batch_size=1, num_heads=8, chunk_size=256, seq_len=256, head_dim=64, topk=32, seed=42):
    """
    测试完整流程精度 (Full版本)
    
    Args:
        batch_size: 批次大小
        num_heads: 注意力头数
        chunk_size: 当前chunk的query长度
        seq_len: 完整序列长度 (KV cache的长度)
        head_dim: 每个头的维度
        topk: 每个query位置选择的top-k个key
        seed: 随机种子
    """
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    # query: [batch, num_heads, chunk_size, head_dim]
    query = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device, dtype=torch.float32)
    # key: [batch, seq_len, head_dim]
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Full版本: index_score是 [batch, chunk_size, seq_len]
    index_score_full = torch.randn(batch_size, chunk_size, seq_len, device=device, dtype=torch.float32)
    
    # 从index_score生成mask和indices
    # chunk_offset: 当前chunk在完整序列中的起始位置
    chunk_offset = seq_len - chunk_size
    index_mask, topk_indices = generate_index_mask_from_score(index_score_full, topk, device, chunk_offset=chunk_offset)
    
    # 从full index_score中gather出sparse index_score给Triton kernel使用
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    # PyTorch Full版本参考
    ref = pytorch_reference(query, key, index_score_full, index_mask, scaling)
    
    # Triton Sparse版本 (传入 chunk_offset)
    tri = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    
    diff = abs(ref.item() - tri.item())
    passed = diff < 1e-3
    print(f"Accuracy - PyTorch(Full): {ref.item():.6f}, Triton(Sparse): {tri.item():.6f}, Diff: {diff:.6e}, Pass: {passed}")
    return passed


def test_performance(
    batch_size: int = 1,
    num_heads: int = 16,
    chunk_size: int = 16 * 1024,
    seq_len: int = 4096,
    head_dim: int = 256,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 10,
    num_benchmark: int = 50,
    triton_only: bool = False,
):
    """
    性能测试
    
    Args:
        batch_size: 批次大小
        num_heads: 注意力头数
        chunk_size: 当前chunk的query长度 (通常 chunk_size <= seq_len)
        seq_len: 完整序列长度 (KV cache的长度)
        head_dim: 每个头的维度
        topk: 每个query位置选择的top-k个key
        seed: 随机种子
        num_warmup: 预热次数
        num_benchmark: 测试次数
        triton_only: 只测试Triton kernel，跳过PyTorch参考实现 (避免OOM或加速测试)
    
    数据形状:
        query: [batch, num_heads, chunk_size, head_dim]
        key: [batch, seq_len, head_dim]
        index_score: [batch, chunk_size, seq_len] -> sparse: [batch, chunk_size, topk]
        indices: [batch, chunk_size, topk]
    """
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("=" * 70)
    print("Triton Sparse 性能测试" if triton_only else "Triton Sparse vs PyTorch Full 性能测试")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print(f"Sparse复杂度: O(chunk * topk * head_dim * num_heads) = O({chunk_size * topk * head_dim * num_heads:,})")
    if not triton_only:
        print(f"Full复杂度:   O(chunk * seq * head_dim * num_heads) = O({chunk_size * seq_len * head_dim * num_heads:,})")
        print(f"理论加速比:   seq / topk = {seq_len} / {topk} = {seq_len / topk:.2f}x")
    print("=" * 70)
    
    # query: [batch, num_heads, chunk_size, head_dim] - 当前chunk的query
    query = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device, dtype=torch.float32)
    # key: [batch, seq_len, head_dim] - 完整序列的key (KV cache)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # chunk_offset: 假设当前chunk从序列末尾开始
    chunk_offset = seq_len - chunk_size
    
    # # Sparse版本数据: 直接生成，避免生成 Full 版本的大矩阵
    # if triton_only:
    #     # 直接生成 sparse 数据，节省显存
    #     index_score_sparse = torch.randn(batch_size, chunk_size, topk, device=device, dtype=torch.float32)
    #     # 生成 causal 的 topk indices
    #     topk_indices = torch.zeros(batch_size, chunk_size, topk, dtype=torch.int64, device=device)
    #     for t in range(chunk_size):
    #         global_pos = chunk_offset + t
    #         valid_range = global_pos + 1
    #         if valid_range >= topk:
    #             for b in range(batch_size):
    #                 perm = torch.randperm(valid_range, device=device)[:topk]
    #                 topk_indices[b, t] = perm
    #         else:
    #             base = torch.arange(valid_range, device=device)
    #             extra = torch.randint(0, max(1, valid_range), (topk - valid_range,), device=device)
    #             topk_indices[:, t] = torch.cat([base, extra]).unsqueeze(0).expand(batch_size, -1)
    # else:
    # Full版本数据: index_score [batch, chunk_size, seq_len]
    index_score_full = torch.randn(batch_size, chunk_size, seq_len, device=device, dtype=torch.float32)
    index_mask, topk_indices = generate_index_mask_from_score(index_score_full, topk, device, chunk_offset=chunk_offset)
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    results = {}
    
    # Test 1: Triton fused kernel (Sparse)
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    results['triton_sparse'] = triton_time
    
    # Test 2: PyTorch reference (Full) - 仅当 triton_only=False 时执行
    if not triton_only:
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
    if triton_only:
        print(f"  Triton Sparse fused:   {triton_time:.3f} ms")
    else:
        print(f"  PyTorch Full ref:      {pytorch_time:.3f} ms")
        print(f"  Triton Sparse fused:   {triton_time:.3f} ms (加速: {pytorch_time/triton_time:.2f}x)")
    
    return results


if __name__ == "__main__":
    import sys
    
    # 检查是否运行 profiling 模式
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        # 运行 kernel 各部分耗时分析
        profile_kernel_parts(
            batch_size=1,
            num_heads=128,
            chunk_size=4 * 1024,
            seq_len=8 * 1024,
            head_dim=512,
            topk=2048,
            num_warmup=3,
            num_iters=10,
        )
    else:
        print("\n" + "=" * 70)
        print("精度测试 (PyTorch Full vs Triton Sparse)")
        print("=" * 70)
        
        print("\n[小规模测试]")
        test_full_accuracy(batch_size=1, num_heads=4, chunk_size=32, seq_len=64, head_dim=32, topk=16)
        
        print("\n[中等规模测试]")
        test_full_accuracy(batch_size=1, num_heads=8, chunk_size=128, seq_len=256, head_dim=64, topk=64)
        
        print("\n[大规模测试]")
        test_full_accuracy(batch_size=1, num_heads=16, chunk_size=512, seq_len=1024, head_dim=128, topk=256)
        
        print("\n")
        test_performance(
            batch_size=1,
            num_heads=128,
            chunk_size=4 * 1024,
            seq_len=8 * 1024,
            head_dim=512,
            topk=2048,
            num_warmup=1,
            num_benchmark=3,
            triton_only=False,
        )
        
        print("\n提示: 运行 'python triton_fused_optimized.py profile' 进行详细性能分析")


# ============================================================================
# 性能分析：拆分 Kernel 各部分
# ============================================================================

@triton.jit
def _profile_load_only_kernel(
    K_ptr, Indices_ptr, Output_ptr,
    batch_size, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
):
    """Profile: 只测试数据加载"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    offs_topk = tl.arange(0, topk)
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    
    acc = tl.zeros([topk], dtype=tl.float32)
    num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
    for d_idx in range(num_d_blocks):
        d_start = d_idx * BLOCK_D
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < head_dim
        
        k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(k_gathered, axis=1)
    
    tl.store(Output_ptr + pid, tl.sum(acc))


@triton.jit
def _profile_qk_only_kernel(
    Q_ptr, K_ptr, Indices_ptr, Output_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr, scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
):
    """Profile: 测试 QK 计算（不含 softmax）"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    offs_topk = tl.arange(0, topk)
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    
    acc = tl.zeros([topk], dtype=tl.float32)
    
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        qk = tl.zeros([topk], dtype=tl.float32)
        
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        acc += qk * scaling
    
    tl.store(Output_ptr + pid, tl.sum(acc))


@triton.jit
def _profile_qk_softmax_kernel(
    Q_ptr, K_ptr, Indices_ptr, Output_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr, scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
):
    """Profile: 测试 QK + Softmax（不含 KL）"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    NEG_INF = -1e9
    
    offs_topk = tl.arange(0, topk)
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    global_query_pos = chunk_offset + pid_row
    causal_mask = indices > global_query_pos
    
    attn_sum = tl.zeros([topk], dtype=tl.float32)
    
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        qk = tl.zeros([topk], dtype=tl.float32)
        
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        qk = qk * scaling
        qk = tl.where(causal_mask, NEG_INF, qk)
        
        m = tl.max(qk)
        m = tl.where(m == NEG_INF, 0.0, m)
        p = tl.exp(qk - m)
        l = tl.sum(p)
        l = tl.where(l < 1e-9, 1.0, l)
        p = p / l
        p = tl.where(causal_mask, 0.0, p)
        
        attn_sum += p
    
    tl.store(Output_ptr + pid, tl.sum(attn_sum))


def profile_kernel_parts(
    batch_size=1, num_heads=16, chunk_size=4096, seq_len=8192,
    head_dim=512, topk=2048, num_warmup=3, num_iters=10
):
    """
    分步 Profile 各部分耗时
    
    测试内容:
    1. 只加载 K (gather 操作)
    2. QK 计算 (不含 softmax)
    3. QK + Softmax (不含 KL)
    4. 完整 kernel
    """
    import time
    
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    chunk_offset = seq_len - chunk_size
    
    print("=" * 70)
    print("Kernel 各部分耗时分析")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}")
    print(f"       seq={seq_len}, dim={head_dim}, topk={topk}")
    print("=" * 70)
    
    # 准备数据
    query = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, chunk_size, topk, device=device, dtype=torch.float32)
    
    # 生成 indices
    indices = torch.zeros(batch_size, chunk_size, topk, dtype=torch.int64, device=device)
    for t in range(chunk_size):
        global_pos = chunk_offset + t
        valid_range = min(global_pos + 1, seq_len)
        if valid_range >= topk:
            for b in range(batch_size):
                perm = torch.randperm(valid_range, device=device)[:topk]
                indices[b, t] = perm
        else:
            base = torch.arange(valid_range, device=device)
            extra = torch.randint(0, max(1, valid_range), (topk - valid_range,), device=device)
            indices[:, t] = torch.cat([base, extra]).unsqueeze(0).expand(batch_size, -1)
    
    output = torch.zeros(batch_size * chunk_size, device=device, dtype=torch.float32)
    grid = (batch_size * chunk_size,)
    
    results = {}
    
    # Test 1: 只加载 K
    print("\n[1] 只加载 K (gather 操作)...")
    for _ in range(num_warmup):
        _profile_load_only_kernel[grid](
            key, indices, output,
            batch_size, chunk_size, chunk_offset, head_dim, topk,
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _profile_load_only_kernel[grid](
            key, indices, output,
            batch_size, chunk_size, chunk_offset, head_dim, topk,
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['load_k'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['load_k']:.3f} ms")
    
    # Test 2: QK 计算
    print("\n[2] QK 计算 (不含 softmax)...")
    for _ in range(num_warmup):
        _profile_qk_only_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _profile_qk_only_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['qk_only'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['qk_only']:.3f} ms")
    
    # Test 3: QK + Softmax
    print("\n[3] QK + Softmax (不含 KL)...")
    for _ in range(num_warmup):
        _profile_qk_softmax_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _profile_qk_softmax_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['qk_softmax'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['qk_softmax']:.3f} ms")
    
    # Test 4: 完整 kernel
    print("\n[4] 完整 Kernel...")
    loss_per_row = torch.zeros(batch_size, chunk_size, device=device, dtype=torch.float32)
    for _ in range(num_warmup):
        _sparse_attn_loss_fused_kernel[grid](
            query, key, index_score, indices, loss_per_row,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, 1e-10,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _sparse_attn_loss_fused_kernel[grid](
            query, key, index_score, indices, loss_per_row,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, 1e-10,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['full'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['full']:.3f} ms")
    
    # 分析结果
    print("\n" + "=" * 70)
    print("耗时分解分析")
    print("=" * 70)
    print(f"  [1] 加载 K:                    {results['load_k']:.3f} ms ({results['load_k']/results['full']*100:.1f}%)")
    print(f"  [2] QK 计算 (含加载):          {results['qk_only']:.3f} ms ({results['qk_only']/results['full']*100:.1f}%)")
    print(f"  [3] QK + Softmax:              {results['qk_softmax']:.3f} ms ({results['qk_softmax']/results['full']*100:.1f}%)")
    print(f"  [4] 完整 Kernel:               {results['full']:.3f} ms (100%)")
    print("-" * 70)
    print(f"  增量: Softmax 开销:            {results['qk_softmax'] - results['qk_only']:.3f} ms")
    print(f"  增量: KL 开销:                 {results['full'] - results['qk_softmax']:.3f} ms")
    
    return results
