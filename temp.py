"""
Triton Fused Optimized - Sparse Attention (高性能版本)

核心优化:
1. 使用向量化的K gather - 预计算所有K的地址并批量load
2. 利用2D load pattern批量获取K[indices, :]
3. 使用向量化点积替代循环
4. 保持O(seq * topk * head_dim)复杂度，但常数因子大幅降低
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================================
# Kernel 1: Sparse Attention Softmax (向量化优化版本)
# ============================================================================

@triton.jit
def _sparse_attn_v5_kernel(
    Q_ptr, K_ptr, Indices_ptr, Out_ptr,
    batch_size, num_heads, seq_len,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    stride_ob, stride_oh, stride_os, stride_ok,
    BLOCK_D: tl.constexpr,
):
    """
    高效Sparse Attention Kernel V5
    
    关键优化:
    1. 使用2D指针批量load K[indices, :]
    2. 向量化的点积计算
    3. 减少循环开销
    """
    pid = tl.program_id(0)
    num_per_head = seq_len
    num_per_batch = num_heads * num_per_head
    
    pid_batch = pid // num_per_batch
    pid_temp = pid % num_per_batch
    pid_head = pid_temp // num_per_head
    pid_row = pid_temp % num_per_head
    
    NEG_INF = -1e9
    
    # 基地址计算
    q_base = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + pid_row * stride_qs
    k_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    out_base = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + pid_row * stride_os
    
    # 预加载所有indices [topk]
    offs_topk = tl.arange(0, topk)
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    
    # Causal mask
    causal_mask = indices > pid_row
    
    # 计算所有QK值 - 分块处理head_dim
    qk = tl.zeros([topk], dtype=tl.float32)
    
    for d_start in range(0, head_dim, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < head_dim
        
        # 加载Q chunk: [BLOCK_D]
        q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
        
        # 批量load K: 使用2D指针 [topk, BLOCK_D]
        # k_ptrs[i, j] = k_base + indices[i] * stride_ks + offs_d[j] * stride_kd
        k_ptrs = k_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
        
        # 向量化点积: q[d] * k_gathered[topk, d] -> sum over d -> [topk]
        qk += tl.sum(q[None, :] * k_gathered, axis=1)
    
    # 应用scaling和mask
    qk = qk * scaling
    qk = tl.where(causal_mask, NEG_INF, qk)
    
    # Softmax
    m = tl.max(qk)
    p = tl.exp(qk - m)
    l = tl.sum(p)
    l = tl.where(l < 1e-9, 1.0, l)
    p = p / l
    p = tl.where(causal_mask, 0.0, p)
    
    # 写出
    tl.store(out_base + offs_topk * stride_ok, p.to(Out_ptr.dtype.element_ty))


# ============================================================================
# Kernel 1 变体: 批量处理多行query (当BLOCK_M > 1时)
# ============================================================================

@triton.jit
def _sparse_attn_batched_kernel(
    Q_ptr, K_ptr, Indices_ptr, Out_ptr,
    batch_size, num_heads, seq_len,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    stride_ob, stride_oh, stride_os, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    批量处理多行query的Sparse Attention Kernel
    
    每个program处理BLOCK_M行query
    优化：虽然每行query的indices不同，但可以批量处理Q的load
    """
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    num_per_head = num_m_blocks
    num_per_batch = num_heads * num_per_head
    
    pid_batch = pid // num_per_batch
    pid_temp = pid % num_per_batch
    pid_head = pid_temp // num_per_head
    pid_m = pid_temp % num_per_head
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < seq_len
    
    NEG_INF = -1e9
    
    # 对每行query单独处理（因为indices不同）
    for mi in range(BLOCK_M):
        row = pid_m * BLOCK_M + mi
        if row >= seq_len:
            continue
            
        q_base = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + row * stride_qs
        k_base = K_ptr + pid_batch * stride_kb
        idx_base = Indices_ptr + pid_batch * stride_ib + row * stride_is
        out_base = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + row * stride_os
        
        # 加载indices
        offs_topk = tl.arange(0, topk)
        indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
        causal_mask = indices > row
        
        # 计算QK
        qk = tl.zeros([topk], dtype=tl.float32)
        
        for d_start in range(0, head_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs = k_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
            
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        qk = qk * scaling
        qk = tl.where(causal_mask, NEG_INF, qk)
        
        m = tl.max(qk)
        p = tl.exp(qk - m)
        l = tl.sum(p)
        l = tl.where(l < 1e-9, 1.0, l)
        p = p / l
        p = tl.where(causal_mask, 0.0, p)
        
        tl.store(out_base + offs_topk * stride_ok, p.to(Out_ptr.dtype.element_ty))


def sparse_attention_softmax_fused(query, key, indices, scaling):
    """Sparse Attention Softmax (优化版本)"""
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    
    query = query.contiguous()
    key = key.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    attn_scores = torch.empty(batch_size, num_heads, seq_len, topk,
                              device=query.device, dtype=query.dtype)
    
    # 选择block sizes - 需要topk是power of 2
    BLOCK_D = min(128, triton.next_power_of_2(head_dim))
    
    # 使用V5 kernel（每行一个program）
    grid = (batch_size * num_heads * seq_len,)
    
    _sparse_attn_v5_kernel[grid](
        query, key, indices, attn_scores,
        batch_size, num_heads, seq_len, head_dim, topk, scaling,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        BLOCK_D,
    )
    
    return attn_scores


# ============================================================================
# Kernel 2: Sparse Post-Reduce Loss (优化版本)
# ============================================================================

@triton.jit
def _sparse_loss_v2_kernel(
    AttnScores_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    batch_size, num_heads: tl.constexpr, seq_len,
    topk: tl.constexpr,
    eps: tl.constexpr,
    stride_ab, stride_ah, stride_as, stride_ak,
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
):
    """优化的Sparse Loss Kernel"""
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    NEG_INF = -1e9
    offs_k = tl.arange(0, topk)
    
    # 加载indices和causal mask
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    indices = tl.load(idx_base + offs_k * stride_ik).to(tl.int64)
    causal_mask = indices > pid_row
    
    # 累加所有heads的attention scores
    attn_sum = tl.zeros([topk], dtype=tl.float32)
    for h in tl.static_range(num_heads):
        attn_ptr = AttnScores_ptr + pid_batch * stride_ab + h * stride_ah + \
                   pid_row * stride_as + offs_k * stride_ak
        attn = tl.load(attn_ptr)
        attn_sum += attn
    
    # 归一化得到attn分布
    attn_total = tl.sum(attn_sum)
    attn_dist = attn_sum / (attn_total + eps) + eps
    
    # 计算index_score的softmax
    is_ptr = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
    is_val = tl.load(is_ptr)
    is_val = tl.where(causal_mask, NEG_INF, is_val)
    
    m_is = tl.max(is_val)
    p_is = tl.exp(is_val - m_is)
    s_is = tl.sum(p_is)
    s_is = tl.where(s_is < 1e-9, 1.0, s_is)
    index_prob = p_is / s_is + eps
    
    # KL散度
    kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
    kl = tl.where(causal_mask, 0.0, kl)
    kl_sum = tl.sum(kl)
    
    tl.store(Loss_ptr + pid_batch * seq_len + pid_row, kl_sum)


def sparse_post_reduce_loss_fused(attn_scores, index_score, indices, eps=1e-10):
    """Sparse Post-Reduce Loss (优化版本)"""
    batch_size, num_heads, seq_len, topk = attn_scores.shape
    
    attn_scores = attn_scores.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    loss_per_row = torch.zeros(batch_size, seq_len, device=attn_scores.device, dtype=torch.float32)
    
    grid = (batch_size * seq_len,)
    
    _sparse_loss_v2_kernel[grid](
        attn_scores, index_score, indices, loss_per_row,
        batch_size, num_heads, seq_len, topk, eps,
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
    )
    
    return loss_per_row.sum() / batch_size


# ============================================================================
# 组合函数
# ============================================================================

def compute_index_loss_sparse(query, key, index_score, indices, scaling):
    """Sparse版本的完整loss计算"""
    attn_scores = sparse_attention_softmax_fused(query, key, indices, scaling)
    loss = sparse_post_reduce_loss_fused(attn_scores, index_score, indices)
    return loss


# ============================================================================
# PyTorch参考实现
# ============================================================================

def pytorch_reference_sparse(query, key, index_score, indices, scaling):
    """Sparse版本的PyTorch参考实现"""
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    eps = 1e-10
    
    # Gather K: indices扩展到[batch, seq_len, topk, head_dim]
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    # key: [batch, seq_len, head_dim] -> expand to [batch, seq_len, topk, head_dim]
    key_expanded = key.unsqueeze(2).expand(-1, -1, topk, -1)
    # gather: K[batch, indices[batch, row, :], :] for each row
    k_gathered = torch.gather(key_expanded, dim=1, index=indices_expanded)
    
    # 计算attention: query @ k_gathered.T
    # query: [batch, heads, seq, head_dim]
    # k_gathered: [batch, seq, topk, head_dim]
    # attn: [batch, heads, seq, topk]
    attn = torch.einsum('bhsd,bstd->bhst', query, k_gathered) * scaling
    
    # Causal mask
    row_indices = torch.arange(seq_len, device=query.device).view(1, 1, -1, 1)
    causal_mask = indices.unsqueeze(1) > row_indices
    attn = attn.masked_fill(causal_mask, -1e9)
    
    # Softmax
    attn = torch.softmax(attn, dim=-1)
    
    # Head sum + normalize
    attn_sum = attn.sum(dim=1)
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    # Index score softmax
    causal_mask_2d = indices > torch.arange(seq_len, device=query.device).view(1, -1, 1)
    index_score_masked = index_score.masked_fill(causal_mask_2d, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    # KL
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


# ============================================================================
# 测试辅助函数
# ============================================================================

def generate_topk_indices(batch_size, seq_len, topk, device='cuda'):
    """生成causal topk indices"""
    indices = torch.zeros(batch_size, seq_len, topk, dtype=torch.int64, device=device)
    
    for t in range(seq_len):
        valid_range = t + 1
        if valid_range >= topk:
            for b in range(batch_size):
                perm = torch.randperm(valid_range, device=device)[:topk]
                indices[b, t] = perm
        else:
            base = torch.arange(valid_range, device=device)
            if valid_range < topk:
                extra = torch.randint(0, max(1, valid_range), (topk - valid_range,), device=device)
                indices[:, t] = torch.cat([base, extra]).unsqueeze(0).expand(batch_size, -1)
            else:
                indices[:, t] = base.unsqueeze(0).expand(batch_size, -1)
    
    return indices


# ============================================================================
# 测试函数
# ============================================================================

def test_kernel1_accuracy(batch_size=1, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试Kernel 1精度"""
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    # Triton
    tri_attn = sparse_attention_softmax_fused(query, key, indices, scaling)
    
    # PyTorch reference
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    key_expanded = key.unsqueeze(2).expand(-1, -1, topk, -1)
    k_gathered = torch.gather(key_expanded, dim=1, index=indices_expanded)
    ref_attn = torch.einsum('bhsd,bstd->bhst', query, k_gathered) * scaling
    
    row_indices = torch.arange(seq_len, device=device).view(1, 1, -1, 1)
    causal_mask = indices.unsqueeze(1) > row_indices
    ref_attn = ref_attn.masked_fill(causal_mask, -1e9)
    ref_attn = torch.softmax(ref_attn, dim=-1)
    
    diff = (tri_attn - ref_attn).abs().max().item()
    print(f"Kernel 1 Accuracy - Max diff: {diff:.6e}, Pass: {diff < 1e-4}")
    return diff < 1e-4


def test_full_accuracy(batch_size=1, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试完整流程精度"""
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    ref = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    tri = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    
    diff = abs(ref.item() - tri.item())
    print(f"Full Accuracy - Ref: {ref.item():.6f}, Triton: {tri.item():.6f}, Diff: {diff:.6e}")
    return diff < 1e-3


def test_performance(
    batch_size: int = 1,
    num_heads: int = 16,
    seq_len: int = 4096,
    head_dim: int = 256,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 5,
    num_benchmark: int = 20,
):
    """性能测试"""
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("=" * 70)
    print("Sparse Triton Optimized 性能测试")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print(f"理论复杂度: O(seq * topk * head_dim) = O({seq_len * topk * head_dim:,})")
    print("=" * 70)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    # Warmup
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    
    # Warmup PyTorch
    for _ in range(num_warmup):
        _ = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_benchmark):
        _ = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_benchmark * 1000
    
    print(f"\n  PyTorch sparse ref: {pytorch_time:.3f} ms")
    print(f"  Triton optimized: {triton_time:.3f} ms")
    if triton_time > 0:
        print(f"  加速比: {pytorch_time / triton_time:.2f}x")
    
    return {'pytorch_time_ms': pytorch_time, 'triton_time_ms': triton_time}


def test_scaling(topk=512):
    """测试线性scaling"""
    print("\n" + "=" * 70)
    print(f"线性Scaling测试 (固定topk={topk})")
    print("=" * 70)
    
    results = []
    for seq_len in [1024, 2048, 4096, 8192]:
        try:
            r = test_performance(seq_len=seq_len, topk=topk, head_dim=128, num_heads=8, 
                                num_warmup=3, num_benchmark=10)
            results.append({'seq_len': seq_len, **r})
        except Exception as e:
            print(f"seq={seq_len} 失败: {e}")
    
    print("\n汇总:")
    if results:
        base = results[0]['triton_time_ms']
        for r in results:
            ratio = r['triton_time_ms'] / base
            expected = r['seq_len'] / results[0]['seq_len']
            print(f"  seq={r['seq_len']}: {r['triton_time_ms']:.3f}ms (实际{ratio:.2f}x, 预期{expected:.2f}x)")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("精度测试")
    print("=" * 70)
    
    print("\n[小规模测试]")
    test_kernel1_accuracy(batch_size=1, num_heads=4, seq_len=64, head_dim=32, topk=16)
    test_full_accuracy(batch_size=1, num_heads=4, seq_len=64, head_dim=32, topk=16)
    
    print("\n[中等规模测试]")
    test_kernel1_accuracy(batch_size=1, num_heads=8, seq_len=256, head_dim=64, topk=64)
    test_full_accuracy(batch_size=1, num_heads=8, seq_len=256, head_dim=64, topk=64)
    
    print("\n[大规模测试]")
    test_kernel1_accuracy(batch_size=1, num_heads=16, seq_len=1024, head_dim=128, topk=256)
    test_full_accuracy(batch_size=1, num_heads=16, seq_len=1024, head_dim=128, topk=256)
    
    print("\n")
    test_performance(
        batch_size=1,
        num_heads=16,
        seq_len=4096,
        head_dim=256,
        topk=512,
    )
    
    test_scaling(topk=512)
