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
# Fused Kernel: Sparse Attention + Loss (完全融合版本)
# ============================================================================

@triton.jit
def _sparse_attn_loss_fused_kernel(
    Q_ptr, K_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    batch_size, num_heads: tl.constexpr, seq_len,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
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
    完全融合的 Sparse Attention + Loss Kernel
    
    单个kernel完成:
    1. 对每个head计算sparse attention softmax
    2. 累加所有heads的attention scores
    3. 归一化得到attention分布
    4. 计算index_score的softmax
    5. 计算KL散度loss
    
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
    
    # 加载indices [topk]
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    
    # Causal mask: indices > current_row 的位置需要mask
    causal_mask = indices > pid_row
    
    # =========================================================================
    # Part 1: 累加所有heads的attention scores
    # =========================================================================
    attn_sum = tl.zeros([topk], dtype=tl.float32)
    
    # 对每个head循环
    for h in tl.static_range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        # 计算QK^T - 分块处理head_dim
        qk = tl.zeros([topk], dtype=tl.float32)
        
        num_d_blocks = (head_dim + BLOCK_D - 1) // BLOCK_D
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            # 加载Q chunk: [BLOCK_D]
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            
            # 批量load K: [topk, BLOCK_D]
            k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
            
            # 向量化点积: q[d] * k_gathered[topk, d] -> sum over d -> [topk]
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        # 应用scaling和causal mask
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
        
        # 累加到attn_sum
        attn_sum += p
    
    # =========================================================================
    # Part 2: 归一化attention分布
    # =========================================================================
    attn_total = tl.sum(attn_sum)
    attn_total = tl.where(attn_total < eps, 1.0, attn_total)
    attn_dist = attn_sum / attn_total + eps
    
    # =========================================================================
    # Part 3: 计算index_score的softmax
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
    # Part 4: 计算KL散度
    # =========================================================================
    kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
    kl = tl.where(causal_mask, 0.0, kl)
    kl_sum = tl.sum(kl)
    
    # 写出loss
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
# PyTorch参考实现
# ============================================================================

def pytorch_reference_sparse(query, key, index_score, indices, scaling):
    """Sparse版本的PyTorch参考实现"""
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    eps = 1e-10
    
    # Gather K
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    key_expanded = key.unsqueeze(2).expand(-1, -1, topk, -1)
    k_gathered = torch.gather(key_expanded, dim=1, index=indices_expanded)
    
    # 计算attention
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


def pytorch_reference(query, key, index_score, index_mask, scaling):
    """完整的PyTorch参考实现"""
    eps = 1e-10
    
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(index_mask.unsqueeze(1), -1e9)
    attn = torch.softmax(attn, dim=-1)
    
    attn_sum = attn.sum(dim=1)
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    index_score_masked = index_score.masked_fill(index_mask, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


# ============================================================================
# 测试辅助函数
# ============================================================================

def generate_topk_indices(batch_size, seq_len, topk, device='cuda'):
    """生成causal topk indices (向量化版本)"""
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
    passed = diff < 1e-3
    print(f"Accuracy - Ref: {ref.item():.6f}, Triton: {tri.item():.6f}, Diff: {diff:.6e}, Pass: {passed}")
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
    print("Sparse Triton Fused Kernel 性能测试")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print(f"理论复杂度: O(seq * topk * head_dim * num_heads) = O({seq_len * topk * head_dim * num_heads:,})")
    print("=" * 70)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    results = {}
    
    # Test 1: Triton fused kernel
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    results['triton_fused'] = triton_time
    
    # Test 2: PyTorch reference
    for _ in range(num_warmup):
        _ = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_benchmark * 1000
    results['pytorch'] = pytorch_time
    
    print(f"\n>>> 性能结果 (warmup={num_warmup}, iters={num_benchmark})")
    print(f"  PyTorch sparse ref:    {pytorch_time:.3f} ms")
    print(f"  Triton fused:          {triton_time:.3f} ms (加速: {pytorch_time/triton_time:.2f}x)")
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("精度测试")
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
        seq_len=4096,
        head_dim=256,
        topk=512,
        num_warmup=2,
        num_benchmark=3,
    )
