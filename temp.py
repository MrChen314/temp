"""
Triton Fused实现 - 支持Tensor Parallel

将计算分为两个融合kernel以支持中间的reduce操作:
1. Kernel 1 (Pre-Reduce): Attention Softmax
2. Kernel 2 (Post-Reduce): HeadSum + Normalize + IndexSoftmax + KL Divergence

注意: reduce操作在外部进行，不在此文件中实现
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================================
# Kernel 1: Pre-Reduce - Attention Softmax
# 输入: query [B, H, S, D], key [B, S, D], mask [B, S, S]
# 输出: attn_scores [B, H, S, S]
# ============================================================================

@triton.jit
def _attention_softmax_kernel(
    Q_ptr,          # [batch, num_heads, seq_len, head_dim]
    K_ptr,          # [batch, seq_len, head_dim]
    Mask_ptr,       # [batch, seq_len, seq_len]
    Out_ptr,        # [batch, num_heads, seq_len, seq_len]
    # Dimensions
    batch_size,
    num_heads,
    seq_len,
    head_dim: tl.constexpr,
    scaling,
    # Strides for Q
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K
    stride_kb, stride_ks, stride_kd,
    # Strides for Mask
    stride_mb, stride_ms, stride_mk,
    # Strides for Out
    stride_ob, stride_oh, stride_os, stride_ok,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Pre-Reduce Kernel: 计算masked attention softmax
    每个program处理一个 (batch, head, row)
    使用online softmax算法
    """
    pid = tl.program_id(0)
    num_rows_per_batch = num_heads * seq_len
    pid_batch = pid // num_rows_per_batch
    remainder = pid % num_rows_per_batch
    pid_head = remainder // seq_len
    pid_row = remainder % seq_len
    
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    
    # 加载query: [head_dim]
    q_ptr = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + pid_row * stride_qs
    q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
    
    # Pass 1: 计算max (for numerical stability)
    m_max = -float("inf")
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # 加载key: [BLOCK_N, head_dim]
        k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        
        # QK^T * scaling
        qk = tl.sum(q[None, :] * k, axis=1) * scaling
        
        # 加载mask并应用
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
        
        m_max = tl.maximum(m_max, tl.max(qk))
    
    # Pass 2: 计算sum(exp(x - max))
    l_sum = 0.0
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        
        qk = tl.sum(q[None, :] * k, axis=1) * scaling
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
        
        l_sum += tl.sum(tl.exp(qk - m_max))
    
    # Pass 3: 计算softmax并写出
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        
        qk = tl.sum(q[None, :] * k, axis=1) * scaling
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
        
        # softmax
        p = tl.exp(qk - m_max) / l_sum
        
        # 写出
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + pid_row * stride_os + offs_n * stride_ok
        tl.store(out_ptrs, p, mask=n_mask)


def attention_softmax_fused(query, key, mask, scaling):
    """
    Kernel 1: Pre-Reduce - 计算masked attention softmax
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, seq_len, head_dim]
        mask: [batch, seq_len, seq_len] bool, True表示mask掉
        scaling: float
    
    Returns:
        attn_scores: [batch, num_heads, seq_len, seq_len]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    query = query.contiguous()
    key = key.contiguous()
    mask = mask.contiguous()
    
    attn_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len, 
                              device=query.device, dtype=query.dtype)
    
    BLOCK_N = min(1024, triton.next_power_of_2(seq_len))
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    grid = (batch_size * num_heads * seq_len,)
    
    _attention_softmax_kernel[grid](
        query, key, mask, attn_scores,
        batch_size, num_heads, seq_len, head_dim, scaling,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        BLOCK_N, BLOCK_D,
    )
    
    return attn_scores


# ============================================================================
# Kernel 2: Post-Reduce - HeadSum + Normalize + IndexSoftmax + KL Divergence
# 输入: attn_scores [B, H, S, S], index_score [B, S, S], mask [B, S, S]
# 输出: loss [B, S] (每行的KL loss，后续求mean)
# ============================================================================

@triton.jit
def _post_reduce_loss_kernel(
    AttnScores_ptr, # [batch, num_heads, seq_len, seq_len] - reduced attention scores
    IndexScore_ptr, # [batch, seq_len, seq_len]
    Mask_ptr,       # [batch, seq_len, seq_len]
    Loss_ptr,       # [batch, seq_len] - 每行的loss
    # Dimensions
    batch_size,
    num_heads: tl.constexpr,
    seq_len,
    eps: tl.constexpr,
    # Strides for AttnScores
    stride_ab, stride_ah, stride_as, stride_ak,
    # Strides for IndexScore
    stride_isb, stride_iss, stride_isk,
    # Strides for Mask
    stride_mb, stride_ms, stride_mk,
    # Block size
    BLOCK_K: tl.constexpr,
):
    """
    Post-Reduce Kernel: 融合 HeadSum + Normalize + IndexSoftmax + KL
    每个program处理一个 (batch, row)
    
    计算流程:
    1. 计算 attn_sum = sum(attn_scores, dim=heads)
    2. 计算 attn_dist = attn_sum / sum(attn_sum)
    3. 计算 index_prob = softmax(index_score.masked_fill(mask, -inf)) + eps
    4. 计算 kl = sum(attn_dist * (log(attn_dist) - log(index_prob)))
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    # ============ Pass 1: 计算 attn_sum 的总和 (用于归一化) ============
    attn_total = 0.0
    
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        # 对所有heads求和
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h in tl.static_range(num_heads):
            attn_ptrs = AttnScores_ptr + pid_batch * stride_ab + h * stride_ah + pid_row * stride_as + offs_k * stride_ak
            attn_val = tl.load(attn_ptrs, mask=k_mask, other=0.0)
            acc += attn_val
        
        attn_total += tl.sum(tl.where(k_mask, acc, 0.0))
    
    # ============ Pass 2: 计算 index_score 的 max (用于stable softmax) ============
    max_is = -float("inf")
    
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        mask_val = tl.load(mask_ptrs, mask=k_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        max_is = tl.maximum(max_is, tl.max(is_val))
    
    # ============ Pass 3: 计算 index_score 的 sum(exp) ============
    sum_is = 0.0
    
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        mask_val = tl.load(mask_ptrs, mask=k_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        sum_is += tl.sum(tl.exp(is_val - max_is))
    
    # ============ Pass 4: 计算 KL 散度 ============
    kl_sum = 0.0
    
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        # 计算 attn_sum 并归一化得到 attn_dist
        attn_sum = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h in tl.static_range(num_heads):
            attn_ptrs = AttnScores_ptr + pid_batch * stride_ab + h * stride_ah + pid_row * stride_as + offs_k * stride_ak
            attn_val = tl.load(attn_ptrs, mask=k_mask, other=0.0)
            attn_sum += attn_val
        
        attn_dist = attn_sum / (attn_total + eps) + eps
        
        # 计算 index_prob
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        mask_val = tl.load(mask_ptrs, mask=k_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        index_prob = tl.exp(is_val - max_is) / sum_is + eps
        
        # KL散度: attn_dist * (log(attn_dist) - log(index_prob))
        kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
        kl = tl.where(k_mask, kl, 0.0)
        
        kl_sum += tl.sum(kl)
    
    # 存储结果
    tl.store(Loss_ptr + pid, kl_sum)


def post_reduce_loss_fused(attn_scores, index_score, mask, eps=1e-10):
    """
    Kernel 2: Post-Reduce - 融合计算 HeadSum + Normalize + IndexSoftmax + KL
    
    Args:
        attn_scores: [batch, num_heads, seq_len, seq_len] - reduced attention scores
        index_score: [batch, seq_len, seq_len]
        mask: [batch, seq_len, seq_len] bool, True表示mask掉
        eps: 数值稳定性
    
    Returns:
        loss: scalar (batchmean)
    """
    batch_size, num_heads, seq_len, _ = attn_scores.shape
    
    attn_scores = attn_scores.contiguous()
    index_score = index_score.contiguous()
    mask = mask.contiguous()
    
    loss_per_row = torch.zeros(batch_size * seq_len, device=attn_scores.device, dtype=torch.float32)
    
    BLOCK_K = min(1024, triton.next_power_of_2(seq_len))
    
    grid = (batch_size * seq_len,)
    
    _post_reduce_loss_kernel[grid](
        attn_scores, index_score, mask, loss_per_row,
        batch_size, num_heads, seq_len, eps,
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        BLOCK_K,
    )
    
    # batchmean reduction
    return loss_per_row.sum() / batch_size


# ============================================================================
# 组合函数 (不含reduce操作)
# ============================================================================

def compute_index_loss_fused(query, key, index_score, index_mask, scaling):
    """
    Fused版本: 两个kernel串联 (不含reduce)
    
    实际使用时，在两个kernel之间插入reduce操作:
        attn_scores = attention_softmax_fused(query, key, index_mask, scaling)
        attn_scores = reduce_from_tensor_model_parallel_region(attn_scores)  # 外部操作
        loss = post_reduce_loss_fused(attn_scores, index_score, index_mask)
    """
    # Kernel 1: Pre-Reduce
    attn_scores = attention_softmax_fused(query, key, index_mask, scaling)
    
    # (这里应该插入reduce操作，但本文件不实现)
    
    # Kernel 2: Post-Reduce
    loss = post_reduce_loss_fused(attn_scores, index_score, index_mask)
    
    return loss


# ============================================================================
# PyTorch参考实现
# ============================================================================

def pytorch_reference(query, key, index_score, index_mask, scaling):
    """完整的PyTorch参考实现"""
    eps = 1e-10
    
    # Kernel 1 equivalent
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(index_mask.unsqueeze(1), -1e9)
    attn = torch.softmax(attn, dim=-1)
    
    # (reduce would happen here in distributed setting)
    
    # Kernel 2 equivalent
    attn_sum = attn.sum(dim=1)
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    index_score_masked = index_score.masked_fill(index_mask, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


def attention_softmax_ref(query, key, mask, scaling):
    """Kernel 1的PyTorch参考"""
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(mask.unsqueeze(1), -1e9)
    attn = torch.softmax(attn, dim=-1)
    return attn


def post_reduce_loss_ref(attn_scores, index_score, mask, eps=1e-10):
    """Kernel 2的PyTorch参考"""
    # Head sum + normalize
    attn_sum = attn_scores.sum(dim=1)
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    # Index score softmax
    index_score_masked = index_score.masked_fill(mask, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    # KL divergence
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


# ============================================================================
# 测试函数
# ============================================================================

def generate_topk_mask(batch_size, seq_len, topk, device='cuda'):
    """生成基于topk的index mask"""
    t_vals = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len, 1)
    max_vals = torch.clamp(t_vals + 1, min=1).expand(batch_size, seq_len, topk)
    random_floats = torch.rand(batch_size, seq_len, topk, device=device)
    indices = (random_floats * max_vals).to(torch.int64)
    
    mask = torch.ones(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, seq_len, topk)
    row_idx = torch.arange(seq_len, device=device).view(1, -1, 1).expand(batch_size, -1, topk)
    mask[batch_idx, row_idx, indices] = False
    
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = mask | causal_mask.unsqueeze(0)
    
    return mask, indices


def test_kernel1(batch_size=2, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试 Kernel 1: Attention Softmax"""
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    mask, _ = generate_topk_mask(batch_size, seq_len, topk, device)
    
    ref = attention_softmax_ref(query, key, mask, scaling)
    tri = attention_softmax_fused(query, key, mask, scaling)
    
    diff = (ref - tri).abs().max().item()
    print(f"Kernel 1 (Attention Softmax):")
    print(f"  Max diff: {diff:.6e}")
    print(f"  Pass: {diff < 1e-4}")
    
    return diff < 1e-4


def test_kernel2(batch_size=2, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试 Kernel 2: Post-Reduce Loss"""
    torch.manual_seed(seed)
    device = 'cuda'
    
    # 生成模拟的attention scores (已经过softmax)
    attn_scores = torch.rand(batch_size, num_heads, seq_len, seq_len, device=device, dtype=torch.float32)
    attn_scores = attn_scores / attn_scores.sum(dim=-1, keepdim=True)
    
    index_score = torch.randn(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    mask, _ = generate_topk_mask(batch_size, seq_len, topk, device)
    
    ref = post_reduce_loss_ref(attn_scores, index_score, mask)
    tri = post_reduce_loss_fused(attn_scores, index_score, mask)
    
    diff = abs(ref.item() - tri.item())
    print(f"Kernel 2 (Post-Reduce Loss):")
    print(f"  Ref: {ref.item():.6f}, Triton: {tri.item():.6f}")
    print(f"  Diff: {diff:.6e}")
    print(f"  Pass: {diff < 1e-4}")
    
    return diff < 1e-4


def test_fused(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 256,
    head_dim: int = 64,
    topk: int = 32,
    seed: int = 42,
    num_warmup: int = 10,
    num_benchmark: int = 100,
):
    """
    测试Fused实现的精度和性能
    """
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("=" * 60)
    print("Triton Fused 测试")
    print("=" * 60)
    print(f"参数: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print("=" * 60)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    mask, _ = generate_topk_mask(batch_size, seq_len, topk, device)
    
    # ============ 精度测试 ============
    print("\n>>> 精度测试")
    
    # 测试 Kernel 1
    print("\n[Kernel 1: Attention Softmax]")
    attn_ref = attention_softmax_ref(query, key, mask, scaling)
    attn_tri = attention_softmax_fused(query, key, mask, scaling)
    diff1 = (attn_ref - attn_tri).abs().max().item()
    print(f"  Max diff: {diff1:.6e}")
    
    # 测试 Kernel 2 (使用相同的attention输入)
    print("\n[Kernel 2: Post-Reduce Loss]")
    loss_ref = post_reduce_loss_ref(attn_ref, index_score, mask)
    loss_tri = post_reduce_loss_fused(attn_tri, index_score, mask)
    diff2 = abs(loss_ref.item() - loss_tri.item())
    print(f"  Ref: {loss_ref.item():.6f}, Triton: {loss_tri.item():.6f}")
    print(f"  Diff: {diff2:.6e}")
    
    # 完整流程
    print("\n[完整流程]")
    full_ref = pytorch_reference(query, key, index_score, mask, scaling)
    full_tri = compute_index_loss_fused(query, key, index_score, mask, scaling)
    diff_full = abs(full_ref.item() - full_tri.item())
    print(f"  PyTorch: {full_ref.item():.6f}")
    print(f"  Triton Fused: {full_tri.item():.6f}")
    print(f"  Diff: {diff_full:.6e}")
    
    # ============ 性能测试 ============
    print(f"\n>>> 性能测试 (warmup={num_warmup}, iterations={num_benchmark})")
    
    # Warmup Triton
    for _ in range(num_warmup):
        _ = compute_index_loss_fused(query, key, index_score, mask, scaling)
    torch.cuda.synchronize()
    
    # Benchmark Triton Fused
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_fused(query, key, index_score, mask, scaling)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    
    # Warmup PyTorch
    for _ in range(num_warmup):
        _ = pytorch_reference(query, key, index_score, mask, scaling)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_benchmark):
        _ = pytorch_reference(query, key, index_score, mask, scaling)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_benchmark * 1000
    
    print(f"\n  PyTorch: {pytorch_time:.3f} ms")
    print(f"  Triton Fused: {triton_time:.3f} ms")
    if triton_time > 0:
        print(f"  加速比: {pytorch_time / triton_time:.2f}x")
    
    # ============ 内存估算 ============
    print(f"\n>>> 内存使用")
    attn_size = batch_size * num_heads * seq_len * seq_len * 4 / 1024 / 1024
    print(f"  Attention矩阵: {attn_size:.2f} MB")
    print(f"  PyTorch: 需要存储完整QK^T中间矩阵")
    print(f"  Triton: 使用online softmax，无需存储完整QK^T")
    
    return {
        'kernel1_diff': diff1,
        'kernel2_diff': diff2,
        'full_diff': diff_full,
        'pytorch_time_ms': pytorch_time,
        'triton_time_ms': triton_time,
        'speedup': pytorch_time / triton_time if triton_time > 0 else float('inf'),
    }


def test_various_configs():
    """测试多种配置"""
    configs = [
        (1, 8, 128, 64, 16),
        (1, 8, 256, 64, 32),
        (2, 8, 256, 64, 32),
        (1, 8, 512, 64, 64),
        (1, 16, 256, 64, 32),
    ]
    
    print("\n" + "=" * 80)
    print("多配置性能测试")
    print("=" * 80)
    
    results = []
    for batch_size, num_heads, seq_len, head_dim, topk in configs:
        try:
            result = test_fused(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                topk=topk,
                num_warmup=5,
                num_benchmark=50,
            )
            results.append({
                'config': (batch_size, num_heads, seq_len, head_dim, topk),
                **result
            })
        except Exception as e:
            print(f"配置 {(batch_size, num_heads, seq_len, head_dim, topk)} 失败: {e}")
    
    # 汇总
    print("\n" + "=" * 80)
    print("性能汇总")
    print("=" * 80)
    print(f"{'Config':<30} {'PyTorch(ms)':<12} {'Triton(ms)':<12} {'Speedup':<10} {'Precision':<10}")
    print("-" * 80)
    for r in results:
        config_str = str(r['config'])
        precision = "PASS" if r['full_diff'] < 1e-4 else "FAIL"
        print(f"{config_str:<30} {r['pytorch_time_ms']:<12.3f} {r['triton_time_ms']:<12.3f} {r['speedup']:<10.2f}x {precision:<10}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("单独测试 Kernel 1")
    print("=" * 60)
    test_kernel1()
    
    print("\n" + "=" * 60)
    print("单独测试 Kernel 2")
    print("=" * 60)
    test_kernel2()
    
    print("\n")
    test_fused(
        batch_size=2,
        num_heads=8,
        seq_len=256,
        head_dim=64,
        topk=32,
    )

