"""
Triton Fused实现 - 支持Tensor Parallel (高性能版本)

优化策略:
1. 使用Online Softmax算法，减少遍历次数
2. 利用tl.dot进行高效矩阵乘法
3. Tile-based处理，最大化数据复用
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================================
# Kernel 1: Pre-Reduce - Attention Softmax (高性能版本)
# ============================================================================

@triton.jit
def _attention_softmax_kernel_v2(
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
    # Strides for Q: [batch, num_heads, seq_len, head_dim]
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K: [batch, seq_len, head_dim]
    stride_kb, stride_ks, stride_kd,
    # Strides for Mask: [batch, seq_len, seq_len]
    stride_mb, stride_ms, stride_mk,
    # Strides for Out: [batch, num_heads, seq_len, seq_len]
    stride_ob, stride_oh, stride_os, stride_ok,
    # Block sizes
    BLOCK_M: tl.constexpr,  # 每个program处理的query行数
    BLOCK_N: tl.constexpr,  # key的tile大小
    BLOCK_D: tl.constexpr,  # head_dim的block大小
):
    """
    高性能Attention Softmax Kernel (使用循环处理head_dim)
    """
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    num_programs_per_head = num_m_blocks
    num_programs_per_batch = num_heads * num_programs_per_head
    
    pid_batch = pid // num_programs_per_batch
    pid_remainder = pid % num_programs_per_batch
    pid_head = pid_remainder // num_programs_per_head
    pid_m = pid_remainder % num_programs_per_head
    
    # Query行的偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < seq_len
    
    # 初始化online softmax状态
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Pass 1: 计算每行的max和sum
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # 分块计算QK^T (处理大head_dim)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for start_d in range(0, head_dim, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            # 加载Q块: [BLOCK_M, BLOCK_D]
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # 加载K块: [BLOCK_N, BLOCK_D]
            k_ptrs = K_ptr + pid_batch * stride_kb + \
                     offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # 累加 QK^T
            qk += tl.dot(q, tl.trans(k))
        
        qk = qk * scaling
        
        # 加载Mask: [BLOCK_M, BLOCK_N]
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + \
                    offs_m[:, None] * stride_ms + offs_n[None, :] * stride_mk
        mask_load = m_mask[:, None] & n_mask[None, :]
        mask_val = tl.load(mask_ptrs, mask=mask_load, other=True)
        
        # 应用mask和边界
        qk = tl.where(mask_val | ~n_mask[None, :] | ~m_mask[:, None], -float("inf"), qk)
        
        # Online softmax更新
        m_ij = tl.max(qk, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)
        
        # 避免exp溢出：当m_i是-inf时，alpha应该是0
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_new))
        
        # 计算exp(qk - m_new)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
        
        # 更新l_i
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    
    # 处理l_i为0的情况（整行被mask）
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    
    # Pass 2: 计算softmax并写出
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # 分块计算QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for start_d in range(0, head_dim, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            k_ptrs = K_ptr + pid_batch * stride_kb + \
                     offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            qk += tl.dot(q, tl.trans(k))
        
        qk = qk * scaling
        
        # 应用Mask
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + \
                    offs_m[:, None] * stride_ms + offs_n[None, :] * stride_mk
        mask_load = m_mask[:, None] & n_mask[None, :]
        mask_val = tl.load(mask_ptrs, mask=mask_load, other=True)
        qk = tl.where(mask_val | ~n_mask[None, :] | ~m_mask[:, None], -float("inf"), qk)
        
        # Softmax
        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
        p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
        
        # 写出
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + \
                   offs_m[:, None] * stride_os + offs_n[None, :] * stride_ok
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptrs, p.to(Out_ptr.dtype.element_ty), mask=out_mask)


def attention_softmax_fused(query, key, mask, scaling):
    """
    Kernel 1: Pre-Reduce - 高性能masked attention softmax
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    query = query.contiguous()
    key = key.contiguous()
    mask = mask.contiguous()
    
    attn_scores = torch.empty(batch_size, num_heads, seq_len, seq_len, 
                              device=query.device, dtype=query.dtype)
    
    # 选择合适的block size，控制shared memory使用
    # BLOCK_D限制在64-128，大的head_dim通过循环处理
    BLOCK_D = min(64, triton.next_power_of_2(head_dim))
    
    # 根据seq_len和head_dim调整BLOCK_M和BLOCK_N
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 64
    else:
        # 大head_dim时使用更小的block
        BLOCK_M = 32
        BLOCK_N = 32
    
    # 根据seq_len微调
    if seq_len >= 4096:
        BLOCK_M = min(BLOCK_M, 64)
        BLOCK_N = min(BLOCK_N, 64)
    
    num_m_blocks = triton.cdiv(seq_len, BLOCK_M)
    grid = (batch_size * num_heads * num_m_blocks,)
    
    _attention_softmax_kernel_v2[grid](
        query, key, mask, attn_scores,
        batch_size, num_heads, seq_len, head_dim, scaling,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        BLOCK_M, BLOCK_N, BLOCK_D,
    )
    
    return attn_scores


# ============================================================================
# Kernel 2: Post-Reduce - HeadSum + Normalize + IndexSoftmax + KL (高性能版本)
# ============================================================================

@triton.jit
def _post_reduce_loss_kernel_v2(
    AttnScores_ptr, # [batch, num_heads, seq_len, seq_len]
    IndexScore_ptr, # [batch, seq_len, seq_len]
    Mask_ptr,       # [batch, seq_len, seq_len]
    Loss_ptr,       # [batch, seq_len]
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
    # Block sizes
    BLOCK_M: tl.constexpr,  # 每个program处理的行数
    BLOCK_K: tl.constexpr,  # key维度的tile大小
):
    """
    高性能Post-Reduce Loss Kernel
    
    优化:
    1. 每个program处理BLOCK_M行
    2. 合并计算pass
    3. 减少global memory访问
    """
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    
    pid_batch = pid // num_m_blocks
    pid_m = pid % num_m_blocks
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < seq_len
    
    # 初始化累加器 [BLOCK_M]
    attn_total = tl.zeros([BLOCK_M], dtype=tl.float32)
    max_is = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    sum_is = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Pass 1: 计算attn_total, max_is, sum_is
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        mk_mask = m_mask[:, None] & k_mask[None, :]
        
        # 累加attention scores (对所有heads求和)
        attn_sum_block = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for h in tl.static_range(num_heads):
            attn_ptrs = AttnScores_ptr + pid_batch * stride_ab + h * stride_ah + \
                        offs_m[:, None] * stride_as + offs_k[None, :] * stride_ak
            attn_val = tl.load(attn_ptrs, mask=mk_mask, other=0.0)
            attn_sum_block += attn_val
        
        attn_total += tl.sum(tl.where(k_mask[None, :], attn_sum_block, 0.0), axis=1)
        
        # 加载index_score
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + \
                  offs_m[:, None] * stride_iss + offs_k[None, :] * stride_isk
        is_val = tl.load(is_ptrs, mask=mk_mask, other=-float("inf"))
        
        # 加载mask
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + \
                    offs_m[:, None] * stride_ms + offs_k[None, :] * stride_mk
        mask_val = tl.load(mask_ptrs, mask=mk_mask, other=True)
        is_val = tl.where(mask_val | ~k_mask[None, :], -float("inf"), is_val)
        
        # 更新max_is (online softmax)
        block_max = tl.max(is_val, axis=1)
        new_max = tl.maximum(max_is, block_max)
        
        # Rescale sum_is
        alpha = tl.exp(max_is - new_max)
        sum_is = sum_is * alpha + tl.sum(tl.exp(is_val - new_max[:, None]), axis=1)
        max_is = new_max
    
    # Pass 2: 计算KL散度
    kl_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        mk_mask = m_mask[:, None] & k_mask[None, :]
        
        # 重新计算attn_sum并归一化
        attn_sum_block = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for h in tl.static_range(num_heads):
            attn_ptrs = AttnScores_ptr + pid_batch * stride_ab + h * stride_ah + \
                        offs_m[:, None] * stride_as + offs_k[None, :] * stride_ak
            attn_val = tl.load(attn_ptrs, mask=mk_mask, other=0.0)
            attn_sum_block += attn_val
        
        attn_dist = attn_sum_block / (attn_total[:, None] + eps) + eps
        
        # 计算index_prob
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + \
                  offs_m[:, None] * stride_iss + offs_k[None, :] * stride_isk
        is_val = tl.load(is_ptrs, mask=mk_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + \
                    offs_m[:, None] * stride_ms + offs_k[None, :] * stride_mk
        mask_val = tl.load(mask_ptrs, mask=mk_mask, other=True)
        is_val = tl.where(mask_val | ~k_mask[None, :], -float("inf"), is_val)
        
        index_prob = tl.exp(is_val - max_is[:, None]) / sum_is[:, None] + eps
        
        # KL散度
        kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
        kl = tl.where(k_mask[None, :], kl, 0.0)
        kl_sum += tl.sum(kl, axis=1)
    
    # 存储结果
    loss_ptrs = Loss_ptr + pid_batch * seq_len + offs_m
    tl.store(loss_ptrs, kl_sum, mask=m_mask)


def post_reduce_loss_fused(attn_scores, index_score, mask, eps=1e-10):
    """
    Kernel 2: Post-Reduce - 高性能融合计算
    """
    batch_size, num_heads, seq_len, _ = attn_scores.shape
    
    attn_scores = attn_scores.contiguous()
    index_score = index_score.contiguous()
    mask = mask.contiguous()
    
    loss_per_row = torch.zeros(batch_size, seq_len, device=attn_scores.device, dtype=torch.float32)
    
    # 选择合适的block size
    if seq_len <= 512:
        BLOCK_M = 32
        BLOCK_K = 64
    elif seq_len <= 2048:
        BLOCK_M = 64
        BLOCK_K = 128
    else:
        BLOCK_M = 64
        BLOCK_K = 256
    
    num_m_blocks = triton.cdiv(seq_len, BLOCK_M)
    grid = (batch_size * num_m_blocks,)
    
    _post_reduce_loss_kernel_v2[grid](
        attn_scores, index_score, mask, loss_per_row,
        batch_size, num_heads, seq_len, eps,
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        BLOCK_M, BLOCK_K,
    )
    
    return loss_per_row.sum() / batch_size


# ============================================================================
# 组合函数
# ============================================================================

def compute_index_loss_fused(query, key, index_score, index_mask, scaling):
    """
    Fused版本: 两个kernel串联 (不含reduce)
    """
    attn_scores = attention_softmax_fused(query, key, index_mask, scaling)
    loss = post_reduce_loss_fused(attn_scores, index_score, index_mask)
    return loss


# ============================================================================
# PyTorch参考实现
# ============================================================================

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


def attention_softmax_ref(query, key, mask, scaling):
    """Kernel 1的PyTorch参考"""
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(mask.unsqueeze(1), -1e9)
    attn = torch.softmax(attn, dim=-1)
    return attn


def post_reduce_loss_ref(attn_scores, index_score, mask, eps=1e-10):
    """Kernel 2的PyTorch参考"""
    attn_sum = attn_scores.sum(dim=1)
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    index_score_masked = index_score.masked_fill(mask, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
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
    """测试 Kernel 1"""
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
    print(f"  Pass: {diff < 1e-3}")
    
    return diff < 1e-3


def test_kernel2(batch_size=2, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试 Kernel 2"""
    torch.manual_seed(seed)
    device = 'cuda'
    
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
    print(f"  Pass: {diff < 1e-3}")
    
    return diff < 1e-3


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
    """测试Fused实现"""
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
    
    # 精度测试
    print("\n>>> 精度测试")
    
    print("\n[Kernel 1: Attention Softmax]")
    attn_ref = attention_softmax_ref(query, key, mask, scaling)
    attn_tri = attention_softmax_fused(query, key, mask, scaling)
    diff1 = (attn_ref - attn_tri).abs().max().item()
    print(f"  Max diff: {diff1:.6e}")
    
    print("\n[Kernel 2: Post-Reduce Loss]")
    loss_ref = post_reduce_loss_ref(attn_ref, index_score, mask)
    loss_tri = post_reduce_loss_fused(attn_tri, index_score, mask)
    diff2 = abs(loss_ref.item() - loss_tri.item())
    print(f"  Ref: {loss_ref.item():.6f}, Triton: {loss_tri.item():.6f}")
    print(f"  Diff: {diff2:.6e}")
    
    print("\n[完整流程]")
    full_ref = pytorch_reference(query, key, index_score, mask, scaling)
    full_tri = compute_index_loss_fused(query, key, index_score, mask, scaling)
    diff_full = abs(full_ref.item() - full_tri.item())
    print(f"  PyTorch: {full_ref.item():.6f}")
    print(f"  Triton Fused: {full_tri.item():.6f}")
    print(f"  Diff: {diff_full:.6e}")
    
    # 性能测试
    print(f"\n>>> 性能测试 (warmup={num_warmup}, iterations={num_benchmark})")
    
    # Warmup
    for _ in range(num_warmup):
        _ = compute_index_loss_fused(query, key, index_score, mask, scaling)
    torch.cuda.synchronize()
    
    # Benchmark Triton
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
    
    return {
        'kernel1_diff': diff1,
        'kernel2_diff': diff2,
        'full_diff': diff_full,
        'pytorch_time_ms': pytorch_time,
        'triton_time_ms': triton_time,
        'speedup': pytorch_time / triton_time if triton_time > 0 else float('inf'),
    }


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
        batch_size=1,
        num_heads=16,
        seq_len=4096,
        head_dim=256,
        topk=2048,
        num_warmup=5,
        num_benchmark=10,
    )
