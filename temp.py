"""
Triton Fused Optimized - Sparse Attention (H20 GPU优化版本)

针对H20 (Hopper架构, sm_90) 的优化:
1. Autotune配置优化
2. 每个program处理BLOCK_M行query
3. num_stages/num_warps参数调优
4. 内存访问模式优化
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================================
# Autotune配置 - 针对H20 GPU优化
# ============================================================================

def get_autotune_configs():
    """生成autotune配置，针对H20 (Hopper, sm_90) GPU
    
    H20硬件参数:
    - 78 SMs, 228KB shared mem/SM, 65536 regs/SM
    - 每SM最大2048线程 (64 warps)
    - 每Block最大1024线程
    """
    configs = []
    # H20推荐配置: num_warps=4-16, num_stages=2-4
    # 更大的num_stages利用228KB共享内存进行更深流水线
    for block_d in [64, 128, 256]:
        for num_stages in [2, 3, 4]:
            for num_warps in [4, 8]:
                configs.append(
                    triton.Config(
                        {'BLOCK_D': block_d},
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs


# H20 GPU常量
H20_NUM_SMS = 78
H20_SMEM_PER_SM = 228 * 1024  # 228 KB
H20_REGS_PER_SM = 65536


# ============================================================================
# Kernel 1: Sparse Attention Softmax (H20优化版本)
# ============================================================================

@triton.autotune(
    configs=get_autotune_configs(),
    key=['head_dim', 'topk'],
)
@triton.jit
def _sparse_attn_h20_kernel(
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
    H20优化的Sparse Attention Kernel
    
    优化点:
    1. 使用autotune选择最佳BLOCK_D, num_stages, num_warps
    2. 使用2D指针批量load K[indices, :]
    3. 向量化点积计算
    4. 编译器hints优化 (tl.multiple_of)
    5. 内存对齐提示
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
    
    # 提示编译器stride对齐，启用向量化加载
    stride_qd = tl.multiple_of(stride_qd, 8)  # 假设至少8字节对齐
    stride_kd = tl.multiple_of(stride_kd, 8)
    
    # 预加载所有indices [topk]
    offs_topk = tl.arange(0, topk)
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    
    # Causal mask
    causal_mask = indices > pid_row
    
    # 计算所有QK值 - 分块处理head_dim
    qk = tl.zeros([topk], dtype=tl.float32)
    
    # 使用tl.static_range以便编译器展开循环
    num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
    for d_idx in range(num_d_blocks):
        d_start = d_idx * BLOCK_D
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < head_dim
        
        # 加载Q chunk: [BLOCK_D]
        q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
        
        # 批量load K: 使用2D指针 [topk, BLOCK_D]
        k_ptrs = k_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
        
        # 向量化点积: q[d] * k_gathered[topk, d] -> sum over d -> [topk]
        qk += tl.sum(q[None, :] * k_gathered, axis=1)
    
    # 应用scaling和mask
    qk = qk * scaling
    qk = tl.where(causal_mask, NEG_INF, qk)
    
    # Softmax (数值稳定版本)
    m = tl.max(qk)
    # 防止全mask情况
    m = tl.where(m == NEG_INF, 0.0, m)
    p = tl.exp(qk - m)
    l = tl.sum(p)
    l = tl.where(l < 1e-9, 1.0, l)
    p = p / l
    p = tl.where(causal_mask, 0.0, p)
    
    # 写出
    tl.store(out_base + offs_topk * stride_ok, p.to(Out_ptr.dtype.element_ty))


# ============================================================================
# Kernel 1 变体: 每program处理BLOCK_M行 (减少kernel启动开销)
# ============================================================================

@triton.autotune(
    configs=[
        # 小BLOCK_M配置 - 适合小seq_len或大topk
        triton.Config({'BLOCK_M': 4, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_D': 128}, num_stages=3, num_warps=4),
        # 中等BLOCK_M配置 - 平衡选择
        triton.Config({'BLOCK_M': 8, 'BLOCK_D': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_D': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_D': 256}, num_stages=4, num_warps=8),
        # 大BLOCK_M配置 - 减少kernel启动开销
        triton.Config({'BLOCK_M': 16, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_D': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_D': 256}, num_stages=4, num_warps=8),
        # 更大BLOCK_M配置 - 适合大seq_len
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 256}, num_stages=4, num_warps=8),
    ],
    key=['head_dim', 'topk', 'seq_len'],
)
@triton.jit
def _sparse_attn_multirow_kernel(
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
    多行处理的Sparse Attention Kernel (H20优化版本)
    
    优化点:
    1. 每个program处理BLOCK_M行query，减少kernel启动开销
    2. 使用tl.multiple_of提示内存对齐
    3. 预计算共享的基地址和偏移量
    4. 使用tl.static_range优化循环
    """
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    num_per_head = num_m_blocks
    num_per_batch = num_heads * num_per_head
    
    pid_batch = pid // num_per_batch
    pid_temp = pid % num_per_batch
    pid_head = pid_temp // num_per_head
    pid_m = pid_temp % num_per_head
    
    NEG_INF = -1e9
    
    # 预计算共享的基地址
    q_batch_head_base = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_batch_base = Indices_ptr + pid_batch * stride_ib
    out_batch_head_base = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh
    
    # 提示编译器stride对齐
    stride_qd = tl.multiple_of(stride_qd, 8)
    stride_kd = tl.multiple_of(stride_kd, 8)
    
    # 预计算偏移量
    offs_topk = tl.arange(0, topk)
    row_start = pid_m * BLOCK_M
    
    # 处理BLOCK_M行 - 使用tl.static_range让编译器优化
    for mi in tl.static_range(BLOCK_M):
        row = row_start + mi
        # Triton不支持continue，改用条件包裹整个循环体
        if row < seq_len:
            q_base = q_batch_head_base + row * stride_qs
            idx_base = idx_batch_base + row * stride_is
            out_base = out_batch_head_base + row * stride_os
            
            # 加载indices
            indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
            causal_mask = indices > row
            
            # 计算QK - 分块处理head_dim
            qk = tl.zeros([topk], dtype=tl.float32)
            
            num_d_blocks: tl.constexpr = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in tl.static_range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                # 加载Q chunk
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                
                # 批量加载K: [topk, BLOCK_D]
                k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
                
                # 向量化点积
                qk += tl.sum(q[None, :] * k_gathered, axis=1)
            
            # 应用scaling和causal mask
            qk = qk * scaling
            qk = tl.where(causal_mask, NEG_INF, qk)
            
            # 数值稳定的softmax
            m = tl.max(qk)
            m = tl.where(m == NEG_INF, 0.0, m)
            p = tl.exp(qk - m)
            l = tl.sum(p)
            l = tl.where(l < 1e-9, 1.0, l)
            p = p / l
            p = tl.where(causal_mask, 0.0, p)
            
            # 写出结果
            tl.store(out_base + offs_topk * stride_ok, p.to(Out_ptr.dtype.element_ty))


def sparse_attention_softmax_fused(query, key, indices, scaling, use_multirow=True):
    """Sparse Attention Softmax (H20优化版本)"""
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    
    query = query.contiguous()
    key = key.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    attn_scores = torch.empty(batch_size, num_heads, seq_len, topk,
                              device=query.device, dtype=query.dtype)
    
    if use_multirow:
        # 使用多行处理kernel
        grid = lambda META: (
            batch_size * num_heads * triton.cdiv(seq_len, META['BLOCK_M']),
        )
        
        _sparse_attn_multirow_kernel[grid](
            query, key, indices, attn_scores,
            batch_size, num_heads, seq_len, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        )
    else:
        # 使用单行kernel
        grid = (batch_size * num_heads * seq_len,)
        
        _sparse_attn_h20_kernel[grid](
            query, key, indices, attn_scores,
            batch_size, num_heads, seq_len, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        )
    
    return attn_scores


# ============================================================================
# Kernel 2: Sparse Post-Reduce Loss (H20优化版本)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=8),
    ],
    key=['num_heads', 'topk'],
)
@triton.jit
def _sparse_loss_h20_kernel(
    AttnScores_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    batch_size, num_heads: tl.constexpr, seq_len,
    topk: tl.constexpr,
    eps: tl.constexpr,
    stride_ab, stride_ah, stride_as, stride_ak,
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
):
    """H20优化的Sparse Loss Kernel"""
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
    attn_total = tl.where(attn_total < eps, 1.0, attn_total)
    attn_dist = attn_sum / attn_total + eps
    
    # 计算index_score的softmax
    is_ptr = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
    is_val = tl.load(is_ptr)
    is_val = tl.where(causal_mask, NEG_INF, is_val)
    
    m_is = tl.max(is_val)
    m_is = tl.where(m_is == NEG_INF, 0.0, m_is)
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
    """Sparse Post-Reduce Loss (H20优化版本)"""
    batch_size, num_heads, seq_len, topk = attn_scores.shape
    
    attn_scores = attn_scores.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    loss_per_row = torch.zeros(batch_size, seq_len, device=attn_scores.device, dtype=torch.float32)
    
    grid = (batch_size * seq_len,)
    
    _sparse_loss_h20_kernel[grid](
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

def compute_index_loss_sparse(query, key, index_score, indices, scaling, use_multirow=True):
    """Sparse版本的完整loss计算 (H20优化)"""
    attn_scores = sparse_attention_softmax_fused(query, key, indices, scaling, use_multirow)
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
    passed = diff < 1e-4
    print(f"Kernel 1 Accuracy - Max diff: {diff:.6e}, Pass: {passed}")
    return passed


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
    print(f"Full Accuracy - Ref: {ref.item():.6f}, Triton: {tri.item():.6f}, Diff: {diff:.6e}, Pass: {passed}")
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
    print("Sparse Triton H20优化 性能测试")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print(f"理论复杂度: O(seq * topk * head_dim) = O({seq_len * topk * head_dim:,})")
    print("=" * 70)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    results = {}
    
    # Test 1: Triton multirow kernel
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_multirow=True)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_multirow=True)
    torch.cuda.synchronize()
    triton_multirow_time = (time.time() - start) / num_benchmark * 1000
    results['triton_multirow'] = triton_multirow_time
    
    # Test 2: Triton single-row kernel
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_multirow=False)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_multirow=False)
    torch.cuda.synchronize()
    triton_singlerow_time = (time.time() - start) / num_benchmark * 1000
    results['triton_singlerow'] = triton_singlerow_time
    
    # Test 3: PyTorch reference
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
    print(f"  Triton single-row:     {triton_singlerow_time:.3f} ms (加速: {pytorch_time/triton_singlerow_time:.2f}x)")
    print(f"  Triton multi-row:      {triton_multirow_time:.3f} ms (加速: {pytorch_time/triton_multirow_time:.2f}x)")
    
    best_triton = min(triton_multirow_time, triton_singlerow_time)
    print(f"\n  最佳Triton vs PyTorch: {pytorch_time/best_triton:.2f}x 加速")
    
    return results


def test_scaling(topk=512):
    """测试线性scaling"""
    print("\n" + "=" * 70)
    print(f"线性Scaling测试 (固定topk={topk})")
    print("=" * 70)
    
    results = []
    for seq_len in [1024, 2048, 4096, 8192]:
        try:
            r = test_performance(seq_len=seq_len, topk=topk, head_dim=128, num_heads=8, 
                                num_warmup=5, num_benchmark=20)
            results.append({'seq_len': seq_len, **r})
        except Exception as e:
            print(f"seq={seq_len} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n汇总 (Triton multi-row):")
    if results:
        base = results[0]['triton_multirow']
        for r in results:
            ratio = r['triton_multirow'] / base
            expected = r['seq_len'] / results[0]['seq_len']
            print(f"  seq={r['seq_len']}: {r['triton_multirow']:.3f}ms (实际{ratio:.2f}x, 预期{expected:.2f}x)")


def compare_with_dense():
    """与dense实现对比"""
    print("\n" + "=" * 70)
    print("对比: Sparse vs Dense实现")
    print("=" * 70)
    
    # 尝试导入dense实现
    try:
        from triton_fused import compute_index_loss_fused as dense_loss
        has_dense = True
    except:
        has_dense = False
        print("无法导入dense实现")
    
    if has_dense:
        import time
        device = 'cuda'
        batch_size, num_heads, seq_len, head_dim = 1, 16, 2048, 128
        topk = 256
        scaling = 1.0 / (head_dim ** 0.5)
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
        indices = generate_topk_indices(batch_size, seq_len, topk, device)
        index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
        
        # Dense需要不同的输入格式
        # 这里仅测试sparse
        print("Sparse实现性能测试...")
        test_performance(seq_len=seq_len, topk=topk, head_dim=head_dim, num_heads=num_heads)


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
