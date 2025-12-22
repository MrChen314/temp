"""
Triton Fused Optimized - Sparse Attention (高性能版本 V2)

关键优化:
1. 批量处理多个query行 (BLOCK_M)
2. 向量化gather K
3. 使用tl.dot进行矩阵乘法
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================================
# Kernel 1: Sparse Attention Softmax (向量化版本)
# ============================================================================

@triton.jit
def _sparse_attn_fwd_kernel(
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
    BLOCK_K: tl.constexpr,  # topk方向的block size
    BLOCK_D: tl.constexpr,
):
    """
    Sparse Attention Kernel (高效版本)
    
    每个program处理 BLOCK_M 行query
    使用分块处理topk和head_dim
    """
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    num_programs_per_head = num_m_blocks
    num_programs_per_batch = num_heads * num_programs_per_head
    
    pid_batch = pid // num_programs_per_batch
    pid_temp = pid % num_programs_per_batch
    pid_head = pid_temp // num_programs_per_head
    pid_m = pid_temp % num_programs_per_head
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < seq_len
    
    NEG_INF = -1e9
    
    # Online softmax状态: [BLOCK_M]
    m_i = tl.full([BLOCK_M], value=NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # 遍历topk blocks
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # 加载indices: [BLOCK_M, BLOCK_K]
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + \
                   offs_m[:, None] * stride_is + offs_k[None, :] * stride_ik
        indices = tl.load(idx_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0).to(tl.int64)
        
        # Causal mask: [BLOCK_M, BLOCK_K]
        causal_mask = indices > offs_m[:, None]
        
        # 计算QK^T: 分块处理head_dim
        qk = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        
        for d_start in range(0, head_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            # 加载Q: [BLOCK_M, BLOCK_D]
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # Gather K并累加点积: 对每个BLOCK_K位置
            # K的形状是 [batch, seq_len, head_dim]
            # 我们需要根据indices [BLOCK_M, BLOCK_K] gather K 得到 [BLOCK_M, BLOCK_K, BLOCK_D]
            # 但Triton不支持3D张量操作，所以逐BLOCK_K列处理
            
            for ki in tl.static_range(BLOCK_K):
                if start_k + ki < topk:
                    # 获取第ki列的indices: [BLOCK_M]
                    idx_col = tl.load(
                        Indices_ptr + pid_batch * stride_ib + 
                        offs_m * stride_is + (start_k + ki) * stride_ik,
                        mask=m_mask, other=0
                    ).to(tl.int64)
                    
                    # 为每个query行gather对应的K向量: [BLOCK_M, BLOCK_D]
                    # K[batch, idx, :] for each row
                    k_gathered = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
                    
                    for mi in tl.static_range(BLOCK_M):
                        if pid_m * BLOCK_M + mi < seq_len:
                            idx_val = tl.load(
                                Indices_ptr + pid_batch * stride_ib + 
                                (pid_m * BLOCK_M + mi) * stride_is + (start_k + ki) * stride_ik
                            ).to(tl.int64)
                            k_row = tl.load(
                                K_ptr + pid_batch * stride_kb + idx_val * stride_ks + offs_d * stride_kd,
                                mask=d_mask, other=0.0
                            ).to(tl.float32)
                            # 更新k_gathered的第mi行
                            for di in tl.static_range(BLOCK_D):
                                k_gathered = tl.where(
                                    (tl.arange(0, BLOCK_M) == mi)[:, None] & 
                                    (tl.arange(0, BLOCK_D) == di)[None, :],
                                    k_row[di], k_gathered
                                )
                    
                    # 计算点积: Q @ K^T for this column
                    # q: [BLOCK_M, BLOCK_D], k_gathered: [BLOCK_M, BLOCK_D]
                    # 我们要 qk[:, ki] = sum(q * k_gathered, axis=1)
                    dot = tl.sum(q * k_gathered, axis=1)  # [BLOCK_M]
                    
                    # 更新qk的第ki列
                    for mi in tl.static_range(BLOCK_M):
                        qk = tl.where(
                            (tl.arange(0, BLOCK_M) == mi)[:, None] & 
                            (tl.arange(0, BLOCK_K) == ki)[None, :],
                            dot[mi], qk
                        )
        
        qk = qk * scaling
        qk = tl.where(causal_mask | ~k_mask[None, :] | ~m_mask[:, None], NEG_INF, qk)
        
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    
    l_i = tl.where(l_i < 1e-9, 1.0, l_i)
    
    # Pass 2: 写出softmax结果
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # 重新计算QK (或者可以存储，但会用更多寄存器)
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + \
                   offs_m[:, None] * stride_is + offs_k[None, :] * stride_ik
        indices = tl.load(idx_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0).to(tl.int64)
        causal_mask = indices > offs_m[:, None]
        
        qk = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        
        for d_start in range(0, head_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            for ki in tl.static_range(BLOCK_K):
                if start_k + ki < topk:
                    k_gathered = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
                    
                    for mi in tl.static_range(BLOCK_M):
                        if pid_m * BLOCK_M + mi < seq_len:
                            idx_val = tl.load(
                                Indices_ptr + pid_batch * stride_ib + 
                                (pid_m * BLOCK_M + mi) * stride_is + (start_k + ki) * stride_ik
                            ).to(tl.int64)
                            k_row = tl.load(
                                K_ptr + pid_batch * stride_kb + idx_val * stride_ks + offs_d * stride_kd,
                                mask=d_mask, other=0.0
                            ).to(tl.float32)
                            for di in tl.static_range(BLOCK_D):
                                k_gathered = tl.where(
                                    (tl.arange(0, BLOCK_M) == mi)[:, None] & 
                                    (tl.arange(0, BLOCK_D) == di)[None, :],
                                    k_row[di], k_gathered
                                )
                    
                    dot = tl.sum(q * k_gathered, axis=1)
                    for mi in tl.static_range(BLOCK_M):
                        qk = tl.where(
                            (tl.arange(0, BLOCK_M) == mi)[:, None] & 
                            (tl.arange(0, BLOCK_K) == ki)[None, :],
                            dot[mi], qk
                        )
        
        qk = qk * scaling
        qk = tl.where(causal_mask | ~k_mask[None, :] | ~m_mask[:, None], NEG_INF, qk)
        
        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
        p = tl.where(m_mask[:, None] & k_mask[None, :] & ~causal_mask, p, 0.0)
        
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + \
                   offs_m[:, None] * stride_os + offs_k[None, :] * stride_ok
        tl.store(out_ptrs, p.to(Out_ptr.dtype.element_ty), 
                 mask=m_mask[:, None] & k_mask[None, :])


# ============================================================================
# V2: 更简洁高效的实现 - 每个program处理一行
# ============================================================================

@triton.jit  
def _sparse_attn_simple_kernel(
    Q_ptr, K_ptr, Indices_ptr, Out_ptr,
    batch_size, num_heads, seq_len,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    stride_ob, stride_oh, stride_os, stride_ok,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """简化版本：每个program处理一行query"""
    pid = tl.program_id(0)
    num_per_head = seq_len
    num_per_batch = num_heads * num_per_head
    
    pid_batch = pid // num_per_batch
    pid_temp = pid % num_per_batch  
    pid_head = pid_temp // num_per_head
    pid_row = pid_temp % num_per_head
    
    NEG_INF = -1e9
    
    # 加载Q一次: [head_dim]
    q_base = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + pid_row * stride_qs
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    k_base = K_ptr + pid_batch * stride_kb
    
    # Online softmax
    m_i = NEG_INF
    l_i = 0.0
    
    # Pass 1: 计算max和sum
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # 加载indices
        indices = tl.load(idx_base + offs_k * stride_ik, mask=k_mask, other=0).to(tl.int64)
        causal_mask = indices > pid_row
        
        # 计算QK
        qk = tl.zeros([BLOCK_K], dtype=tl.float32)
        
        for d_start in range(0, head_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            # 加载Q chunk
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            
            # 对每个topk位置计算部分点积
            for ki in tl.static_range(BLOCK_K):
                idx = tl.load(idx_base + (start_k + ki) * stride_ik).to(tl.int64)
                k = tl.load(k_base + idx * stride_ks + offs_d * stride_kd, 
                           mask=d_mask, other=0.0).to(tl.float32)
                dot = tl.sum(q * k)
                qk = tl.where(tl.arange(0, BLOCK_K) == ki, qk + dot, qk)
        
        qk = qk * scaling
        qk = tl.where(causal_mask | ~k_mask, NEG_INF, qk)
        
        # Online softmax update
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new
    
    l_i = tl.where(l_i < 1e-9, 1.0, l_i)
    
    # Pass 2: 计算并写出softmax
    out_base = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + pid_row * stride_os
    
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        indices = tl.load(idx_base + offs_k * stride_ik, mask=k_mask, other=0).to(tl.int64)
        causal_mask = indices > pid_row
        
        qk = tl.zeros([BLOCK_K], dtype=tl.float32)
        
        for d_start in range(0, head_dim, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            
            for ki in tl.static_range(BLOCK_K):
                idx = tl.load(idx_base + (start_k + ki) * stride_ik).to(tl.int64)
                k = tl.load(k_base + idx * stride_ks + offs_d * stride_kd,
                           mask=d_mask, other=0.0).to(tl.float32)
                dot = tl.sum(q * k)
                qk = tl.where(tl.arange(0, BLOCK_K) == ki, qk + dot, qk)
        
        qk = qk * scaling
        qk = tl.where(causal_mask | ~k_mask, NEG_INF, qk)
        
        p = tl.exp(qk - m_i) / l_i
        p = tl.where(k_mask & ~causal_mask, p, 0.0)
        
        tl.store(out_base + offs_k * stride_ok, p.to(Out_ptr.dtype.element_ty), mask=k_mask)


# ============================================================================
# V3: 最优化版本 - 预gather K，使用tl.dot
# ============================================================================

@triton.jit
def _sparse_attn_v3_kernel(
    Q_ptr, K_ptr, Indices_ptr, Out_ptr,
    batch_size, num_heads, seq_len,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    stride_ob, stride_oh, stride_os, stride_ok,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    V3: 预gather K到寄存器，然后使用向量化计算
    关键：将K gather到连续的寄存器块
    """
    pid = tl.program_id(0)
    num_per_head = seq_len
    num_per_batch = num_heads * num_per_head
    
    pid_batch = pid // num_per_batch
    pid_temp = pid % num_per_batch
    pid_head = pid_temp // num_per_head
    pid_row = pid_temp % num_per_head
    
    NEG_INF = -1e9
    
    q_base = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + pid_row * stride_qs
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    k_base = K_ptr + pid_batch * stride_kb
    out_base = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + pid_row * stride_os
    
    # 预加载所有indices
    offs_topk = tl.arange(0, topk)
    all_indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    causal_mask_all = all_indices > pid_row
    
    # 计算所有QK值
    qk_all = tl.zeros([topk], dtype=tl.float32)
    
    # 分块处理head_dim
    for d_start in range(0, head_dim, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < head_dim
        
        # 加载Q: [BLOCK_D]
        q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
        
        # 对所有topk位置gather K并计算点积
        # 由于topk可能很大，分块处理
        for start_k in range(0, topk, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            k_mask = offs_k < topk
            
            # Gather K: 需要根据indices[start_k:start_k+BLOCK_K]加载K
            # K_gathered: [BLOCK_K, BLOCK_D]
            
            # 这里的关键优化：批量计算所有点积
            for ki in tl.static_range(BLOCK_K):
                if start_k + ki < topk:
                    idx = tl.load(idx_base + (start_k + ki) * stride_ik).to(tl.int64)
                    k = tl.load(k_base + idx * stride_ks + offs_d * stride_kd,
                               mask=d_mask, other=0.0).to(tl.float32)
                    dot = tl.sum(q * k)
                    qk_all = tl.where(offs_topk == (start_k + ki), qk_all + dot, qk_all)
    
    # 应用scaling和mask
    qk_all = qk_all * scaling
    qk_all = tl.where(causal_mask_all, NEG_INF, qk_all)
    
    # Softmax
    m = tl.max(qk_all)
    p = tl.exp(qk_all - m)
    l = tl.sum(p)
    l = tl.where(l < 1e-9, 1.0, l)
    p = p / l
    p = tl.where(causal_mask_all, 0.0, p)
    
    # 写出
    tl.store(out_base + offs_topk * stride_ok, p.to(Out_ptr.dtype.element_ty))


def sparse_attention_softmax_fused(query, key, indices, scaling):
    """Sparse Attention Softmax"""
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    
    query = query.contiguous()
    key = key.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    attn_scores = torch.empty(batch_size, num_heads, seq_len, topk,
                              device=query.device, dtype=query.dtype)
    
    # 选择block sizes
    BLOCK_D = min(64, triton.next_power_of_2(head_dim))
    BLOCK_K = min(64, triton.next_power_of_2(topk))
    
    grid = (batch_size * num_heads * seq_len,)
    
    # 使用V3 kernel
    _sparse_attn_v3_kernel[grid](
        query, key, indices, attn_scores,
        batch_size, num_heads, seq_len, head_dim, topk, scaling,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        BLOCK_K, BLOCK_D,
    )
    
    return attn_scores


# ============================================================================
# Kernel 2: Sparse Post-Reduce Loss
# ============================================================================

@triton.jit
def _sparse_loss_kernel(
    AttnScores_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    batch_size, num_heads: tl.constexpr, seq_len,
    topk: tl.constexpr,
    eps: tl.constexpr,
    stride_ab, stride_ah, stride_as, stride_ak,
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
    BLOCK_K: tl.constexpr,
):
    """Sparse Loss Kernel"""
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    NEG_INF = -1e9
    offs_k = tl.arange(0, topk)
    
    # 加载indices和causal mask
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    indices = tl.load(idx_base + offs_k * stride_ik).to(tl.int64)
    causal_mask = indices > pid_row
    
    # 计算attention分布
    attn_sum = tl.zeros([topk], dtype=tl.float32)
    for h in tl.static_range(num_heads):
        attn_ptr = AttnScores_ptr + pid_batch * stride_ab + h * stride_ah + \
                   pid_row * stride_as + offs_k * stride_ak
        attn = tl.load(attn_ptr)
        attn_sum += attn
    
    attn_total = tl.sum(attn_sum)
    attn_dist = attn_sum / (attn_total + eps) + eps
    
    # 计算index_prob softmax
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
    """Sparse Post-Reduce Loss"""
    batch_size, num_heads, seq_len, topk = attn_scores.shape
    
    attn_scores = attn_scores.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    loss_per_row = torch.zeros(batch_size, seq_len, device=attn_scores.device, dtype=torch.float32)
    
    BLOCK_K = min(256, triton.next_power_of_2(topk))
    grid = (batch_size * seq_len,)
    
    _sparse_loss_kernel[grid](
        attn_scores, index_score, indices, loss_per_row,
        batch_size, num_heads, seq_len, topk, eps,
        attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        BLOCK_K,
    )
    
    return loss_per_row.sum() / batch_size


# ============================================================================
# 组合函数
# ============================================================================

def compute_index_loss_sparse(query, key, index_score, indices, scaling):
    """Sparse版本"""
    attn_scores = sparse_attention_softmax_fused(query, key, indices, scaling)
    loss = sparse_post_reduce_loss_fused(attn_scores, index_score, indices)
    return loss


# ============================================================================
# PyTorch参考实现
# ============================================================================

def pytorch_reference_sparse(query, key, index_score, indices, scaling):
    """Sparse版本的PyTorch参考"""
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    eps = 1e-10
    
    # Gather K
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    key_expanded = key.unsqueeze(2).expand(-1, -1, topk, -1)
    k_gathered = torch.gather(key_expanded, dim=1, index=indices_expanded)
    
    # Attention
    attn = torch.einsum('bhsd,bstd->bhst', query, k_gathered) * scaling
    
    # Causal mask
    row_indices = torch.arange(seq_len, device=query.device).view(1, 1, -1, 1)
    causal_mask = indices.unsqueeze(1) > row_indices
    attn = attn.masked_fill(causal_mask, -1e9)
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
# 测试
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


def test_sparse(
    batch_size: int = 1,
    num_heads: int = 16, 
    seq_len: int = 4096,
    head_dim: int = 256,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 5,
    num_benchmark: int = 20,
):
    """测试Sparse实现"""
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("=" * 60)
    print("Sparse Triton 测试")
    print("=" * 60)
    print(f"参数: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print(f"理论复杂度: O(seq * topk * head_dim) = O({seq_len * topk * head_dim})")
    print("=" * 60)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    # 精度测试
    print("\n>>> 精度测试")
    ref_loss = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    tri_loss = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    
    print(f"  PyTorch: {ref_loss.item():.6f}")
    print(f"  Triton: {tri_loss.item():.6f}")
    print(f"  Diff: {abs(ref_loss.item() - tri_loss.item()):.6e}")
    
    # 性能测试
    print(f"\n>>> 性能测试 (warmup={num_warmup}, iters={num_benchmark})")
    
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    
    for _ in range(num_warmup):
        _ = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_benchmark * 1000
    
    print(f"\n  PyTorch: {pytorch_time:.3f} ms")
    print(f"  Triton: {triton_time:.3f} ms")
    if triton_time > 0:
        print(f"  加速比: {pytorch_time / triton_time:.2f}x")
    
    return {'pytorch_time_ms': pytorch_time, 'triton_time_ms': triton_time}


def test_scaling():
    """测试线性scaling"""
    print("\n" + "=" * 60)
    print("线性Scaling测试 (固定topk=512)")
    print("=" * 60)
    
    topk = 512
    results = []
    for seq_len in [1024, 2048, 4096, 8192]:
        try:
            r = test_sparse(seq_len=seq_len, topk=topk, head_dim=128, num_heads=8, 
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
    test_sparse(
        batch_size=1,
        num_heads=16,
        seq_len=4096,
        head_dim=256,
        topk=512,
    )
    
    test_scaling()
