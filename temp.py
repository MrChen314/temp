"""
Triton Fused Optimized - Sparse Attention实现

关键优化: 使用indices直接索引KV，只计算topk个位置
复杂度从 O(seq_len * seq_len) 降为 O(seq_len * topk)

当topk固定时，耗时与seq_len成线性关系
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================================
# Kernel 1: Sparse Attention Softmax
# 只计算indices指定的topk个位置的attention
# 输入: Q [B, H, S, D], K [B, S, D], indices [B, S, topk]
# 输出: attn_scores [B, H, S, topk]
# ============================================================================

@triton.jit
def _sparse_attention_softmax_kernel(
    Q_ptr,          # [batch, num_heads, seq_len, head_dim]
    K_ptr,          # [batch, seq_len, head_dim]
    Indices_ptr,    # [batch, seq_len, topk] - int32/int64
    Out_ptr,        # [batch, num_heads, seq_len, topk]
    # Dimensions
    batch_size,
    num_heads,
    seq_len,
    head_dim: tl.constexpr,
    topk,
    scaling,
    # Strides for Q: [batch, num_heads, seq_len, head_dim]
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K: [batch, seq_len, head_dim]
    stride_kb, stride_ks, stride_kd,
    # Strides for Indices: [batch, seq_len, topk]
    stride_ib, stride_is, stride_ik,
    # Strides for Out: [batch, num_heads, seq_len, topk]
    stride_ob, stride_oh, stride_os, stride_ok,
    # Block sizes
    BLOCK_M: tl.constexpr,  # 每个program处理的query行数
    BLOCK_K: tl.constexpr,  # topk的tile大小
    BLOCK_D: tl.constexpr,  # head_dim的block大小
):
    """
    Sparse Attention Kernel: 只计算indices指定的topk个位置
    复杂度 O(seq_len * topk) 而不是 O(seq_len * seq_len)
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
    
    NEG_INF = -1e9
    
    # 初始化online softmax状态
    m_i = tl.full([BLOCK_M], value=NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Pass 1: 计算max和sum (遍历topk，不是seq_len)
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # 加载indices: [BLOCK_M, BLOCK_K]
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + \
                   offs_m[:, None] * stride_is + offs_k[None, :] * stride_ik
        indices = tl.load(idx_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)
        
        # 创建causal mask: indices > query位置的要mask掉
        causal_mask = indices > offs_m[:, None]
        
        # 分块计算QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for start_d in range(0, head_dim, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            # 加载Q: [BLOCK_M, BLOCK_D]
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # 使用indices gather K: [BLOCK_M, BLOCK_K, BLOCK_D]
            # 这里需要对每个(m, k)位置加载对应的K向量
            # indices[m, k] 指向K的位置
            k_gathered = tl.zeros([BLOCK_M, BLOCK_K, BLOCK_D], dtype=tl.float32)
            for mi in range(BLOCK_M):
                for ki in range(BLOCK_K):
                    if mi < seq_len and ki < topk:
                        idx = tl.load(Indices_ptr + pid_batch * stride_ib + 
                                      (offs_m[mi]) * stride_is + (offs_k[ki]) * stride_ik)
                        for di in range(BLOCK_D):
                            if start_d + di < head_dim:
                                k_val = tl.load(K_ptr + pid_batch * stride_kb + 
                                               idx * stride_ks + (start_d + di) * stride_kd)
                                k_gathered[mi, ki, di] = k_val
            
            # 计算点积
            for mi in range(BLOCK_M):
                for ki in range(BLOCK_K):
                    dot_sum = 0.0
                    for di in range(BLOCK_D):
                        dot_sum += q[mi, di] * k_gathered[mi, ki, di]
                    qk[mi, ki] += dot_sum
        
        qk = qk * scaling
        
        # 应用causal mask
        qk = tl.where(causal_mask | ~k_mask[None, :] | ~m_mask[:, None], NEG_INF, qk)
        
        # Online softmax更新
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    
    l_i = tl.where(l_i < 1e-9, 1.0, l_i)
    
    # Pass 2: 计算softmax并写出
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # 加载indices
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + \
                   offs_m[:, None] * stride_is + offs_k[None, :] * stride_ik
        indices = tl.load(idx_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)
        
        causal_mask = indices > offs_m[:, None]
        
        # 重新计算QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for start_d in range(0, head_dim, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # Gather K using indices
            k_gathered = tl.zeros([BLOCK_M, BLOCK_K, BLOCK_D], dtype=tl.float32)
            for mi in range(BLOCK_M):
                for ki in range(BLOCK_K):
                    if mi < seq_len and ki < topk:
                        idx = tl.load(Indices_ptr + pid_batch * stride_ib + 
                                      (offs_m[mi]) * stride_is + (offs_k[ki]) * stride_ik)
                        for di in range(BLOCK_D):
                            if start_d + di < head_dim:
                                k_val = tl.load(K_ptr + pid_batch * stride_kb + 
                                               idx * stride_ks + (start_d + di) * stride_kd)
                                k_gathered[mi, ki, di] = k_val
            
            for mi in range(BLOCK_M):
                for ki in range(BLOCK_K):
                    dot_sum = 0.0
                    for di in range(BLOCK_D):
                        dot_sum += q[mi, di] * k_gathered[mi, ki, di]
                    qk[mi, ki] += dot_sum
        
        qk = qk * scaling
        qk = tl.where(causal_mask | ~k_mask[None, :] | ~m_mask[:, None], NEG_INF, qk)
        
        # Softmax
        p = tl.exp(qk - m_i[:, None]) / l_i[:, None]
        p = tl.where(m_mask[:, None] & k_mask[None, :], p, 0.0)
        
        # 写出
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + \
                   offs_m[:, None] * stride_os + offs_k[None, :] * stride_ok
        out_mask = m_mask[:, None] & k_mask[None, :]
        tl.store(out_ptrs, p.to(Out_ptr.dtype.element_ty), mask=out_mask)


# 更高效的实现：使用tl.load with indices
@triton.jit
def _sparse_attention_softmax_kernel_v2(
    Q_ptr,          # [batch, num_heads, seq_len, head_dim]
    K_ptr,          # [batch, seq_len, head_dim]
    Indices_ptr,    # [batch, seq_len, topk] - int64
    Out_ptr,        # [batch, num_heads, seq_len, topk]
    # Dimensions
    batch_size,
    num_heads,
    seq_len,
    head_dim: tl.constexpr,
    topk,
    scaling,
    # Strides for Q: [batch, num_heads, seq_len, head_dim]
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K: [batch, seq_len, head_dim]
    stride_kb, stride_ks, stride_kd,
    # Strides for Indices: [batch, seq_len, topk]
    stride_ib, stride_is, stride_ik,
    # Strides for Out: [batch, num_heads, seq_len, topk]
    stride_ob, stride_oh, stride_os, stride_ok,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Sparse Attention V2: 向量化gather
    每个program处理一个(batch, head, query_row)
    """
    pid = tl.program_id(0)
    num_programs_per_head = seq_len
    num_programs_per_batch = num_heads * num_programs_per_head
    
    pid_batch = pid // num_programs_per_batch
    pid_remainder = pid % num_programs_per_batch
    pid_head = pid_remainder // num_programs_per_head
    pid_row = pid_remainder % num_programs_per_head
    
    NEG_INF = -1e9
    
    # 初始化
    m_max = NEG_INF
    l_sum = 0.0
    
    # 加载这一行的query: [head_dim]
    offs_d = tl.arange(0, BLOCK_D)
    
    # Pass 1: 计算max和sum
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # 加载indices: [BLOCK_K]
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is + offs_k * stride_ik
        indices = tl.load(idx_ptrs, mask=k_mask, other=0)
        
        # Causal mask
        causal_mask = indices > pid_row
        
        # 计算QK^T: 对每个topk位置
        qk = tl.zeros([BLOCK_K], dtype=tl.float32)
        
        for start_d in range(0, head_dim, BLOCK_D):
            offs_d_local = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d_local < head_dim
            
            # 加载Q: [BLOCK_D]
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     pid_row * stride_qs + offs_d_local * stride_qd
            q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
            
            # 对每个topk位置，加载对应的K并计算点积
            for ki in range(BLOCK_K):
                if ki < topk - start_k:
                    idx = tl.load(Indices_ptr + pid_batch * stride_ib + 
                                  pid_row * stride_is + (start_k + ki) * stride_ik)
                    # 加载K[idx]: [BLOCK_D]
                    k_ptrs = K_ptr + pid_batch * stride_kb + idx * stride_ks + offs_d_local * stride_kd
                    k = tl.load(k_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                    
                    # 点积
                    qk_partial = tl.sum(q * k)
                    qk = tl.where(tl.arange(0, BLOCK_K) == ki, qk + qk_partial, qk)
        
        qk = qk * scaling
        
        # 应用mask
        qk = tl.where(causal_mask | ~k_mask, NEG_INF, qk)
        
        # Online softmax
        block_max = tl.max(qk)
        m_new = max(m_max, block_max)
        alpha = tl.exp(m_max - m_new)
        l_sum = l_sum * alpha + tl.sum(tl.exp(qk - m_new))
        m_max = m_new
    
    l_sum = max(l_sum, 1e-9)
    
    # Pass 2: 计算softmax并写出
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is + offs_k * stride_ik
        indices = tl.load(idx_ptrs, mask=k_mask, other=0)
        causal_mask = indices > pid_row
        
        qk = tl.zeros([BLOCK_K], dtype=tl.float32)
        
        for start_d in range(0, head_dim, BLOCK_D):
            offs_d_local = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d_local < head_dim
            
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     pid_row * stride_qs + offs_d_local * stride_qd
            q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
            
            for ki in range(BLOCK_K):
                if ki < topk - start_k:
                    idx = tl.load(Indices_ptr + pid_batch * stride_ib + 
                                  pid_row * stride_is + (start_k + ki) * stride_ik)
                    k_ptrs = K_ptr + pid_batch * stride_kb + idx * stride_ks + offs_d_local * stride_kd
                    k = tl.load(k_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                    qk_partial = tl.sum(q * k)
                    qk = tl.where(tl.arange(0, BLOCK_K) == ki, qk + qk_partial, qk)
        
        qk = qk * scaling
        qk = tl.where(causal_mask | ~k_mask, NEG_INF, qk)
        
        p = tl.exp(qk - m_max) / l_sum
        p = tl.where(k_mask, p, 0.0)
        
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + \
                   pid_row * stride_os + offs_k * stride_ok
        tl.store(out_ptrs, p.to(Out_ptr.dtype.element_ty), mask=k_mask)


# V3: 更高效的gather实现
@triton.jit
def _sparse_attention_softmax_kernel_v3(
    Q_ptr,          # [batch, num_heads, seq_len, head_dim]
    K_ptr,          # [batch, seq_len, head_dim]
    Indices_ptr,    # [batch, seq_len, topk]
    Out_ptr,        # [batch, num_heads, seq_len, topk]
    # Dimensions
    batch_size,
    num_heads,
    seq_len,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    scaling,
    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    stride_ob, stride_oh, stride_os, stride_ok,
    # Block sizes
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Sparse Attention V3: 每个program处理一个(batch, head, row)
    使用向量化加载和简化的循环结构
    """
    pid = tl.program_id(0)
    num_programs_per_head = seq_len
    num_programs_per_batch = num_heads * num_programs_per_head
    
    pid_batch = pid // num_programs_per_batch
    pid_remainder = pid % num_programs_per_batch
    pid_head = pid_remainder // num_programs_per_head
    pid_row = pid_remainder % num_programs_per_head
    
    NEG_INF = -1e9
    
    # 初始化softmax状态
    m_i = NEG_INF
    l_i = 0.0
    
    # Pass 1: 计算max和sum
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # 加载这个block的indices: [BLOCK_K]
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is + offs_k * stride_ik
        indices = tl.load(idx_ptrs, mask=k_mask, other=0).to(tl.int64)
        
        # Causal mask
        causal_mask = indices > pid_row
        
        # 计算QK^T
        qk = tl.zeros([BLOCK_K], dtype=tl.float32)
        
        for start_d in range(0, head_dim, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            # 加载Q: [BLOCK_D]
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     pid_row * stride_qs + offs_d * stride_qd
            q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
            
            # 对每个topk位置gather K并计算点积
            # K base pointer for this batch
            k_base = K_ptr + pid_batch * stride_kb
            
            for ki in tl.static_range(BLOCK_K):
                if start_k + ki < topk:
                    # 获取这个位置的index
                    idx = tl.load(Indices_ptr + pid_batch * stride_ib + 
                                  pid_row * stride_is + (start_k + ki) * stride_ik).to(tl.int64)
                    
                    # 加载K[idx, :]: [BLOCK_D]
                    k_ptrs = k_base + idx * stride_ks + offs_d * stride_kd
                    k = tl.load(k_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                    
                    # 点积累加
                    dot = tl.sum(q * k)
                    qk = tl.where(offs_k == start_k + ki, qk + dot, qk)
        
        qk = qk * scaling
        qk = tl.where(causal_mask | ~k_mask, NEG_INF, qk)
        
        # Online softmax
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p)
        m_i = m_new
    
    l_i = tl.where(l_i < 1e-9, 1.0, l_i)
    
    # Pass 2: 写出softmax结果
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is + offs_k * stride_ik
        indices = tl.load(idx_ptrs, mask=k_mask, other=0).to(tl.int64)
        causal_mask = indices > pid_row
        
        qk = tl.zeros([BLOCK_K], dtype=tl.float32)
        
        for start_d in range(0, head_dim, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
                     pid_row * stride_qs + offs_d * stride_qd
            q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
            
            k_base = K_ptr + pid_batch * stride_kb
            
            for ki in tl.static_range(BLOCK_K):
                if start_k + ki < topk:
                    idx = tl.load(Indices_ptr + pid_batch * stride_ib + 
                                  pid_row * stride_is + (start_k + ki) * stride_ik).to(tl.int64)
                    k_ptrs = k_base + idx * stride_ks + offs_d * stride_kd
                    k = tl.load(k_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                    dot = tl.sum(q * k)
                    qk = tl.where(offs_k == start_k + ki, qk + dot, qk)
        
        qk = qk * scaling
        qk = tl.where(causal_mask | ~k_mask, NEG_INF, qk)
        
        p = tl.exp(qk - m_i) / l_i
        p = tl.where(k_mask & ~causal_mask, p, 0.0)
        
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + \
                   pid_row * stride_os + offs_k * stride_ok
        tl.store(out_ptrs, p.to(Out_ptr.dtype.element_ty), mask=k_mask)


def sparse_attention_softmax_fused(query, key, indices, scaling):
    """
    Sparse Attention Softmax: 只计算indices指定的topk个位置
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, seq_len, head_dim]
        indices: [batch, seq_len, topk] - 每个query位置要attend的key索引
        scaling: float
    
    Returns:
        attn_scores: [batch, num_heads, seq_len, topk]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    
    query = query.contiguous()
    key = key.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    attn_scores = torch.empty(batch_size, num_heads, seq_len, topk,
                              device=query.device, dtype=query.dtype)
    
    # Block sizes
    BLOCK_K = min(64, triton.next_power_of_2(topk))
    BLOCK_D = min(64, triton.next_power_of_2(head_dim))
    
    grid = (batch_size * num_heads * seq_len,)
    
    _sparse_attention_softmax_kernel_v3[grid](
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
# 输入: attn_scores [B, H, S, topk], index_score [B, S, topk]
# 输出: loss
# ============================================================================

@triton.jit
def _sparse_post_reduce_loss_kernel(
    AttnScores_ptr, # [batch, num_heads, seq_len, topk]
    IndexScore_ptr, # [batch, seq_len, topk]
    Indices_ptr,    # [batch, seq_len, topk] - 用于causal mask
    Loss_ptr,       # [batch, seq_len]
    # Dimensions
    batch_size,
    num_heads: tl.constexpr,
    seq_len,
    topk: tl.constexpr,
    eps: tl.constexpr,
    # Strides
    stride_ab, stride_ah, stride_as, stride_ak,
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
    # Block size
    BLOCK_K: tl.constexpr,
):
    """
    Sparse Post-Reduce Loss: 在topk维度上计算
    复杂度 O(seq_len * topk)
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    NEG_INF = -1e9
    
    # 计算attn_total (heads求和后的总和)
    attn_total = 0.0
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h in tl.static_range(num_heads):
            attn_ptrs = AttnScores_ptr + pid_batch * stride_ab + h * stride_ah + \
                        pid_row * stride_as + offs_k * stride_ak
            attn_val = tl.load(attn_ptrs, mask=k_mask, other=0.0)
            acc += attn_val
        
        attn_total += tl.sum(tl.where(k_mask, acc, 0.0))
    
    # 计算index_score的softmax参数
    # 加载indices用于causal mask
    max_is = NEG_INF
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # 加载indices检查causal
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is + offs_k * stride_ik
        indices = tl.load(idx_ptrs, mask=k_mask, other=0).to(tl.int64)
        causal_mask = indices > pid_row
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=NEG_INF)
        is_val = tl.where(causal_mask | ~k_mask, NEG_INF, is_val)
        
        max_is = tl.maximum(max_is, tl.max(is_val))
    
    sum_is = 0.0
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is + offs_k * stride_ik
        indices = tl.load(idx_ptrs, mask=k_mask, other=0).to(tl.int64)
        causal_mask = indices > pid_row
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=NEG_INF)
        is_val = tl.where(causal_mask | ~k_mask, NEG_INF, is_val)
        
        sum_is += tl.sum(tl.exp(is_val - max_is))
    
    sum_is = tl.where(sum_is < 1e-9, 1.0, sum_is)
    
    # 计算KL散度
    kl_sum = 0.0
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < topk
        
        # attn_sum归一化
        attn_sum = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h in tl.static_range(num_heads):
            attn_ptrs = AttnScores_ptr + pid_batch * stride_ab + h * stride_ah + \
                        pid_row * stride_as + offs_k * stride_ak
            attn_val = tl.load(attn_ptrs, mask=k_mask, other=0.0)
            attn_sum += attn_val
        
        attn_dist = attn_sum / (attn_total + eps) + eps
        
        # index_prob
        idx_ptrs = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is + offs_k * stride_ik
        indices = tl.load(idx_ptrs, mask=k_mask, other=0).to(tl.int64)
        causal_mask = indices > pid_row
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=NEG_INF)
        is_val = tl.where(causal_mask | ~k_mask, NEG_INF, is_val)
        
        index_prob = tl.exp(is_val - max_is) / sum_is + eps
        
        # KL
        kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
        kl = tl.where(k_mask & ~causal_mask, kl, 0.0)
        kl_sum += tl.sum(kl)
    
    tl.store(Loss_ptr + pid_batch * seq_len + pid_row, kl_sum)


def sparse_post_reduce_loss_fused(attn_scores, index_score, indices, eps=1e-10):
    """
    Sparse Post-Reduce Loss
    
    Args:
        attn_scores: [batch, num_heads, seq_len, topk]
        index_score: [batch, seq_len, topk]
        indices: [batch, seq_len, topk]
        eps: float
    
    Returns:
        loss: scalar
    """
    batch_size, num_heads, seq_len, topk = attn_scores.shape
    
    attn_scores = attn_scores.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    loss_per_row = torch.zeros(batch_size, seq_len, device=attn_scores.device, dtype=torch.float32)
    
    BLOCK_K = min(64, triton.next_power_of_2(topk))
    
    grid = (batch_size * seq_len,)
    
    _sparse_post_reduce_loss_kernel[grid](
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
    """
    Sparse版本: 复杂度 O(seq_len * topk)
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, seq_len, head_dim]
        index_score: [batch, seq_len, topk]
        indices: [batch, seq_len, topk]
        scaling: float
    """
    # Kernel 1: Sparse Attention Softmax
    attn_scores = sparse_attention_softmax_fused(query, key, indices, scaling)
    
    # (这里可以插入reduce操作)
    
    # Kernel 2: Sparse Post-Reduce Loss
    loss = sparse_post_reduce_loss_fused(attn_scores, index_score, indices)
    
    return loss


# ============================================================================
# PyTorch参考实现 (Sparse版本)
# ============================================================================

def pytorch_reference_sparse(query, key, index_score, indices, scaling):
    """Sparse版本的PyTorch参考实现"""
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    eps = 1e-10
    
    # Gather K using indices: [batch, seq_len, topk, head_dim]
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, S, topk, D]
    k_gathered = torch.gather(key.unsqueeze(2).expand(-1, -1, topk, -1), 
                              dim=1, 
                              index=indices_expanded)  # [B, S, topk, D]
    
    # 计算attention: [batch, num_heads, seq_len, topk]
    # query: [B, H, S, D], k_gathered: [B, S, topk, D] -> [B, 1, S, topk, D]
    attn = torch.einsum('bhsd,bstd->bhst', query, k_gathered) * scaling
    
    # Causal mask
    row_indices = torch.arange(seq_len, device=query.device).view(1, 1, -1, 1)
    causal_mask = indices.unsqueeze(1) > row_indices
    attn = attn.masked_fill(causal_mask, -1e9)
    
    # Softmax
    attn = torch.softmax(attn, dim=-1)
    
    # Head sum + normalize
    attn_sum = attn.sum(dim=1)  # [B, S, topk]
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    # Index score softmax (也需要causal mask)
    index_score_masked = index_score.masked_fill(indices > torch.arange(seq_len, device=query.device).view(1, -1, 1), -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    # KL
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


# ============================================================================
# 测试函数
# ============================================================================

def generate_topk_indices(batch_size, seq_len, topk, device='cuda'):
    """生成topk indices (causal)"""
    indices = torch.zeros(batch_size, seq_len, topk, dtype=torch.int64, device=device)
    
    for b in range(batch_size):
        for t in range(seq_len):
            # 从[0, t]中随机选择topk个
            valid_range = min(t + 1, topk)
            if valid_range <= topk:
                # 可选的不够topk个，就全选然后补0
                perm = torch.randperm(max(1, t + 1), device=device)[:min(topk, t + 1)]
                indices[b, t, :len(perm)] = perm
            else:
                perm = torch.randperm(t + 1, device=device)[:topk]
                indices[b, t] = perm
    
    return indices


def test_sparse_kernel(
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
    print(f"理论复杂度: O(seq_len * topk) = O({seq_len} * {topk}) = O({seq_len * topk})")
    print("=" * 60)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    # 精度测试
    print("\n>>> 精度测试")
    ref_loss = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    tri_loss = compute_index_loss_sparse(query, key, index_score, indices, scaling)
    
    print(f"  PyTorch Sparse: {ref_loss.item():.6f}")
    print(f"  Triton Sparse: {tri_loss.item():.6f}")
    print(f"  Diff: {abs(ref_loss.item() - tri_loss.item()):.6e}")
    
    # 性能测试
    print(f"\n>>> 性能测试 (warmup={num_warmup}, iterations={num_benchmark})")
    
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
    
    print(f"\n  PyTorch Sparse: {pytorch_time:.3f} ms")
    print(f"  Triton Sparse: {triton_time:.3f} ms")
    if triton_time > 0:
        print(f"  加速比: {pytorch_time / triton_time:.2f}x")
    
    return {
        'pytorch_time_ms': pytorch_time,
        'triton_time_ms': triton_time,
    }


def test_scaling():
    """测试seq_len与耗时的线性关系"""
    print("\n" + "=" * 60)
    print("线性scaling测试 (topk固定)")
    print("=" * 60)
    
    topk = 512
    head_dim = 128
    num_heads = 8
    
    results = []
    for seq_len in [1024, 2048, 4096, 8192]:
        try:
            result = test_sparse_kernel(
                batch_size=1,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                topk=topk,
                num_warmup=3,
                num_benchmark=10,
            )
            results.append({
                'seq_len': seq_len,
                **result
            })
        except Exception as e:
            print(f"seq_len={seq_len} 失败: {e}")
    
    print("\n" + "=" * 60)
    print("Scaling汇总")
    print("=" * 60)
    print(f"{'seq_len':<10} {'Triton(ms)':<15} {'Expected Ratio':<15} {'Actual Ratio':<15}")
    print("-" * 60)
    
    base_time = results[0]['triton_time_ms'] if results else 1
    base_seq = results[0]['seq_len'] if results else 1024
    
    for r in results:
        expected_ratio = r['seq_len'] / base_seq
        actual_ratio = r['triton_time_ms'] / base_time
        print(f"{r['seq_len']:<10} {r['triton_time_ms']:<15.3f} {expected_ratio:<15.2f}x {actual_ratio:<15.2f}x")


if __name__ == "__main__":
    test_sparse_kernel(
        batch_size=1,
        num_heads=16,
        seq_len=4096,
        head_dim=256,
        topk=512,
    )
    
    # 测试线性scaling
    # test_scaling()

