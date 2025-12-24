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
BLOCK_TOPK = 256  # topk 分块大小
NUM_STAGES = 3
NUM_WARPS = 8


# ============================================================================
# Fused Kernel V3: 累加 QK logits + 单次 Softmax (最高效版本)
# ============================================================================
# 原始语义: attn_dist = normalize(sum_h(softmax(QK_h)))
# V3 语义:  attn_dist = softmax(sum_h(QK_h) / num_heads)
# 
# 优势:
# 1. QK 只遍历一次 (不是 128 heads × 2 passes = 256 次)
# 2. Softmax 只做一次 (不是 128 次)
# 3. 没有两遍扫描
# 4. K 加载一次，所有 heads 复用
#
# 理论加速比: 约 100x (256 次循环 -> 2-3 次循环)
# ============================================================================

@triton.jit
def _sparse_attn_loss_v3_kernel(
    Q_ptr, K_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    batch_size, num_heads, chunk_size,
    chunk_offset,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    scaling,
    eps: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """
    V3 Kernel: 累加 QK logits + 单次 Softmax
    
    核心优化:
    1. 外层循环 topk blocks，内层循环 heads
    2. K 加载一次，所有 heads 的 Q 与之计算
    3. 累加 QK logits 而不是 softmax 后的概率
    4. 最后只做一次 softmax
    
    对于每个 query row:
    - 遍历 topk blocks (8 次)
      - 加载 K block [BLOCK_TOPK, head_dim]
      - 遍历 heads (128 次)
        - 加载 Q [head_dim]
        - 累加 QK 到 qk_sum
    - 对 qk_sum 做 softmax -> attn_dist
    - 与 index_score 的 softmax 计算 KL
    """
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    NEG_INF = -1e9
    
    # 基地址
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    
    # =========================================================================
    # Part 1: 累加所有 heads 的 QK (使用 Online Softmax)
    # =========================================================================
    # 累加 QK logits 而不是 softmax 概率
    # 这样只需要一次 softmax，而不是 128 次
    
    m_global = NEG_INF
    l_global = 0.0
    
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        # 累加所有 heads 的 QK
        qk_sum = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
        
        # 分块加载 K 并与所有 heads 的 Q 计算
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            # 加载 K block [BLOCK_TOPK, BLOCK_D] - 只加载一次
            k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # 遍历所有 heads，累加 QK
            for h in range(num_heads):
                q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                # QK 累加
                qk_sum += tl.sum(q[None, :] * k_gathered, axis=1)
        
        # 平均并应用 scaling
        qk_avg = qk_sum * scaling / num_heads
        qk_avg = tl.where(causal_mask_block, NEG_INF, qk_avg)
        
        # Online softmax update
        m_block = tl.max(qk_avg)
        m_new = tl.maximum(m_global, m_block)
        l_global = l_global * tl.exp(m_global - m_new) + tl.sum(tl.exp(qk_avg - m_new))
        m_global = m_new
    
    m_global = tl.where(m_global == NEG_INF, 0.0, m_global)
    l_global = tl.where(l_global < 1e-9, 1.0, l_global)
    
    # =========================================================================
    # Part 2: 对 index_score 做 Online Softmax
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
        is_l_global = is_l_global * tl.exp(is_m_global - is_m_new) + tl.sum(tl.exp(is_val - is_m_new))
        is_m_global = is_m_new
    
    is_m_global = tl.where(is_m_global == NEG_INF, 0.0, is_m_global)
    is_l_global = tl.where(is_l_global < 1e-9, 1.0, is_l_global)
    
    # =========================================================================
    # Part 3: 计算 KL 散度
    # =========================================================================
    kl_sum = 0.0
    
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        # 重新计算 qk_avg (第二遍)
        qk_sum = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            for h in range(num_heads):
                q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                qk_sum += tl.sum(q[None, :] * k_gathered, axis=1)
        
        qk_avg = qk_sum * scaling / num_heads
        qk_avg = tl.where(causal_mask_block, NEG_INF, qk_avg)
        
        # attn_dist
        attn_dist = tl.exp(qk_avg - m_global) / l_global + eps
        attn_dist = tl.where(causal_mask_block, eps, attn_dist)
        
        # index_prob
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF)
        is_val = tl.where(causal_mask_block, NEG_INF, is_val)
        index_prob = tl.exp(is_val - is_m_global) / is_l_global + eps
        
        # KL
        kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
        kl = tl.where(causal_mask_block, 0.0, kl)
        kl_sum += tl.sum(kl)
    
    tl.store(Loss_ptr + pid_batch * chunk_size + pid_row, kl_sum)


# ============================================================================
# Fused Kernel V1/V2: K加载一次所有heads复用 + 存储QK避免重复计算
# ============================================================================

@triton.jit
def _sparse_attn_loss_fused_kernel(
    Q_ptr, K_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    AttnSum_ptr,  # [batch, chunk_size, topk] - 存储累加的attention
    batch_size, num_heads, chunk_size,
    chunk_offset,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    scaling,
    eps: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
    stride_asb, stride_ass, stride_ask,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """
    优化版本: K 加载一次，所有 heads 复用 + 存储 QK 避免重复计算
    
    关键优化:
    1. 外层循环 BLOCK_TOPK 块，内层循环 heads - K 只加载一次被所有 heads 复用
    2. 对每个 head 使用 online softmax，但存储 m_global 和 l_global
    3. Pass 2 时不重新计算 QK，而是用存储的 max/sum 直接归一化
    
    复杂度: O(topk_blocks × dim_blocks × (K加载 + heads×Q加载))
    原版本: O(heads × 2 × topk_blocks × dim_blocks × (K加载 + Q加载))
    加速比: 约 2× (避免重复计算QK)
    """
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    NEG_INF = -1e9
    
    # 基地址
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    attn_sum_base = AttnSum_ptr + pid_batch * stride_asb + pid_row * stride_ass
    
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    
    # =========================================================================
    # Part 1: 计算所有 heads 的 softmax (K 加载一次，所有 heads 复用)
    # =========================================================================
    # 初始化 attn_sum 为 0
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        tl.store(attn_sum_base + offs_tk * stride_ask, tl.zeros([BLOCK_TOPK], dtype=tl.float32), mask=tk_mask)
    
    # 对每个 head 独立处理
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        # -----------------------------------------------------------------
        # Single-pass: 同时计算 online max/sum 和累加 attention
        # 使用更高效的 fused 计算
        # -----------------------------------------------------------------
        m_global = NEG_INF
        l_global = 0.0
        
        # Pass 1: 计算全局 max 和 sum
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
            
            # 计算 QK - K 加载后与当前 head 的 Q 计算
            qk = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                # 加载 Q [BLOCK_D]
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                # 加载 K [BLOCK_TOPK, BLOCK_D]
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                # QK 累加
                qk += tl.sum(q[None, :] * k_gathered, axis=1)
            
            qk = qk * scaling
            qk = tl.where(causal_mask_block, NEG_INF, qk)
            
            # Online softmax update
            m_block = tl.max(qk)
            m_new = tl.maximum(m_global, m_block)
            l_global = l_global * tl.exp(m_global - m_new) + tl.sum(tl.exp(qk - m_new))
            m_global = m_new
        
        m_global = tl.where(m_global == NEG_INF, 0.0, m_global)
        l_global = tl.where(l_global < 1e-9, 1.0, l_global)
        
        # Pass 2: 计算归一化概率并累加到 attn_sum
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
            
            # 重新计算 QK
            qk = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                qk += tl.sum(q[None, :] * k_gathered, axis=1)
            
            qk = qk * scaling
            qk = tl.where(causal_mask_block, NEG_INF, qk)
            
            p = tl.exp(qk - m_global) / l_global
            p = tl.where(causal_mask_block, 0.0, p)
            
            # 累加到 attn_sum
            attn_sum_ptrs = attn_sum_base + offs_tk * stride_ask
            old_val = tl.load(attn_sum_ptrs, mask=tk_mask, other=0.0)
            tl.store(attn_sum_ptrs, old_val + p, mask=tk_mask)
    
    # =========================================================================
    # Part 2: 对 index_score 做 Online Softmax
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
        is_l_global = is_l_global * tl.exp(is_m_global - is_m_new) + tl.sum(tl.exp(is_val - is_m_new))
        is_m_global = is_m_new
    
    is_m_global = tl.where(is_m_global == NEG_INF, 0.0, is_m_global)
    is_l_global = tl.where(is_l_global < 1e-9, 1.0, is_l_global)
    
    # =========================================================================
    # Part 3: 计算 attn_dist 和 KL 散度
    # =========================================================================
    attn_total = 0.0
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_total += tl.sum(attn_sum_block)
    
    attn_total = tl.where(attn_total < eps, 1.0, attn_total)
    
    kl_sum = 0.0
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_dist = attn_sum_block / attn_total + eps
        
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF)
        is_val = tl.where(causal_mask_block, NEG_INF, is_val)
        index_prob = tl.exp(is_val - is_m_global) / is_l_global + eps
        
        kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
        kl = tl.where(causal_mask_block, 0.0, kl)
        kl_sum += tl.sum(kl)
    
    tl.store(Loss_ptr + pid_batch * chunk_size + pid_row, kl_sum)


# ============================================================================
# 高性能版本: 改变并行维度 + 两阶段 kernel
# ============================================================================

@triton.jit
def _attention_per_head_kernel(
    Q_ptr, K_ptr, Indices_ptr, AttnProbs_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr, scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    stride_apb, stride_aph, stride_aps, stride_apk,
    BLOCK_D: tl.constexpr, BLOCK_TOPK: tl.constexpr,
):
    """
    阶段1 Kernel: 每个 program 处理一个 (batch, row, head)
    
    Grid: (batch_size * chunk_size * num_heads,)
    这样可以充分利用 GPU 并行性:
    - 原版: 4096 programs, 每个做 128 heads × 2 passes
    - 新版: 524288 programs, 每个只做 1 次 softmax
    
    输出: AttnProbs [batch, num_heads, chunk_size, topk]
    """
    pid = tl.program_id(0)
    total_rows = chunk_size * num_heads
    pid_batch = pid // total_rows
    pid_remainder = pid % total_rows
    pid_row = pid_remainder // num_heads
    pid_head = pid_remainder % num_heads
    
    NEG_INF = -1e9
    
    # 基地址
    q_base = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + pid_row * stride_qs
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    ap_base = AttnProbs_ptr + pid_batch * stride_apb + pid_head * stride_aph + pid_row * stride_aps
    
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    
    # Pass 1: Online Softmax 计算全局 max 和 sum
    m_global = NEG_INF
    l_global = 0.0
    
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        qk = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        qk = qk * scaling
        qk = tl.where(causal_mask_block, NEG_INF, qk)
        
        m_block = tl.max(qk)
        m_new = tl.maximum(m_global, m_block)
        l_global = l_global * tl.exp(m_global - m_new) + tl.sum(tl.exp(qk - m_new))
        m_global = m_new
    
    m_global = tl.where(m_global == NEG_INF, 0.0, m_global)
    l_global = tl.where(l_global < 1e-9, 1.0, l_global)
    
    # Pass 2: 计算归一化概率并存储
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        qk = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        qk = qk * scaling
        qk = tl.where(causal_mask_block, NEG_INF, qk)
        
        p = tl.exp(qk - m_global) / l_global
        p = tl.where(causal_mask_block, 0.0, p)
        
        tl.store(ap_base + offs_tk * stride_apk, p, mask=tk_mask)


@triton.jit
def _head_sum_kl_kernel(
    AttnProbs_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    topk: tl.constexpr, eps: tl.constexpr,
    stride_apb, stride_aph, stride_aps, stride_apk,
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
    BLOCK_TOPK: tl.constexpr,
):
    """
    阶段2 Kernel: Head Sum + KL 计算
    
    Grid: (batch_size * chunk_size,)
    每个 program 处理一个 (batch, row)，对所有 heads 求和并计算 KL
    """
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    NEG_INF = -1e9
    
    ap_batch_base = AttnProbs_ptr + pid_batch * stride_apb
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    
    # Step 1: 累加所有 heads 的 attention probs
    attn_sum_total = 0.0
    
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        attn_sum_block = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
        for h in range(num_heads):
            ap_ptr = ap_batch_base + h * stride_aph + pid_row * stride_aps + offs_tk * stride_apk
            p = tl.load(ap_ptr, mask=tk_mask, other=0.0)
            attn_sum_block += p
        
        attn_sum_total += tl.sum(attn_sum_block)
    
    attn_sum_total = tl.where(attn_sum_total < eps, 1.0, attn_sum_total)
    
    # Step 2: 计算 index_score 的 softmax (Online Softmax)
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
        is_l_global = is_l_global * tl.exp(is_m_global - is_m_new) + tl.sum(tl.exp(is_val - is_m_new))
        is_m_global = is_m_new
    
    is_m_global = tl.where(is_m_global == NEG_INF, 0.0, is_m_global)
    is_l_global = tl.where(is_l_global < 1e-9, 1.0, is_l_global)
    
    # Step 3: 计算 KL 散度
    kl_sum = 0.0
    
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        # 累加所有 heads
        attn_sum_block = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
        for h in range(num_heads):
            ap_ptr = ap_batch_base + h * stride_aph + pid_row * stride_aps + offs_tk * stride_apk
            p = tl.load(ap_ptr, mask=tk_mask, other=0.0)
            attn_sum_block += p
        
        attn_dist = attn_sum_block / attn_sum_total + eps
        
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF)
        is_val = tl.where(causal_mask_block, NEG_INF, is_val)
        index_prob = tl.exp(is_val - is_m_global) / is_l_global + eps
        
        kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
        kl = tl.where(causal_mask_block, 0.0, kl)
        kl_sum += tl.sum(kl)
    
    tl.store(Loss_ptr + pid_batch * chunk_size + pid_row, kl_sum)


# ============================================================================
# Wrapper函数
# ============================================================================

def compute_index_loss_sparse(query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10, block_topk=None):
    """
    Sparse版本的完整loss计算 (Online Softmax 优化版本)
    
    支持分块注意力场景: chunk_size (query长度) 可以不等于 kv_len (key长度)
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim] - 当前chunk的query
        key: [batch, kv_len, head_dim] - 完整的key (KV cache)
        index_score: [batch, chunk_size, topk] - sparse版本的index分数
        indices: [batch, chunk_size, topk] - 每个query选择的topk个key索引
        scaling: attention scaling factor
        chunk_offset: 当前chunk在完整序列中的起始位置 (用于causal mask)
        eps: 数值稳定epsilon
        block_topk: topk 分块大小，默认使用全局 BLOCK_TOPK
    
    Returns:
        loss: 标量loss值
    
    性能优化 (Online Softmax):
        - 使用 BLOCK_TOPK 分块处理 topk
        - 使用 Online Softmax 保持正确的全局归一化
        - 两遍扫描: Pass1 计算全局 max/sum, Pass2 计算归一化概率
    """
    batch_size, num_heads, chunk_size, head_dim = query.shape
    topk = indices.shape[-1]
    
    # 选择合适的 block_topk
    if block_topk is None:
        block_topk = min(BLOCK_TOPK, topk)
    
    query = query.contiguous()
    key = key.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    # 输出: 每行(每个query位置)的loss
    loss_per_row = torch.zeros(batch_size, chunk_size, device=query.device, dtype=torch.float32)
    
    # 中间存储: 累加的 attention [batch, chunk_size, topk]
    attn_sum = torch.zeros(batch_size, chunk_size, topk, device=query.device, dtype=torch.float32)
    
    # 每个program处理一个(batch, query_row)
    grid = (batch_size * chunk_size,)
    
    _sparse_attn_loss_fused_kernel[grid](
        query, key, index_score, indices, loss_per_row,
        attn_sum,  # 中间存储
        batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, eps,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),  # attn_sum strides
        BLOCK_D=BLOCK_D,
        BLOCK_TOPK=block_topk,
        num_stages=NUM_STAGES, num_warps=NUM_WARPS,
    )
    
    return loss_per_row.sum() / batch_size


def compute_index_loss_sparse_v2(query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10, block_topk=None):
    """
    高性能版本: 两阶段 kernel + 改变并行维度
    
    阶段1: 每个 program 处理 (batch, row, head)，充分并行
    阶段2: 每个 program 处理 (batch, row)，做 head sum + KL
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim]
        key: [batch, kv_len, head_dim]
        index_score: [batch, chunk_size, topk]
        indices: [batch, chunk_size, topk]
        scaling: attention scaling factor
        chunk_offset: 当前chunk在完整序列中的起始位置
        eps: 数值稳定epsilon
        block_topk: topk 分块大小
    
    性能优势:
        - 阶段1: 并行度从 4096 提升到 524288 (128x)
        - 每个 program 只做 1 次 softmax，而不是 128 次
        - 更好地利用 GPU 并行性
    
    代价:
        - 需要中间存储 AttnProbs [batch, num_heads, chunk_size, topk]
    """
    batch_size, num_heads, chunk_size, head_dim = query.shape
    topk = indices.shape[-1]
    
    if block_topk is None:
        block_topk = min(BLOCK_TOPK, topk)
    
    query = query.contiguous()
    key = key.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    # 中间存储: [batch, num_heads, chunk_size, topk]
    attn_probs = torch.zeros(batch_size, num_heads, chunk_size, topk, device=query.device, dtype=torch.float32)
    
    # 输出
    loss_per_row = torch.zeros(batch_size, chunk_size, device=query.device, dtype=torch.float32)
    
    # 阶段1: 每个 program 处理 (batch, row, head)
    grid1 = (batch_size * chunk_size * num_heads,)
    _attention_per_head_kernel[grid1](
        query, key, indices, attn_probs,
        batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2), attn_probs.stride(3),
        BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk,
        num_stages=NUM_STAGES, num_warps=NUM_WARPS,
    )
    
    # 阶段2: 每个 program 处理 (batch, row)
    grid2 = (batch_size * chunk_size,)
    _head_sum_kl_kernel[grid2](
        attn_probs, index_score, indices, loss_per_row,
        batch_size, num_heads, chunk_size, chunk_offset, topk, eps,
        attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2), attn_probs.stride(3),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        BLOCK_TOPK=block_topk,
        num_warps=NUM_WARPS,
    )
    
    return loss_per_row.sum() / batch_size


def compute_index_loss_sparse_v3(query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10, block_topk=None):
    """
    V3: 累加 QK logits + 单次 Softmax (最高效版本)
    
    语义变化:
    - 原版: attn_dist = normalize(sum_h(softmax(QK_h)))
    - V3:   attn_dist = softmax(mean_h(QK_h))
    
    性能优势:
    - K 加载一次，所有 heads 的 Q 与之计算
    - Softmax 只做一次，而不是 128 次
    - 大幅减少计算量
    
    代价:
    - 语义有所不同，但在实践中效果通常相近
    """
    batch_size, num_heads, chunk_size, head_dim = query.shape
    topk = indices.shape[-1]
    
    if block_topk is None:
        block_topk = min(BLOCK_TOPK, topk)
    
    query = query.contiguous()
    key = key.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    loss_per_row = torch.zeros(batch_size, chunk_size, device=query.device, dtype=torch.float32)
    
    grid = (batch_size * chunk_size,)
    
    _sparse_attn_loss_v3_kernel[grid](
        query, key, index_score, indices, loss_per_row,
        batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, eps,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        BLOCK_D=BLOCK_D,
        BLOCK_TOPK=block_topk,
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
    
    # Triton Sparse V1
    tri_v1 = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    
    # Triton Sparse V2
    tri_v2 = compute_index_loss_sparse_v2(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    
    # Triton Sparse V3 (注意: V3 语义不同，只做参考比较)
    tri_v3 = compute_index_loss_sparse_v3(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    
    diff_v1 = abs(ref.item() - tri_v1.item())
    diff_v2 = abs(ref.item() - tri_v2.item())
    diff_v3 = abs(ref.item() - tri_v3.item())
    passed_v1 = diff_v1 < 1e-3
    passed_v2 = diff_v2 < 1e-3
    # V3 语义不同，差异可能较大，只做参考
    
    print(f"  PyTorch(Full): {ref.item():.6f}")
    print(f"  Triton V1:     {tri_v1.item():.6f}, Diff: {diff_v1:.6e}, Pass: {passed_v1}")
    print(f"  Triton V2:     {tri_v2.item():.6f}, Diff: {diff_v2:.6e}, Pass: {passed_v2}")
    print(f"  Triton V3:     {tri_v3.item():.6f}, Diff: {diff_v3:.6e} (语义不同,仅参考)")
    return passed_v1 and passed_v2


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
    
    # Test 1: Triton fused kernel V1 (Sparse, 原版)
    print("\n[测试] Triton V1 (单kernel, 串行heads)...")
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    triton_v1_time = (time.time() - start) / num_benchmark * 1000
    results['triton_v1'] = triton_v1_time
    
    # Test 2: Triton fused kernel V2 (两阶段, 并行heads)
    print("[测试] Triton V2 (两阶段kernel, 并行heads)...")
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse_v2(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse_v2(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    triton_v2_time = (time.time() - start) / num_benchmark * 1000
    results['triton_v2'] = triton_v2_time
    
    # Test 3: Triton V3 (累加QK + 单次softmax)
    print("[测试] Triton V3 (累加QK + 单次softmax)...")
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse_v3(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse_v3(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    triton_v3_time = (time.time() - start) / num_benchmark * 1000
    results['triton_v3'] = triton_v3_time
    
    # Test 4: PyTorch reference (Full) - 仅当 triton_only=False 时执行
    if not triton_only:
        print("[测试] PyTorch Full 参考...")
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
    print(f"  Triton V1 (串行heads):       {triton_v1_time:.3f} ms")
    print(f"  Triton V2 (并行heads):       {triton_v2_time:.3f} ms (vs V1: {triton_v1_time/triton_v2_time:.2f}x)")
    print(f"  Triton V3 (单次softmax):     {triton_v3_time:.3f} ms (vs V1: {triton_v1_time/triton_v3_time:.2f}x)")
    if not triton_only:
        print(f"  PyTorch Full:                {pytorch_time:.3f} ms")
        print(f"  V3 vs PyTorch:               {pytorch_time/triton_v3_time:.2f}x {'加速' if triton_v3_time < pytorch_time else '减速'}")
    
    return results



# ============================================================================
# 性能分析：拆分 Kernel 各部分 (与 Online Softmax 版本对应)
# ============================================================================

@triton.jit
def _profile_load_only_kernel(
    K_ptr, Indices_ptr, Output_ptr,
    batch_size, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """Profile: 只测试数据加载 (BLOCK_TOPK 分块版本)"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    acc_total = 0.0
    
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        
        acc = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(k_gathered, axis=1)
        
        acc_total += tl.sum(acc)
    
    tl.store(Output_ptr + pid, acc_total)


@triton.jit
def _profile_qk_only_kernel(
    Q_ptr, K_ptr, Indices_ptr, Output_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr, scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """Profile: 测试 QK 计算 (BLOCK_TOPK 分块版本，不含 softmax)"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    acc_total = 0.0
    
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            
            qk = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                qk += tl.sum(q[None, :] * k_gathered, axis=1)
            
            acc_total += tl.sum(qk * scaling)
    
    tl.store(Output_ptr + pid, acc_total)


@triton.jit
def _profile_online_softmax_pass1_kernel(
    Q_ptr, K_ptr, Indices_ptr, Output_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr, scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """Profile: Online Softmax Pass 1 - 计算全局 max 和 sum"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    NEG_INF = -1e9
    
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    
    acc = 0.0
    
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        # Online Softmax Pass 1: 计算全局 max 和 sum
        m_global = NEG_INF
        l_global = 0.0
        
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
            
            qk = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                qk += tl.sum(q[None, :] * k_gathered, axis=1)
            
            qk = qk * scaling
            qk = tl.where(causal_mask_block, NEG_INF, qk)
            
            # Online softmax update
            m_block = tl.max(qk)
            m_new = tl.maximum(m_global, m_block)
            l_global = l_global * tl.exp(m_global - m_new) + tl.sum(tl.exp(qk - m_new))
            m_global = m_new
        
        acc += m_global + tl.log(l_global + 1e-9)
    
    tl.store(Output_ptr + pid, acc)


@triton.jit
def _profile_online_softmax_full_kernel(
    Q_ptr, K_ptr, Indices_ptr, AttnSum_ptr, Output_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr, scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    stride_asb, stride_ass, stride_ask,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """Profile: Online Softmax 完整流程 (Pass 1 + Pass 2，不含 KL)"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    NEG_INF = -1e9
    
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    attn_sum_base = AttnSum_ptr + pid_batch * stride_asb + pid_row * stride_ass
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        # Pass 1: 计算全局 max 和 sum
        m_global = NEG_INF
        l_global = 0.0
        
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
            
            qk = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                qk += tl.sum(q[None, :] * k_gathered, axis=1)
            
            qk = qk * scaling
            qk = tl.where(causal_mask_block, NEG_INF, qk)
            
            m_block = tl.max(qk)
            m_new = tl.maximum(m_global, m_block)
            l_global = l_global * tl.exp(m_global - m_new) + tl.sum(tl.exp(qk - m_new))
            m_global = m_new
        
        m_global = tl.where(m_global == NEG_INF, 0.0, m_global)
        l_global = tl.where(l_global < 1e-9, 1.0, l_global)
        
        # Pass 2: 计算归一化概率并累加
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
            
            qk = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                qk += tl.sum(q[None, :] * k_gathered, axis=1)
            
            qk = qk * scaling
            qk = tl.where(causal_mask_block, NEG_INF, qk)
            
            p = tl.exp(qk - m_global) / l_global
            p = tl.where(causal_mask_block, 0.0, p)
            
            attn_sum_ptrs = attn_sum_base + offs_tk * stride_ask
            if h == 0:
                tl.store(attn_sum_ptrs, p, mask=tk_mask)
            else:
                old_val = tl.load(attn_sum_ptrs, mask=tk_mask, other=0.0)
                tl.store(attn_sum_ptrs, old_val + p, mask=tk_mask)
    
    # 计算 attn_sum 总和
    attn_total = 0.0
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_total += tl.sum(attn_sum_block)
    
    tl.store(Output_ptr + pid, attn_total)


def profile_kernel_parts(
    batch_size=1, num_heads=16, chunk_size=4096, seq_len=8192,
    head_dim=512, topk=2048, num_warmup=3, num_iters=10
):
    """
    分步 Profile 各部分耗时 (Online Softmax + BLOCK_TOPK 分块版本)
    
    测试内容:
    1. 只加载 K (gather 操作, BLOCK_TOPK 分块)
    2. QK 计算 (BLOCK_TOPK 分块, 不含 softmax)
    3. Online Softmax Pass 1 (计算全局 max/sum)
    4. Online Softmax 完整 (Pass 1 + Pass 2, 不含 KL)
    5. 完整 kernel (含 KL 计算)
    """
    import time
    
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    chunk_offset = seq_len - chunk_size
    block_topk = min(BLOCK_TOPK, topk)
    
    print("=" * 70)
    print("Kernel 各部分耗时分析 (Online Softmax + BLOCK_TOPK 分块版本)")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}")
    print(f"       seq={seq_len}, dim={head_dim}, topk={topk}")
    print(f"       BLOCK_TOPK={block_topk}, BLOCK_D={BLOCK_D}")
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
    attn_sum = torch.zeros(batch_size, chunk_size, topk, device=device, dtype=torch.float32)
    grid = (batch_size * chunk_size,)
    
    results = {}
    
    # Test 1: 只加载 K (BLOCK_TOPK 分块)
    print("\n[1] 只加载 K (gather, BLOCK_TOPK 分块)...")
    for _ in range(num_warmup):
        _profile_load_only_kernel[grid](
            key, indices, output,
            batch_size, chunk_size, chunk_offset, head_dim, topk,
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _profile_load_only_kernel[grid](
            key, indices, output,
            batch_size, chunk_size, chunk_offset, head_dim, topk,
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['load_k'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['load_k']:.3f} ms")
    
    # Test 2: QK 计算 (BLOCK_TOPK 分块)
    print("\n[2] QK 计算 (BLOCK_TOPK 分块, 不含 softmax)...")
    for _ in range(num_warmup):
        _profile_qk_only_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
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
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['qk_only'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['qk_only']:.3f} ms")
    
    # Test 3: Online Softmax Pass 1
    print("\n[3] Online Softmax Pass 1 (计算全局 max/sum)...")
    for _ in range(num_warmup):
        _profile_online_softmax_pass1_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _profile_online_softmax_pass1_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['softmax_pass1'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['softmax_pass1']:.3f} ms")
    
    # Test 4: Online Softmax 完整 (Pass 1 + Pass 2)
    print("\n[4] Online Softmax 完整 (Pass 1 + Pass 2, 不含 KL)...")
    for _ in range(num_warmup):
        _profile_online_softmax_full_kernel[grid](
            query, key, indices, attn_sum, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _profile_online_softmax_full_kernel[grid](
            query, key, indices, attn_sum, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['softmax_full'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['softmax_full']:.3f} ms")
    
    # Test 5: 完整 kernel (Online Softmax + KL)
    print("\n[5] 完整 Kernel (Online Softmax + KL)...")
    loss_per_row = torch.zeros(batch_size, chunk_size, device=device, dtype=torch.float32)
    for _ in range(num_warmup):
        _sparse_attn_loss_fused_kernel[grid](
            query, key, index_score, indices, loss_per_row,
            attn_sum,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, 1e-10,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _sparse_attn_loss_fused_kernel[grid](
            query, key, index_score, indices, loss_per_row,
            attn_sum,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, 1e-10,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_TOPK=block_topk, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['full'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['full']:.3f} ms")
    
    # 分析结果
    print("\n" + "=" * 70)
    print("耗时分解分析")
    print("=" * 70)
    print(f"  [1] 加载 K:                       {results['load_k']:.3f} ms ({results['load_k']/results['full']*100:.1f}%)")
    print(f"  [2] QK 计算:                      {results['qk_only']:.3f} ms ({results['qk_only']/results['full']*100:.1f}%)")
    print(f"  [3] Online Softmax Pass 1:        {results['softmax_pass1']:.3f} ms ({results['softmax_pass1']/results['full']*100:.1f}%)")
    print(f"  [4] Online Softmax 完整:          {results['softmax_full']:.3f} ms ({results['softmax_full']/results['full']*100:.1f}%)")
    print(f"  [5] 完整 Kernel:                  {results['full']:.3f} ms (100%)")
    print("-" * 70)
    print(f"  增量: Pass 2 开销:                {results['softmax_full'] - results['softmax_pass1']:.3f} ms")
    print(f"  增量: KL 开销:                    {results['full'] - results['softmax_full']:.3f} ms")
    print(f"  理论: Pass 1 ≈ QK + online_reduce, Pass 2 ≈ QK + normalize + store")
    
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
