import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

# 固定配置常量
BLOCK_D = 128
BLOCK_TOPK = 256  # topk 分块大小
BLOCK_H = 16      # head 分块大小 (满足 tl.dot 的 N >= 16 要求)
NUM_STAGES = 3
NUM_WARPS = 8


@triton.jit
def _indexer_loss_fwd_kernel_opt(
    Q_ptr, K_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    # 中间存储指针 (用于两遍扫描)
    AttnSum_ptr,  # [batch, chunk_size, topk] - 存储累加的attention
    QKCache_ptr,  # [batch, chunk_size, topk, BLOCK_H] - 缓存 QK 结果
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
    stride_asb, stride_ass, stride_ask,  # AttnSum strides
    stride_qkc_b, stride_qkc_s, stride_qkc_k, stride_qkc_h,  # QKCache strides
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Sparse Attention + Loss Kernel (QK 缓存优化版本)
    
    优化点：
    - 在 Pass 1 中计算 QK 后缓存到 QKCache_ptr
    - 在 Pass 2 中直接从 QKCache_ptr 读取，避免重复计算
    
    使用 tl.dot 利用 Tensor Core，一次处理 BLOCK_H 个 head：
    - qT: [BLOCK_D, BLOCK_H] - 转置形式
    - k_gathered: [BLOCK_TOPK, BLOCK_D]
    - qk = tl.dot(k_gathered, qT) -> [BLOCK_TOPK, BLOCK_H]
    
    Online Softmax 核心公式 (对每个 head 独立计算):
      m_new = max(m_old, max(block))
      l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))
      最终: p[i] = exp(x[i] - m_final) / l_final
    """
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    NEG_INF = -1e9
    
    # 基地址
    q_batch_base = Q_ptr + pid_batch * stride_qb + pid_row * stride_qs
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    attn_sum_base = AttnSum_ptr + pid_batch * stride_asb + pid_row * stride_ass
    qk_cache_base = QKCache_ptr + pid_batch * stride_qkc_b + pid_row * stride_qkc_s
    
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    num_head_blocks = tl.cdiv(num_heads, BLOCK_H)
    
    # head 偏移和 dim 偏移
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    
    # =========================================================================
    # Part 1: 按 BLOCK_H 分组处理 head，使用 Online Softmax 计算 attention
    # =========================================================================
    
    for h_block in range(num_head_blocks):
        h_start = h_block * BLOCK_H
        h_mask = (h_start + offs_h) < num_heads
        
        # -----------------------------------------------------------------
        # Pass 1: 计算全局 max 和 sum (Online Softmax) - 每个 head 独立
        #         同时将 QK 结果缓存到 QKCache
        # -----------------------------------------------------------------
        # m_global, l_global: [BLOCK_H] - 每个 head 一个值
        m_global = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
        l_global = tl.zeros([BLOCK_H], dtype=tl.float32)
        
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            # 加载 indices 并计算 causal mask: [BLOCK_TOPK]
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)  # [BLOCK_TOPK]
            
            # 分块计算 QK (支持 head_dim > BLOCK_D)
            qk = tl.zeros([BLOCK_TOPK, BLOCK_H], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d_block = d_start + offs_d
                d_block_mask = offs_d_block < head_dim
                
                # 加载 qT: [BLOCK_D, BLOCK_H] - 一次加载 BLOCK_H 个 head 的 q (转置形式)
                q_ptrs = q_batch_base + (h_start + offs_h[None, :]) * stride_qh + offs_d_block[:, None] * stride_qd
                qT = tl.load(q_ptrs, mask=d_block_mask[:, None] & h_mask[None, :], other=0.0)  # [BLOCK_D, BLOCK_H]
                
                # 加载 k_gathered: [BLOCK_TOPK, BLOCK_D]
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d_block[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_block_mask[None, :], other=0.0)  # [BLOCK_TOPK, BLOCK_D]
                
                # 累加 QK: [BLOCK_TOPK, BLOCK_H]
                qk += tl.dot(k_gathered, qT)
            
            qk = qk * scaling
            
            # ============= 缓存 QK 结果 (优化点) =============
            # 存储到 QKCache: [batch, chunk_size, topk, BLOCK_H]
            # 注意: 这里存储的是 scaled QK，还未应用 mask
            qk_cache_ptrs = qk_cache_base + offs_tk[:, None] * stride_qkc_k + offs_h[None, :] * stride_qkc_h
            tl.store(qk_cache_ptrs, qk, mask=tk_mask[:, None] & h_mask[None, :])
            # ================================================
            
            # 应用 causal mask 和 head mask: 对被 mask 的位置设为 NEG_INF
            # causal_mask_block[:, None]: [BLOCK_TOPK, 1] -> 广播到 [BLOCK_TOPK, BLOCK_H]
            # h_mask[None, :]: [1, BLOCK_H] -> 广播到 [BLOCK_TOPK, BLOCK_H]
            invalid_mask = causal_mask_block[:, None] | (~h_mask[None, :])
            qk = tl.where(invalid_mask, NEG_INF, qk)  # [BLOCK_TOPK, BLOCK_H]
            
            # Online softmax update - 对每个 head 独立
            m_block = tl.max(qk, axis=0)  # [BLOCK_H]
            m_new = tl.maximum(m_global, m_block)  # [BLOCK_H]
            # 修正旧的 sum，并加上新块的 exp
            # 关键修复: 在exp之后将无效位置显式设为0，避免 exp(NEG_INF - NEG_INF) = 1 的问题
            exp_qk = tl.exp(qk - m_new[None, :])
            exp_qk = tl.where(invalid_mask, 0.0, exp_qk)
            l_global = l_global * tl.exp(m_global - m_new) + tl.sum(exp_qk, axis=0)
            m_global = m_new
        
        # 处理全 NEG_INF 情况
        m_global = tl.where(m_global == NEG_INF, 0.0, m_global)  # [BLOCK_H]
        l_global = tl.where(l_global < 1e-9, 1.0, l_global)  # [BLOCK_H]
        
        # -----------------------------------------------------------------
        # Pass 2: 从缓存读取 QK，使用全局 max/sum 计算归一化概率，累加到 attn_sum
        # -----------------------------------------------------------------
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)  # [BLOCK_TOPK]
            
            # ============= 从缓存读取 QK (优化点) =============
            # 从 QKCache 读取已缓存的 QK 结果
            qk_cache_ptrs = qk_cache_base + offs_tk[:, None] * stride_qkc_k + offs_h[None, :] * stride_qkc_h
            qk = tl.load(qk_cache_ptrs, mask=tk_mask[:, None] & h_mask[None, :], other=0.0)
            # =================================================
            
            # 应用 causal mask 和 head mask
            invalid_mask = causal_mask_block[:, None] | (~h_mask[None, :])
            qk = tl.where(invalid_mask, NEG_INF, qk)
            
            # 使用全局 max/sum 归一化: [BLOCK_TOPK, BLOCK_H]
            p = tl.exp(qk - m_global[None, :]) / l_global[None, :]
            p = tl.where(invalid_mask, 0.0, p)  # 对 padding head 和 causal masked 位置设为 0
            
            # 对所有有效 head 求和: [BLOCK_TOPK]
            p_sum = tl.sum(p, axis=1)
            
            # 累加到 attn_sum
            attn_sum_ptrs = attn_sum_base + offs_tk * stride_ask
            if h_block == 0:
                tl.store(attn_sum_ptrs, p_sum, mask=tk_mask)
            else:
                old_val = tl.load(attn_sum_ptrs, mask=tk_mask, other=0.0)
                tl.store(attn_sum_ptrs, old_val + p_sum, mask=tk_mask)
    
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
        # 关键修复: 在exp之后将无效位置显式设为0
        is_exp_val = tl.exp(is_val - is_m_new)
        is_exp_val = tl.where(causal_mask_block, 0.0, is_exp_val)
        is_l_global = is_l_global * tl.exp(is_m_global - is_m_new) + tl.sum(is_exp_val)
        is_m_global = is_m_new
    
    is_m_global = tl.where(is_m_global == NEG_INF, 0.0, is_m_global)
    is_l_global = tl.where(is_l_global < 1e-9, 1.0, is_l_global)
    
    # =========================================================================
    # Part 3: 计算 attn_dist 和 KL 散度
    # =========================================================================
    # 先计算 attn_sum 的总和 (用于归一化)
    attn_total = 0.0
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_total += tl.sum(attn_sum_block)
    
    attn_total = tl.where(attn_total < eps, 1.0, attn_total)
    
    # 计算 KL 散度
    kl_sum = 0.0
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        # 加载 attn_sum 并归一化
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_dist = attn_sum_block / attn_total + eps
        
        # 计算 index_prob
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF)
        is_val = tl.where(causal_mask_block, NEG_INF, is_val)
        index_prob = tl.exp(is_val - is_m_global) / is_l_global + eps
        
        # KL 散度
        kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
        kl = tl.where(causal_mask_block, 0.0, kl)
        kl_sum += tl.sum(kl)
    
    # 写出 loss
    tl.store(Loss_ptr + pid_batch * chunk_size + pid_row, kl_sum)


@triton.jit
def _indexer_loss_bwd_kernel(
    # 输入
    IndexScore_ptr,  # [batch, chunk_size, topk]
    Indices_ptr,     # [batch, chunk_size, topk]
    AttnSum_ptr,     # [batch, chunk_size, topk] - 前向保存的中间结果
    # 输出
    dIndexScore_ptr, # [batch, chunk_size, topk]
    # 标量参数
    batch_size, chunk_size, chunk_offset,
    topk: tl.constexpr,
    eps: tl.constexpr,
    # strides
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
    stride_asb, stride_ass, stride_ask,
    stride_disb, stride_diss, stride_disk,
    BLOCK_TOPK: tl.constexpr,
):
    """
    反向传播 kernel：计算 index_score 的梯度
    
    数学推导：
    Loss = KL(attn_dist || index_prob) = sum_k attn_dist_k * log(attn_dist_k / index_prob_k)
    
    由于 attn_dist 不依赖于 index_score（只依赖于 Q, K），所以：
    d Loss / d index_score_j = d/d(index_score_j) [ -sum_k attn_dist_k * log(index_prob_k) ]
    
    使用 softmax 的梯度公式：
    d Loss / d index_score_j = index_prob_j - attn_dist_j
    """
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    NEG_INF = -1e9
    global_query_pos = chunk_offset + pid_row
    num_topk_blocks = tl.cdiv(topk, BLOCK_TOPK)
    
    # 基地址
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    attn_sum_base = AttnSum_ptr + pid_batch * stride_asb + pid_row * stride_ass
    dis_base = dIndexScore_ptr + pid_batch * stride_disb + pid_row * stride_diss
    
    # =========================================================================
    # Step 1: 计算 attn_total (attn_sum 的总和，用于归一化成 attn_dist)
    # =========================================================================
    attn_total = 0.0
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_total += tl.sum(attn_sum_block)
    
    attn_total = tl.where(attn_total < eps, 1.0, attn_total)
    
    # =========================================================================
    # Step 2: 对 index_score 做 Online Softmax 得到 index_prob
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
        is_exp_val = tl.exp(is_val - is_m_new)
        is_exp_val = tl.where(causal_mask_block, 0.0, is_exp_val)
        is_l_global = is_l_global * tl.exp(is_m_global - is_m_new) + tl.sum(is_exp_val)
        is_m_global = is_m_new
    
    is_m_global = tl.where(is_m_global == NEG_INF, 0.0, is_m_global)
    is_l_global = tl.where(is_l_global < 1e-9, 1.0, is_l_global)
    
    # =========================================================================
    # Step 3: 计算梯度 grad_index_score = index_prob - attn_dist
    # =========================================================================
    for tk_idx in range(num_topk_blocks):
        tk_start = tk_idx * BLOCK_TOPK
        offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
        tk_mask = offs_tk < topk
        
        indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
        causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
        
        # 计算 attn_dist = attn_sum / attn_total
        attn_sum_block = tl.load(attn_sum_base + offs_tk * stride_ask, mask=tk_mask, other=0.0)
        attn_dist = attn_sum_block / attn_total
        
        # 计算 index_prob = softmax(index_score)
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF)
        is_val = tl.where(causal_mask_block, NEG_INF, is_val)
        index_prob = tl.exp(is_val - is_m_global) / is_l_global
        
        # 梯度 = index_prob - attn_dist
        grad = index_prob - attn_dist
        grad = tl.where(causal_mask_block, 0.0, grad)
        
        # 写出梯度
        dis_ptrs = dis_base + offs_tk * stride_disk
        tl.store(dis_ptrs, grad, mask=tk_mask)


class IndexerLossFunctionOpt(torch.autograd.Function):
    """
    自定义 autograd Function，实现稀疏注意力索引损失的前向和反向传播 (QK 缓存优化版本)
    
    前向传播:
        1. 计算 sparse attention (Q @ K[indices]) 并应用 online softmax
        2. 缓存 QK 结果到 qk_cache，避免 Pass 2 重复计算
        3. 将所有 head 的 attention 累加到 attn_sum
        4. 对 index_score 做 softmax 得到 index_prob
        5. 计算 KL(attn_dist || index_prob) 作为损失
    
    反向传播:
        由于 attn_dist 不依赖于 index_score（只依赖于 Q, K），所以：
        grad_index_score = index_prob - attn_dist
    """
    
    @staticmethod
    def forward(ctx, query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10, block_topk=None):
        """
        前向传播 (QK 缓存优化版本)
        
        Args:
            query: [batch, num_heads, chunk_size, head_dim]
            key: [batch, kv_len, head_dim]
            index_score: [batch, chunk_size, topk] - 稀疏版本的 index 分数
            indices: [batch, chunk_size, topk] - 每个 query 选择的 topk 个 key 索引
            scaling: attention scaling factor
            chunk_offset: 当前 chunk 在完整序列中的起始位置 (用于 causal mask)
            eps: 数值稳定 epsilon
            block_topk: topk 分块大小
        
        Returns:
            loss: 标量 loss 值
        """
        batch_size, num_heads, chunk_size, head_dim = query.shape
        topk = indices.shape[-1]
        
        # 选择合适的 block_topk (确保 >= 16 以满足 tl.dot 要求)
        if block_topk is None:
            block_topk = min(BLOCK_TOPK, topk)
        if block_topk < 16:
            block_topk = 16
        
        # 选择合适的 block_h (确保 >= 16 以满足 tl.dot 要求)
        block_h = min(BLOCK_H, num_heads)
        if block_h < 16:
            block_h = 16
        
        query = query.contiguous()
        key = key.contiguous()
        index_score = index_score.contiguous()
        indices = indices.contiguous().to(torch.int64)
        
        # 输出: 每行(每个 query 位置)的 loss
        loss_per_row = torch.zeros(batch_size, chunk_size, device=query.device, dtype=torch.float32)
        
        # 中间存储: 累加的 attention [batch, chunk_size, topk]
        attn_sum = torch.zeros(batch_size, chunk_size, topk, device=query.device, dtype=torch.float32)
        
        # QK 缓存: [batch, chunk_size, topk, block_h]
        # 仅缓存当前 head_block 的 QK 结果，显存开销较小
        qk_cache = torch.zeros(batch_size, chunk_size, topk, block_h, device=query.device, dtype=torch.float32)
        
        # 每个 program 处理一个 (batch, query_row)
        grid = (batch_size * chunk_size,)
        
        _indexer_loss_fwd_kernel_opt[grid](
            query, key, index_score, indices, loss_per_row,
            attn_sum,  # 中间存储
            qk_cache,  # QK 缓存
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling, eps,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),
            qk_cache.stride(0), qk_cache.stride(1), qk_cache.stride(2), qk_cache.stride(3),
            BLOCK_D=BLOCK_D,
            BLOCK_TOPK=block_topk,
            BLOCK_H=block_h,
            num_stages=NUM_STAGES, num_warps=NUM_WARPS,
        )
        
        loss = loss_per_row.sum() / batch_size
        
        # 保存反向传播需要的张量
        ctx.save_for_backward(index_score, indices, attn_sum)
        ctx.chunk_offset = chunk_offset
        ctx.eps = eps
        ctx.batch_size = batch_size
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        只计算 index_score 的梯度 (Q, K 视为常量，不需要梯度)
        
        数学推导:
        Loss = KL(attn_dist || index_prob) = sum_k attn_dist_k * log(attn_dist_k / index_prob_k)
        
        d Loss / d index_score_j = index_prob_j - attn_dist_j
        """
        index_score, indices, attn_sum = ctx.saved_tensors
        chunk_offset = ctx.chunk_offset
        eps = ctx.eps
        batch_size = ctx.batch_size
        
        _, chunk_size, topk = index_score.shape
        
        # 选择合适的 block_topk
        block_topk = min(BLOCK_TOPK, topk)
        if block_topk < 16:
            block_topk = 16
        
        # 输出: index_score 的梯度
        grad_index_score = torch.zeros_like(index_score, dtype=torch.float32)
        
        # 每个 program 处理一个 (batch, query_row)
        grid = (batch_size * chunk_size,)
        
        _indexer_loss_bwd_kernel[grid](
            index_score, indices, attn_sum,
            grad_index_score,
            batch_size, chunk_size, chunk_offset, topk, eps,
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2),
            grad_index_score.stride(0), grad_index_score.stride(1), grad_index_score.stride(2),
            BLOCK_TOPK=block_topk,
            num_stages=NUM_STAGES, num_warps=NUM_WARPS,
        )
        
        # 乘以上游梯度并除以 batch_size (对应前向 loss = sum / batch_size)
        grad_index_score = grad_index_score * grad_output / batch_size
        
        # 返回对应输入的梯度
        # (query, key, index_score, indices, scaling, chunk_offset, eps, block_topk)
        return None, None, grad_index_score, None, None, None, None, None


def indexer_loss_opt(query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10, block_topk=None):
    """
    计算稀疏注意力索引损失 (Sparse Attention Index Loss) - QK 缓存优化版本
    
    该函数使用 Triton kernel 实现高效的稀疏注意力损失计算，支持自动梯度反向传播。
    通过缓存 QK 计算结果，避免在 Pass 2 中重复计算，提升性能。
    
    核心思想:
        1. 计算 sparse attention: softmax(Q @ K[indices] * scaling)
        2. 缓存 QK 结果，Pass 2 直接读取
        3. 将所有 head 的 attention 求和并归一化得到 attn_dist
        4. 对 index_score 做 softmax 得到 index_prob
        5. 计算 KL(attn_dist || index_prob) 作为损失
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim] - 当前 chunk 的 query
        key: [batch, kv_len, head_dim] - 完整的 key (KV cache)
        index_score: [batch, chunk_size, topk] - 稀疏版本的 index 分数，需要梯度
        indices: [batch, chunk_size, topk] - 每个 query 选择的 topk 个 key 索引
        scaling: attention scaling factor，通常为 1/sqrt(head_dim)
        chunk_offset: 当前 chunk 在完整序列中的起始位置 (用于 causal mask)，默认为 0
        eps: 数值稳定 epsilon，默认为 1e-10
        block_topk: topk 分块大小，默认自动选择
    
    Returns:
        loss: 标量 loss 值，表示 index_score 与实际 attention 分布之间的 KL 散度
    
    Example:
        >>> batch, heads, chunk_size, head_dim = 1, 8, 256, 64
        >>> seq_len, topk = 1024, 128
        >>> query = torch.randn(batch, heads, chunk_size, head_dim, device='cuda', dtype=torch.bfloat16)
        >>> key = torch.randn(batch, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
        >>> index_score = torch.randn(batch, chunk_size, topk, device='cuda', requires_grad=True)
        >>> indices = torch.randint(0, seq_len, (batch, chunk_size, topk), device='cuda')
        >>> scaling = 1.0 / (head_dim ** 0.5)
        >>> loss = indexer_loss_opt(query, key, index_score, indices, scaling, chunk_offset=seq_len - chunk_size)
        >>> loss.backward()  # index_score.grad 现在包含梯度
    """
    return IndexerLossFunctionOpt.apply(query, key, index_score, indices, scaling, chunk_offset, eps, block_topk)


# ============================================================================
# PyTorch 参考实现 (从原文件复用)
# ============================================================================

def pytorch_reference_fwd(query, key, index_score_full, index_mask, scaling):
    """
    PyTorch 参考实现: 前向传播 (Full 版本)
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim]
        key: [batch, kv_len, head_dim]
        index_score_full: [batch, chunk_size, kv_len] - Full 版本的 index 分数
        index_mask: [batch, 1, chunk_size, kv_len] - True 表示需要 mask 的位置
        scaling: attention scaling factor
    
    Returns:
        kl_loss: 标量 loss 值
    """
    eps = 1e-10
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    index_score_full = index_score_full.to(torch.float32)
    
    # 计算 attention: [batch, num_heads, chunk_size, kv_len]
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(index_mask, -1e9)
    attn = torch.softmax(attn, dim=-1)
    
    # Head sum + normalize
    attn_sum = attn.sum(dim=1)  # [batch, chunk_size, kv_len]
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    # Index score softmax
    index_mask_2d = index_mask.squeeze(1)  # [batch, chunk_size, kv_len]
    index_score_masked = index_score_full.masked_fill(index_mask_2d, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    # KL 散度
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


def pytorch_reference_bwd(query, key, index_score_full, index_mask, topk_indices, scaling):
    """
    PyTorch 参考实现: 使用 autograd 计算反向传播
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim]
        key: [batch, kv_len, head_dim]
        index_score_full: [batch, chunk_size, kv_len] - 需要梯度
        index_mask: [batch, 1, chunk_size, kv_len]
        topk_indices: [batch, chunk_size, topk]
        scaling: attention scaling factor
    
    Returns:
        grad_index_score_sparse: [batch, chunk_size, topk] - topk 位置的梯度
    """
    if not index_score_full.requires_grad:
        index_score_full = index_score_full.detach().requires_grad_(True)
    
    loss = pytorch_reference_fwd(query, key, index_score_full, index_mask, scaling)
    loss.backward()
    
    grad_full = index_score_full.grad  # [batch, chunk_size, kv_len]
    grad_sparse = torch.gather(grad_full, dim=-1, index=topk_indices)
    
    return grad_sparse


# ============================================================================
# 测试辅助函数
# ============================================================================

def generate_index_mask_from_score(index_score, topk, device='cuda', chunk_offset=0):
    """
    从 index_score 生成 index_mask 和 topk_indices
    
    Args:
        index_score: [batch, chunk_size, seq_len]
        topk: 每个 query 位置选择的 top-k 个 key
        device: 设备
        chunk_offset: 当前 chunk 在完整序列中的起始位置
    
    Returns:
        index_mask: [batch, 1, chunk_size, seq_len] - True 表示需要 mask 的位置
        topk_indices: [batch, chunk_size, topk]
    """
    batch_size, chunk_size, seq_len = index_score.shape
    
    # 创建 causal mask
    query_positions = chunk_offset + torch.arange(chunk_size, device=device).view(-1, 1)
    key_positions = torch.arange(seq_len, device=device).view(1, -1)
    causal_mask = key_positions > query_positions  # [chunk_size, seq_len]
    
    # 对 index_score 应用 causal mask
    causal_index_score = index_score.masked_fill(causal_mask, float('-inf'))
    
    # 取 topk
    topk_indices = causal_index_score.topk(topk, dim=-1)[1]
    
    # 创建 index_mask
    index_mask = torch.full(causal_index_score.shape, True, device=device)
    index_mask = index_mask.scatter_(-1, topk_indices, False)
    index_mask = torch.logical_or(index_mask, causal_mask)
    index_mask = index_mask.unsqueeze(1)  # [batch, 1, chunk_size, seq_len]
    
    return index_mask, topk_indices


# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class TestConfig:
    """测试配置"""
    name: str
    batch_size: int = 1
    num_heads: int = 8
    chunk_size: int = 256
    seq_len: int = 256
    head_dim: int = 64
    topk: int = 32
    seed: int = 42
    
    def __str__(self):
        return (f"batch={self.batch_size}, heads={self.num_heads}, "
                f"chunk={self.chunk_size}, seq={self.seq_len}, "
                f"dim={self.head_dim}, topk={self.topk}")


# ============================================================================
# 前向精度测试
# ============================================================================

def run_fwd_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """运行单个前向精度测试"""
    torch.manual_seed(config.seed)
    scaling = 1.0 / (config.head_dim ** 0.5)
    
    query = torch.randn(config.batch_size, config.num_heads, config.chunk_size, 
                        config.head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(config.batch_size, config.seq_len, config.head_dim, 
                      device=device, dtype=torch.bfloat16)
    index_score_full = torch.randn(config.batch_size, config.chunk_size, config.seq_len, 
                                   device=device, dtype=torch.bfloat16)
    
    chunk_offset = config.seq_len - config.chunk_size
    index_mask, topk_indices = generate_index_mask_from_score(
        index_score_full, config.topk, device, chunk_offset=chunk_offset)
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    # PyTorch 参考
    ref = pytorch_reference_fwd(query, key, index_score_full, index_mask, scaling)
    
    # Triton 优化版本
    tri = indexer_loss_opt(query, key, index_score_sparse, topk_indices, 
                           scaling, chunk_offset=chunk_offset)
    
    abs_diff = abs(ref.item() - tri.item())
    rel_diff = abs_diff / (abs(ref.item()) + 1e-10)
    passed = rel_diff < 1e-3
    
    return {
        'config': config,
        'ref': ref.item(),
        'tri': tri.item(),
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'passed': passed
    }


def test_fwd_accuracy(configs: List[TestConfig]):
    """批量运行前向精度测试"""
    print("\n" + "=" * 100)
    print("前向精度测试 (PyTorch Full vs Triton Sparse Optimized)")
    print("=" * 100)
    
    results = []
    for config in configs:
        result = run_fwd_accuracy_test(config)
        results.append(result)
    
    print(f"\n{'Name':<12} {'Config':<55} {'PyTorch':<12} {'Triton':<12} {'RelDiff':<12} {'Pass':<6}")
    print("-" * 109)
    for r in results:
        print(f"{r['config'].name:<12} {str(r['config']):<55} "
              f"{r['ref']:<12.6f} {r['tri']:<12.6f} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 109)
    print(f"前向测试: {passed_count}/{len(results)} 通过")
    
    return results


# ============================================================================
# 反向精度测试
# ============================================================================

def run_bwd_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """运行单个反向精度测试"""
    torch.manual_seed(config.seed)
    scaling = 1.0 / (config.head_dim ** 0.5)
    
    query = torch.randn(config.batch_size, config.num_heads, config.chunk_size, 
                        config.head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(config.batch_size, config.seq_len, config.head_dim, 
                      device=device, dtype=torch.bfloat16)
    index_score_full = torch.randn(config.batch_size, config.chunk_size, config.seq_len, 
                                   device=device, dtype=torch.float32, requires_grad=True)
    
    chunk_offset = config.seq_len - config.chunk_size
    index_mask, topk_indices = generate_index_mask_from_score(
        index_score_full.detach(), config.topk, device, chunk_offset=chunk_offset)
    index_score_sparse = torch.gather(index_score_full.detach(), dim=-1, index=topk_indices)
    index_score_sparse.requires_grad_(True)
    
    # PyTorch 参考: 使用 autograd 计算梯度
    ref_grad = pytorch_reference_bwd(query, key, index_score_full, index_mask, topk_indices, scaling)
    
    # Triton 优化版本: 使用自定义 backward
    loss = indexer_loss_opt(query, key, index_score_sparse, topk_indices, 
                            scaling, chunk_offset=chunk_offset)
    loss.backward()
    tri_grad = index_score_sparse.grad
    
    # 比较梯度
    abs_diff = (ref_grad - tri_grad).abs().max().item()
    rel_diff = abs_diff / (ref_grad.abs().max().item() + 1e-10)
    passed = rel_diff < 1e-2  # 由于精度累积，放宽到 1%
    
    return {
        'config': config,
        'ref_grad_max': ref_grad.abs().max().item(),
        'tri_grad_max': tri_grad.abs().max().item(),
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'passed': passed
    }


def test_bwd_accuracy(configs: List[TestConfig]):
    """批量运行反向精度测试"""
    print("\n" + "=" * 100)
    print("反向精度测试 (PyTorch autograd vs Triton kernel Optimized)")
    print("=" * 100)
    
    results = []
    for config in configs:
        result = run_bwd_accuracy_test(config)
        results.append(result)
    
    print(f"\n{'Name':<12} {'Config':<55} {'RefMax':<12} {'TriMax':<12} {'RelDiff':<12} {'Pass':<6}")
    print("-" * 109)
    for r in results:
        print(f"{r['config'].name:<12} {str(r['config']):<55} "
              f"{r['ref_grad_max']:<12.2e} {r['tri_grad_max']:<12.2e} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 109)
    print(f"反向测试: {passed_count}/{len(results)} 通过")
    
    return results


# ============================================================================
# 性能对比测试 (原版 vs 优化版)
# ============================================================================

def test_performance_comparison(
    batch_size: int = 1,
    num_heads: int = 16,
    chunk_size: int = 4 * 1024,
    seq_len: int = 8 * 1024,
    head_dim: int = 128,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 10,
    num_benchmark: int = 50,
):
    """性能对比测试: 原版 vs 优化版"""
    import time
    from indexer_loss_kernel import indexer_loss as indexer_loss_orig
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("\n" + "=" * 80)
    print("性能对比测试 (原版 vs QK缓存优化版)")
    print("=" * 80)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print("=" * 80)
    
    query = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    chunk_offset = seq_len - chunk_size
    index_score_full = torch.randn(batch_size, chunk_size, seq_len, device=device, dtype=torch.bfloat16)
    index_mask, topk_indices = generate_index_mask_from_score(index_score_full, topk, device, chunk_offset=chunk_offset)
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    results = {}
    memory_stats = {}
    
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated() / (1024**3)
    
    # 原版测试
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = indexer_loss_orig(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        _ = indexer_loss_orig(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    orig_time = (time.time() - start) / num_benchmark * 1000
    orig_peak = torch.cuda.max_memory_allocated() / (1024**3)
    results['original'] = orig_time
    memory_stats['original'] = orig_peak
    
    # 优化版测试
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = indexer_loss_opt(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        _ = indexer_loss_opt(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    opt_time = (time.time() - start) / num_benchmark * 1000
    opt_peak = torch.cuda.max_memory_allocated() / (1024**3)
    results['optimized'] = opt_time
    memory_stats['optimized'] = opt_peak
    
    print(f"\n>>> 前向性能 (warmup={num_warmup}, iters={num_benchmark})")
    print(f"  原版:       {orig_time:.3f} ms")
    print(f"  优化版:     {opt_time:.3f} ms (加速: {orig_time/opt_time:.2f}x)")
    
    print(f"\n>>> 显存峰值")
    print(f"  基准显存:   {base_memory:.2f} GB")
    print(f"  原版峰值:   {memory_stats['original']:.2f} GB (增量: {memory_stats['original'] - base_memory:.2f} GB)")
    print(f"  优化版峰值: {memory_stats['optimized']:.2f} GB (增量: {memory_stats['optimized'] - base_memory:.2f} GB)")
    
    return results, memory_stats


def test_fwd_bwd_performance_comparison(
    batch_size: int = 1,
    num_heads: int = 16,
    chunk_size: int = 4 * 1024,
    seq_len: int = 8 * 1024,
    head_dim: int = 128,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 10,
    num_benchmark: int = 50,
):
    """前向+反向性能对比测试: 原版 vs 优化版"""
    import time
    from indexer_loss_kernel import indexer_loss as indexer_loss_orig
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("\n" + "=" * 80)
    print("前向+反向性能对比测试 (原版 vs QK缓存优化版)")
    print("=" * 80)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print("=" * 80)
    
    query = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    chunk_offset = seq_len - chunk_size
    index_score_full = torch.randn(batch_size, chunk_size, seq_len, device=device, dtype=torch.float32)
    index_mask, topk_indices = generate_index_mask_from_score(index_score_full, topk, device, chunk_offset=chunk_offset)
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    results = {}
    memory_stats = {}
    
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated() / (1024**3)
    
    # 原版测试 (前向 + 反向)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        index_score_sparse_copy = index_score_sparse.clone().requires_grad_(True)
        loss = indexer_loss_orig(query, key, index_score_sparse_copy, topk_indices, scaling, chunk_offset=chunk_offset)
        loss.backward()
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        index_score_sparse_copy = index_score_sparse.clone().requires_grad_(True)
        loss = indexer_loss_orig(query, key, index_score_sparse_copy, topk_indices, scaling, chunk_offset=chunk_offset)
        loss.backward()
    torch.cuda.synchronize()
    orig_time = (time.time() - start) / num_benchmark * 1000
    orig_peak = torch.cuda.max_memory_allocated() / (1024**3)
    results['original_fwd_bwd'] = orig_time
    memory_stats['original'] = orig_peak
    
    # 优化版测试 (前向 + 反向)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        index_score_sparse_copy = index_score_sparse.clone().requires_grad_(True)
        loss = indexer_loss_opt(query, key, index_score_sparse_copy, topk_indices, scaling, chunk_offset=chunk_offset)
        loss.backward()
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        index_score_sparse_copy = index_score_sparse.clone().requires_grad_(True)
        loss = indexer_loss_opt(query, key, index_score_sparse_copy, topk_indices, scaling, chunk_offset=chunk_offset)
        loss.backward()
    torch.cuda.synchronize()
    opt_time = (time.time() - start) / num_benchmark * 1000
    opt_peak = torch.cuda.max_memory_allocated() / (1024**3)
    results['optimized_fwd_bwd'] = opt_time
    memory_stats['optimized'] = opt_peak
    
    print(f"\n>>> 前向+反向性能 (warmup={num_warmup}, iters={num_benchmark})")
    print(f"  原版:       {orig_time:.3f} ms")
    print(f"  优化版:     {opt_time:.3f} ms (加速: {orig_time/opt_time:.2f}x)")
    
    print(f"\n>>> 显存峰值")
    print(f"  基准显存:   {base_memory:.2f} GB")
    print(f"  原版峰值:   {memory_stats['original']:.2f} GB (增量: {memory_stats['original'] - base_memory:.2f} GB)")
    print(f"  优化版峰值: {memory_stats['optimized']:.2f} GB (增量: {memory_stats['optimized'] - base_memory:.2f} GB)")
    
    return results, memory_stats


# ============================================================================
# 主测试入口
# ============================================================================

if __name__ == "__main__":
    # 精度测试配置
    accuracy_configs = [
        TestConfig(name="小规模", batch_size=1, num_heads=4, chunk_size=32, seq_len=64, head_dim=32, topk=16),
        TestConfig(name="中等规模", batch_size=1, num_heads=8, chunk_size=128, seq_len=256, head_dim=64, topk=64),
        TestConfig(name="大规模", batch_size=1, num_heads=16, chunk_size=512, seq_len=1024, head_dim=128, topk=256),
        TestConfig(name="多batch", batch_size=4, num_heads=8, chunk_size=64, seq_len=128, head_dim=64, topk=32),
        TestConfig(name="大head_dim", batch_size=1, num_heads=8, chunk_size=128, seq_len=256, head_dim=256, topk=64),
    ]
    
    # ========== 前向精度测试 ==========
    test_fwd_accuracy(accuracy_configs)
    
    # ========== 反向精度测试 ==========
    test_bwd_accuracy(accuracy_configs)
    
    # ========== 性能对比测试 ==========
    test_performance_comparison(
        batch_size=1,
        num_heads=128,
        chunk_size=4 * 1024,
        seq_len=8 * 1024,
        head_dim=576,
        topk=2048,
        num_warmup=3,
        num_benchmark=10,
    )
    
    # ========== 前向+反向性能对比测试 ==========
    test_fwd_bwd_performance_comparison(
        batch_size=1,
        num_heads=128,
        chunk_size=4 * 1024,
        seq_len=8 * 1024,
        head_dim=576,
        topk=2048,
        num_warmup=3,
        num_benchmark=10,
    )

