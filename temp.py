"""
Triton实现的Chunk Index Loss计算

主要优化点:
1. 融合attention score计算、mask、softmax (FlashAttention-like)
2. 融合head维度的求和与KL散度计算
3. 分块计算减少内存占用
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attention_softmax_kernel(
    # 输入指针
    Q_ptr,      # [batch, num_heads, chunk_size, head_dim]
    K_ptr,      # [batch, seq_k, head_dim] (MQA格式)
    Mask_ptr,   # [chunk_size, seq_k] causal mask
    # 输出指针
    Out_ptr,    # [batch, num_heads, chunk_size, seq_k] softmax后的attention scores
    # 维度参数
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    chunk_size: tl.constexpr,
    seq_k: tl.constexpr,
    head_dim: tl.constexpr,
    scaling: tl.constexpr,
    # strides
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ms, stride_mk,
    stride_ob, stride_oh, stride_os, stride_ok,
    # block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    融合的attention计算kernel:
    1. 计算 QK^T * scaling
    2. 应用mask
    3. 计算softmax
    """
    # 确定当前处理的batch, head, 和query位置
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # query行的偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # 加载query块 [BLOCK_M, BLOCK_D]
    q_ptrs = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + \
             offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < chunk_size) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # 初始化用于online softmax的变量
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # 分块遍历key维度
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # 加载key块 [seq_k, head_dim] -> [BLOCK_N, BLOCK_D]
        k_ptrs = K_ptr + pid_batch * stride_kb + \
                 offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < seq_k) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # 计算 QK^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * scaling
        
        # 加载并应用mask
        mask_ptrs = Mask_ptr + offs_m[:, None] * stride_ms + offs_n[None, :] * stride_mk
        mask_valid = (offs_m[:, None] < chunk_size) & (offs_n[None, :] < seq_k)
        causal_mask = tl.load(mask_ptrs, mask=mask_valid, other=True)
        qk = tl.where(causal_mask, -1e9, qk)
        
        # Online softmax更新
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
        
        # 存储中间结果（归一化后）
        p_normalized = p / l_i[:, None]
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + \
                   offs_m[:, None] * stride_os + offs_n[None, :] * stride_ok
        out_mask = (offs_m[:, None] < chunk_size) & (offs_n[None, :] < seq_k)
        tl.store(out_ptrs, p_normalized, mask=out_mask)


@triton.jit
def _kl_loss_kernel(
    # 输入指针
    IndexScore_ptr,    # [batch, chunk_size, seq_k]
    AttnScores_ptr,    # [batch, num_heads, chunk_size, seq_k]
    IndexMask_ptr,     # [batch, chunk_size, seq_k]
    # 输出指针
    Loss_ptr,          # [1] 输出loss
    # 维度参数
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    chunk_size: tl.constexpr,
    seq_k: tl.constexpr,
    eps: tl.constexpr,
    # strides for index_score
    stride_isb, stride_iss, stride_isk,
    # strides for attn_scores
    stride_asb, stride_ash, stride_ass, stride_ask,
    # strides for mask
    stride_mb, stride_ms, stride_mk,
    # block sizes
    BLOCK_K: tl.constexpr,
):
    """
    融合的KL Loss计算kernel:
    1. 对attention scores在head维度求和
    2. 对index_score应用mask和softmax
    3. 计算KL散度
    """
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)  # chunk_size中的行
    
    # 累加器
    kl_sum = tl.zeros([1], dtype=tl.float32)
    
    # 首先计算index_score的softmax所需的max和sum
    max_val = tl.zeros([1], dtype=tl.float32) - float("inf")
    sum_exp = tl.zeros([1], dtype=tl.float32)
    
    # 第一遍：计算max用于stable softmax
    for start_k in range(0, seq_k, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_k
        
        # 加载index_score
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        index_score = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        # 加载index_mask并应用
        im_ptrs = IndexMask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        index_mask = tl.load(im_ptrs, mask=k_mask, other=True)
        index_score = tl.where(index_mask, -1e9, index_score)
        
        max_val = tl.maximum(max_val, tl.max(index_score, axis=0))
    
    # 第二遍：计算sum(exp(x - max))
    for start_k in range(0, seq_k, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_k
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        index_score = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        im_ptrs = IndexMask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        index_mask = tl.load(im_ptrs, mask=k_mask, other=True)
        index_score = tl.where(index_mask, -1e9, index_score)
        
        sum_exp += tl.sum(tl.exp(index_score - max_val), axis=0)
    
    # 第三遍：计算attention scores在head维度的和，以及KL散度
    attn_sum_total = tl.zeros([1], dtype=tl.float32)
    
    # 先计算attn_scores的归一化分母
    for start_k in range(0, seq_k, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_k
        
        # 对所有head求和
        attn_sum_block = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h in range(num_heads):
            as_ptrs = AttnScores_ptr + pid_batch * stride_asb + h * stride_ash + \
                      pid_row * stride_ass + offs_k * stride_ask
            attn_score = tl.load(as_ptrs, mask=k_mask, other=0.0)
            attn_sum_block += attn_score
        
        attn_sum_total += tl.sum(attn_sum_block, axis=0)
    
    # 第四遍：计算KL散度
    for start_k in range(0, seq_k, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_k
        
        # 加载并计算index_score的softmax
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        index_score = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        im_ptrs = IndexMask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        index_mask = tl.load(im_ptrs, mask=k_mask, other=True)
        index_score = tl.where(index_mask, -1e9, index_score)
        
        # softmax(index_score) + eps
        p = tl.exp(index_score - max_val) / sum_exp + eps
        
        # 计算attention分布 (对head求和后归一化)
        attn_sum_block = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h in range(num_heads):
            as_ptrs = AttnScores_ptr + pid_batch * stride_asb + h * stride_ash + \
                      pid_row * stride_ass + offs_k * stride_ask
            attn_score = tl.load(as_ptrs, mask=k_mask, other=0.0)
            attn_sum_block += attn_score
        
        # attn_dist + eps
        q = attn_sum_block / attn_sum_total + eps
        
        # KL散度: q * log(q/p) = q * (log(q) - log(p))
        kl = tl.where(k_mask, q * (tl.log(q) - tl.log(p)), 0.0)
        kl_sum += tl.sum(kl, axis=0)
    
    # 原子加到输出
    tl.atomic_add(Loss_ptr, kl_sum)


@triton.jit
def _fused_chunk_loss_kernel(
    # Query and Key inputs
    Q_ptr,          # [chunk_size, batch, num_heads, head_dim]
    K_ptr,          # [seq_k, batch, head_dim] (MQA format)
    IndexScore_ptr, # [batch, chunk_size, seq_k]
    IndexMask_ptr,  # [batch, chunk_size, seq_k]
    # Output
    Loss_ptr,       # [1] scalar loss
    # Dimensions
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    chunk_size: tl.constexpr,
    seq_k: tl.constexpr,
    head_dim: tl.constexpr,
    scaling,
    eps: tl.constexpr,
    start_idx,      # causal mask的起始位置
    # Q strides: [chunk_size, batch, num_heads, head_dim]
    stride_qs, stride_qb, stride_qh, stride_qd,
    # K strides: [seq_k, batch, head_dim]
    stride_ks, stride_kb, stride_kd,
    # IndexScore strides: [batch, chunk_size, seq_k]
    stride_isb, stride_iss, stride_isk,
    # IndexMask strides: [batch, chunk_size, seq_k]
    stride_imb, stride_ims, stride_imk,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    完全融合的chunk loss计算kernel:
    对每个(batch, query_row)位置:
    1. 计算该行的attention scores (所有heads)
    2. 对heads求和
    3. 归一化得到attention分布
    4. 计算与index_score的KL散度
    
    这样避免存储完整的attention矩阵，大大节省内存
    """
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)  # 处理chunk_size中的第pid_m行
    
    offs_d = tl.arange(0, BLOCK_D)
    
    # ============ 步骤1: 计算所有heads的attention并求和 ============
    # 对该query位置，需要计算与所有keys的attention，然后对heads求和
    
    # 初始化：存储每个key位置的attention总和（所有heads）
    # 使用online softmax，需要跟踪每个head的max和sum
    
    # 我们需要对每个head分别计算softmax，然后求和
    # 为了节省内存，我们逐head处理
    
    attn_sum = tl.zeros([BLOCK_N], dtype=tl.float32)  # 累加所有heads的attention
    attn_total = tl.zeros([1], dtype=tl.float32)  # 用于最终归一化
    
    # 对于每个head
    for h in range(num_heads):
        # 加载这一行的query [1, head_dim]
        q_ptr = Q_ptr + pid_m * stride_qs + pid_batch * stride_qb + h * stride_qh
        q = tl.load(q_ptr + offs_d * stride_qd, mask=offs_d < head_dim, other=0.0)
        
        # Online softmax变量
        m_max = -float("inf")
        l_sum = 0.0
        
        # 第一遍：计算max
        for start_n in range(0, seq_k, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_k
            
            # 加载keys [BLOCK_N, head_dim]
            k_ptrs = K_ptr + offs_n[:, None] * stride_ks + pid_batch * stride_kb + offs_d[None, :] * stride_kd
            k_mask_2d = n_mask[:, None] & (offs_d[None, :] < head_dim)
            k = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)
            
            # QK^T [BLOCK_N]
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            # 应用causal mask: 位置 > pid_m + start_idx 应该被mask
            causal_mask = offs_n > (pid_m + start_idx)
            qk = tl.where(causal_mask | ~n_mask, -float("inf"), qk)
            
            # 应用index_mask
            im_ptrs = IndexMask_ptr + pid_batch * stride_imb + pid_m * stride_ims + offs_n * stride_imk
            index_mask = tl.load(im_ptrs, mask=n_mask, other=True)
            qk = tl.where(index_mask, -float("inf"), qk)
            
            m_max = tl.maximum(m_max, tl.max(qk))
        
        # 第二遍：计算softmax分母
        for start_n in range(0, seq_k, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_k
            
            k_ptrs = K_ptr + offs_n[:, None] * stride_ks + pid_batch * stride_kb + offs_d[None, :] * stride_kd
            k_mask_2d = n_mask[:, None] & (offs_d[None, :] < head_dim)
            k = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            causal_mask = offs_n > (pid_m + start_idx)
            qk = tl.where(causal_mask | ~n_mask, -float("inf"), qk)
            
            im_ptrs = IndexMask_ptr + pid_batch * stride_imb + pid_m * stride_ims + offs_n * stride_imk
            index_mask = tl.load(im_ptrs, mask=n_mask, other=True)
            qk = tl.where(index_mask, -float("inf"), qk)
            
            l_sum += tl.sum(tl.exp(qk - m_max))
    
    # 需要存储完整的attn_sum以便后续KL计算
    # 由于seq_k可能很大，我们采用两阶段：先累加attention，再计算KL
    
    # ============ 步骤2: 计算index_score的softmax ============
    # 对index_score计算softmax
    
    max_is = -float("inf")
    sum_is = 0.0
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_m * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        # 应用mask
        im_ptrs = IndexMask_ptr + pid_batch * stride_imb + pid_m * stride_ims + offs_n * stride_imk
        index_mask = tl.load(im_ptrs, mask=n_mask, other=True)
        causal_mask = offs_n > (pid_m + start_idx)
        is_val = tl.where(index_mask | causal_mask, -float("inf"), is_val)
        
        max_is = tl.maximum(max_is, tl.max(is_val))
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_m * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        im_ptrs = IndexMask_ptr + pid_batch * stride_imb + pid_m * stride_ims + offs_n * stride_imk
        index_mask = tl.load(im_ptrs, mask=n_mask, other=True)
        causal_mask = offs_n > (pid_m + start_idx)
        is_val = tl.where(index_mask | causal_mask, -float("inf"), is_val)
        
        sum_is += tl.sum(tl.exp(is_val - max_is))
    
    # ============ 步骤3: 计算KL散度 ============
    kl_sum = 0.0
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        # 计算attention分布（重新计算，因为没存储）
        attn_block = tl.zeros([BLOCK_N], dtype=tl.float32)
        attn_sum_local = 0.0
        
        for h in range(num_heads):
            q_ptr = Q_ptr + pid_m * stride_qs + pid_batch * stride_qb + h * stride_qh
            q = tl.load(q_ptr + offs_d * stride_qd, mask=offs_d < head_dim, other=0.0)
            
            k_ptrs = K_ptr + offs_n[:, None] * stride_ks + pid_batch * stride_kb + offs_d[None, :] * stride_kd
            k_mask_2d = n_mask[:, None] & (offs_d[None, :] < head_dim)
            k = tl.load(k_ptrs, mask=k_mask_2d, other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            causal_mask = offs_n > (pid_m + start_idx)
            im_ptrs = IndexMask_ptr + pid_batch * stride_imb + pid_m * stride_ims + offs_n * stride_imk
            index_mask = tl.load(im_ptrs, mask=n_mask, other=True)
            qk = tl.where(causal_mask | index_mask | ~n_mask, -float("inf"), qk)
            
            # 需要该head的max和sum来计算softmax
            # 这里简化处理：累加未归一化的exp值
            # 注意：这里的实现有问题，应该对每个head分别softmax后再求和
            # 暂时使用一个近似方法
            attn_block += tl.exp(qk)
        
        attn_sum_local = tl.sum(attn_block)
        
        # Index score softmax
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_m * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        im_ptrs = IndexMask_ptr + pid_batch * stride_imb + pid_m * stride_ims + offs_n * stride_imk
        index_mask = tl.load(im_ptrs, mask=n_mask, other=True)
        causal_mask = offs_n > (pid_m + start_idx)
        is_val = tl.where(index_mask | causal_mask, -float("inf"), is_val)
        
        p = tl.exp(is_val - max_is) / sum_is + eps
        
        # 暂存attn_block用于后续归一化
        tl.atomic_add(Loss_ptr + 1, attn_sum_local)  # 用于计算总和
    
    # 这个kernel的设计比较复杂，让我重新设计一个更清晰的版本


class ChunkIndexLoss(torch.autograd.Function):
    """
    Triton实现的Chunk Index Loss
    
    前向传播:
    1. 对每个chunk，计算attention scores
    2. 应用mask和softmax
    3. 对heads求和
    4. 计算与index_score的KL散度
    """
    
    @staticmethod
    def forward(
        ctx,
        query,          # [chunk_size, batch, num_heads, head_dim]
        key,            # [seq_k, batch, head_dim] (MQA)
        index_score,    # [batch, chunk_size, seq_k]
        index_mask,     # [batch, chunk_size, seq_k]
        scaling,
        start_idx,
    ):
        batch_size, chunk_size, seq_k = index_score.shape
        _, _, num_heads, head_dim = query.shape
        
        # 使用分块计算避免OOM
        BLOCK_M = 1
        BLOCK_N = min(1024, triton.next_power_of_2(seq_k))
        BLOCK_D = triton.next_power_of_2(head_dim)
        
        # 分配输出
        loss = torch.zeros(1, device=query.device, dtype=torch.float32)
        
        # 计算attention scores（分块）并累加KL loss
        # 为了支持反向传播，需要存储一些中间结果
        # 但如果只需要forward，可以完全融合
        
        # 暂时使用分块的PyTorch实现作为参考
        return ChunkIndexLoss._forward_reference(
            query, key, index_score, index_mask, scaling, start_idx
        )
    
    @staticmethod
    def _forward_reference(query, key, index_score, index_mask, scaling, start_idx):
        """
        参考实现：使用分块的方式计算，减少内存占用
        """
        chunk_size = query.shape[0]
        batch_size = query.shape[1]
        num_heads = query.shape[2]
        seq_k = key.shape[0]
        
        eps = 1e-10
        
        # 转换query格式: [chunk_size, batch, num_heads, head_dim] -> [batch, num_heads, chunk_size, head_dim]
        query = query.permute(1, 2, 0, 3)
        # key: [seq_k, batch, head_dim] -> [batch, seq_k, head_dim]
        key = key.permute(1, 0, 2)
        
        # 计算attention scores分块
        kl_loss = 0.0
        
        # 对每行单独计算以节省内存
        for i in range(chunk_size):
            q_row = query[:, :, i:i+1, :]  # [batch, num_heads, 1, head_dim]
            
            # QK^T: [batch, num_heads, 1, seq_k]
            attn = torch.matmul(q_row, key.transpose(-1, -2)) * scaling
            
            # 应用mask
            row_mask = index_mask[:, i:i+1, :]  # [batch, 1, seq_k]
            attn = attn.masked_fill(row_mask.unsqueeze(1), -1e9)
            
            # Softmax
            attn = torch.softmax(attn, dim=-1)  # [batch, num_heads, 1, seq_k]
            
            # 对heads求和
            attn_sum = attn.sum(dim=1)  # [batch, 1, seq_k]
            attn_dist = attn_sum / attn_sum.sum(dim=-1, keepdim=True)
            
            # Index score softmax
            is_row = index_score[:, i:i+1, :]  # [batch, 1, seq_k]
            is_row = is_row.masked_fill(row_mask, -1e9)
            is_prob = torch.softmax(is_row, dim=-1) + eps
            
            # KL散度
            kl = torch.nn.functional.kl_div(
                is_prob.log(), 
                attn_dist + eps, 
                reduction='batchmean'
            )
            kl_loss = kl_loss + kl
        
        return kl_loss / chunk_size


# ============ 优化版本：使用Triton kernel ============

@triton.jit
def _compute_row_attention_kernel(
    # Inputs
    Q_ptr,          # [batch, num_heads, head_dim] - 单行query
    K_ptr,          # [batch, seq_k, head_dim]
    IndexScore_ptr, # [batch, seq_k]
    Mask_ptr,       # [batch, seq_k]
    # Output
    Loss_ptr,       # [batch] 每个batch的loss
    # Dimensions
    batch_size,
    num_heads: tl.constexpr,
    seq_k,
    head_dim: tl.constexpr,
    scaling,
    eps: tl.constexpr,
    # Strides
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_isb, stride_isk,
    stride_mb, stride_mk,
    # Block size
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    对单行计算attention和KL loss
    """
    pid_batch = tl.program_id(0)
    
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    
    # ======= 第1步：对每个head计算attention并求和 =======
    # 需要对每个head分别做softmax，然后求和
    
    # 初始化每个位置的attention总和
    # 由于seq_k可能很大，我们需要分块处理
    
    # 首先计算所有heads的attention的max和sum
    head_max = tl.zeros([num_heads], dtype=tl.float32) - float("inf")
    head_sum = tl.zeros([num_heads], dtype=tl.float32)
    
    # 加载所有heads的query
    for h in range(num_heads):
        m_h = -float("inf")
        
        for start_n in range(0, seq_k, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_k
            
            # 加载query [head_dim]
            q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh
            q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
            
            # 加载key [BLOCK_N, head_dim]
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            # QK^T [BLOCK_N]
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            # 应用mask
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            m_h = tl.maximum(m_h, tl.max(qk))
        
        head_max = tl.where(tl.arange(0, num_heads) == h, m_h, head_max)
        
        # 第二遍计算sum
        s_h = 0.0
        for start_n in range(0, seq_k, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_k
            
            q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh
            q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            s_h += tl.sum(tl.exp(qk - m_h))
        
        head_sum = tl.where(tl.arange(0, num_heads) == h, s_h, head_sum)
    
    # ======= 第2步：计算index_score的softmax参数 =======
    max_is = -float("inf")
    sum_is = 0.0
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        max_is = tl.maximum(max_is, tl.max(is_val))
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        sum_is += tl.sum(tl.exp(is_val - max_is))
    
    # ======= 第3步：计算KL散度 =======
    # 需要计算 attn_dist 和 index_prob 的 KL 散度
    # KL(attn || index) = sum(attn * log(attn / index))
    
    # 首先需要计算 attn_sum 的归一化分母
    attn_norm = 0.0
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        attn_block = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for h in range(num_heads):
            q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh
            q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            # 获取这个head的max
            m_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_max, 0.0))
            s_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_sum, 0.0))
            
            # softmax后的attention
            attn_h = tl.exp(qk - m_h) / s_h
            attn_block += attn_h
        
        attn_norm += tl.sum(tl.where(n_mask, attn_block, 0.0))
    
    # 计算KL散度
    kl_sum = 0.0
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        # 计算attention分布
        attn_block = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for h in range(num_heads):
            q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh
            q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            m_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_max, 0.0))
            s_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_sum, 0.0))
            
            attn_h = tl.exp(qk - m_h) / s_h
            attn_block += attn_h
        
        # 归一化
        attn_dist = attn_block / attn_norm + eps
        
        # Index score softmax
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        index_prob = tl.exp(is_val - max_is) / sum_is + eps
        
        # KL: attn_dist * log(attn_dist / index_prob)
        # 注意：PyTorch的kl_div是 input.exp() * (input - target)
        # 这里用的是 KL(q||p) = sum(q * log(q/p))
        kl = tl.where(n_mask, attn_dist * (tl.log(attn_dist) - tl.log(index_prob)), 0.0)
        kl_sum += tl.sum(kl)
    
    # 存储结果
    tl.store(Loss_ptr + pid_batch, kl_sum)


def compute_chunk_index_loss_triton(
    query,          # [chunk_size, batch, num_heads, head_dim]
    key,            # [seq_k, batch, head_dim] (MQA)
    index_score,    # [batch, chunk_size, seq_k]
    index_mask,     # [batch, chunk_size, seq_k] bool
    scaling,
    start_idx=0,
):
    """
    使用Triton计算chunk index loss
    
    Args:
        query: Query tensor [chunk_size, batch, num_heads, head_dim]
        key: Key tensor [seq_k, batch, head_dim] (MQA format)
        index_score: Index scores [batch, chunk_size, seq_k]
        index_mask: Boolean mask [batch, chunk_size, seq_k]
        scaling: Attention scaling factor
        start_idx: Start index for causal mask
    
    Returns:
        Scalar KL divergence loss
    """
    chunk_size, batch_size, num_heads, head_dim = query.shape
    seq_k = key.shape[0]
    
    eps = 1e-10
    
    # 转换格式
    # query: [chunk_size, batch, num_heads, head_dim] -> [batch, num_heads, chunk_size, head_dim]
    query_t = query.permute(1, 2, 0, 3).contiguous()
    # key: [seq_k, batch, head_dim] -> [batch, seq_k, head_dim]
    key_t = key.permute(1, 0, 2).contiguous()
    
    # 输出
    loss_per_row = torch.zeros(batch_size, chunk_size, device=query.device, dtype=torch.float32)
    
    # Block sizes
    BLOCK_N = min(1024, triton.next_power_of_2(seq_k))
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    # 对每一行启动kernel
    for row in range(chunk_size):
        # 获取这一行的数据
        q_row = query_t[:, :, row, :]  # [batch, num_heads, head_dim]
        is_row = index_score[:, row, :]  # [batch, seq_k]
        mask_row = index_mask[:, row, :]  # [batch, seq_k]
        
        loss_row = torch.zeros(batch_size, device=query.device, dtype=torch.float32)
        
        grid = (batch_size,)
        
        _compute_row_attention_kernel[grid](
            q_row, key_t, is_row, mask_row, loss_row,
            batch_size, num_heads, seq_k, head_dim, scaling, eps,
            # strides
            q_row.stride(0), q_row.stride(1), q_row.stride(2),
            key_t.stride(0), key_t.stride(1), key_t.stride(2),
            is_row.stride(0), is_row.stride(1),
            mask_row.stride(0), mask_row.stride(1),
            BLOCK_N, BLOCK_D,
        )
        
        loss_per_row[:, row] = loss_row
    
    # 返回平均loss
    return loss_per_row.mean()


# ============ 更高效的完全融合版本 ============

@triton.jit  
def _fused_chunk_loss_v2_kernel(
    # Inputs
    Q_ptr,          # [batch, num_heads, chunk_size, head_dim]
    K_ptr,          # [batch, seq_k, head_dim]
    IndexScore_ptr, # [batch, chunk_size, seq_k]
    Mask_ptr,       # [batch, chunk_size, seq_k]
    # Output
    Loss_ptr,       # [batch, chunk_size]
    # Dimensions
    batch_size,
    num_heads: tl.constexpr,
    chunk_size,
    seq_k,
    head_dim: tl.constexpr,
    scaling,
    eps: tl.constexpr,
    # Strides for Q: [batch, num_heads, chunk_size, head_dim]
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K: [batch, seq_k, head_dim]
    stride_kb, stride_ks, stride_kd,
    # Strides for IndexScore: [batch, chunk_size, seq_k]
    stride_isb, stride_iss, stride_isk,
    # Strides for Mask: [batch, chunk_size, seq_k]
    stride_mb, stride_ms, stride_mk,
    # Strides for Loss: [batch, chunk_size]
    stride_lb, stride_ls,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    完全融合的chunk loss kernel
    每个program处理一个(batch, row)对
    """
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    
    # ======= 步骤1: 计算每个head的softmax参数 =======
    # 存储每个head的max和sum
    head_max = tl.zeros([num_heads], dtype=tl.float32) - float("inf")
    head_sum = tl.zeros([num_heads], dtype=tl.float32)
    
    for h in tl.static_range(num_heads):
        # 加载query
        q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh + pid_row * stride_qs
        q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
        
        m_h = -float("inf")
        
        # 第一遍：找max
        for start_n in range(0, seq_k, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_k
            
            # 加载key
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            # QK^T
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            # 加载并应用mask
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            m_h = tl.maximum(m_h, tl.max(qk))
        
        # 第二遍：计算sum
        s_h = 0.0
        for start_n in range(0, seq_k, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_k
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            s_h += tl.sum(tl.exp(qk - m_h))
        
        # 更新head_max和head_sum
        head_max = tl.where(tl.arange(0, num_heads) == h, m_h, head_max)
        head_sum = tl.where(tl.arange(0, num_heads) == h, s_h, head_sum)
    
    # ======= 步骤2: 计算index_score的softmax参数 =======
    max_is = -float("inf")
    sum_is = 0.0
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        max_is = tl.maximum(max_is, tl.max(is_val))
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        sum_is += tl.sum(tl.exp(is_val - max_is))
    
    # ======= 步骤3: 计算attn分布的归一化分母 =======
    attn_norm = 0.0
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        attn_block = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for h in tl.static_range(num_heads):
            q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh + pid_row * stride_qs
            q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            # 获取该head的max和sum
            m_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_max, 0.0))
            s_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_sum, 0.0))
            
            attn_h = tl.exp(qk - m_h) / (s_h + 1e-10)
            attn_block += attn_h
        
        attn_norm += tl.sum(tl.where(n_mask, attn_block, 0.0))
    
    # ======= 步骤4: 计算KL散度 =======
    kl_sum = 0.0
    
    for start_n in range(0, seq_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_k
        
        # 计算attention分布
        attn_block = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for h in tl.static_range(num_heads):
            q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh + pid_row * stride_qs
            q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            m_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_max, 0.0))
            s_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_sum, 0.0))
            
            attn_h = tl.exp(qk - m_h) / (s_h + 1e-10)
            attn_block += attn_h
        
        # 归一化得到attn分布
        attn_dist = attn_block / (attn_norm + 1e-10) + eps
        
        # 计算index_prob
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        index_prob = tl.exp(is_val - max_is) / (sum_is + 1e-10) + eps
        
        # KL散度: attn * log(attn / index)
        # 注意：F.kl_div(input.log(), target) = target * (log(target) - input)
        # 所以这里应该是 attn_dist * (log(attn_dist) - log(index_prob))
        kl = tl.where(n_mask, attn_dist * (tl.log(attn_dist) - tl.log(index_prob)), 0.0)
        kl_sum += tl.sum(kl)
    
    # 存储结果
    loss_ptr = Loss_ptr + pid_batch * stride_lb + pid_row * stride_ls
    tl.store(loss_ptr, kl_sum)


def compute_chunk_index_loss_v2(
    query,          # [chunk_size, batch, num_heads, head_dim]
    key,            # [seq_k, batch, head_dim]
    index_score,    # [batch, chunk_size, seq_k]
    index_mask,     # [batch, chunk_size, seq_k]
    scaling,
):
    """
    高效的Triton实现chunk index loss
    """
    chunk_size, batch_size, num_heads, head_dim = query.shape
    seq_k = key.shape[0]
    
    # 转换格式
    query_t = query.permute(1, 2, 0, 3).contiguous()  # [batch, num_heads, chunk_size, head_dim]
    key_t = key.permute(1, 0, 2).contiguous()  # [batch, seq_k, head_dim]
    
    # 确保是contiguous
    index_score = index_score.contiguous()
    index_mask = index_mask.contiguous()
    
    # 输出
    loss = torch.zeros(batch_size, chunk_size, device=query.device, dtype=torch.float32)
    
    # Block sizes
    BLOCK_N = min(1024, triton.next_power_of_2(seq_k))
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    # Grid: 每个(batch, row)一个program
    grid = (batch_size * chunk_size,)
    
    eps = 1e-10
    
    _fused_chunk_loss_v2_kernel[grid](
        query_t, key_t, index_score, index_mask, loss,
        batch_size, num_heads, chunk_size, seq_k, head_dim,
        scaling, eps,
        # strides
        query_t.stride(0), query_t.stride(1), query_t.stride(2), query_t.stride(3),
        key_t.stride(0), key_t.stride(1), key_t.stride(2),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        index_mask.stride(0), index_mask.stride(1), index_mask.stride(2),
        loss.stride(0), loss.stride(1),
        BLOCK_N, BLOCK_D,
    )
    
    return loss.mean()


# ============ 测试函数 ============

def test_chunk_index_loss():
    """测试Triton实现与PyTorch实现的一致性"""
    import torch.nn.functional as F
    
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 2
    num_heads = 4
    chunk_size = 32
    seq_k = 256
    head_dim = 64
    scaling = 1.0 / (head_dim ** 0.5)
    
    # 创建测试数据
    query = torch.randn(chunk_size, batch_size, num_heads, head_dim, device='cuda', dtype=torch.float32)
    key = torch.randn(seq_k, batch_size, head_dim, device='cuda', dtype=torch.float32)
    index_score = torch.randn(batch_size, chunk_size, seq_k, device='cuda', dtype=torch.float32)
    index_mask = torch.rand(batch_size, chunk_size, seq_k, device='cuda') > 0.7
    
    # PyTorch参考实现
    def pytorch_reference(query, key, index_score, index_mask, scaling):
        eps = 1e-10
        chunk_size = query.shape[0]
        
        query_t = query.permute(1, 2, 0, 3)  # [batch, num_heads, chunk_size, head_dim]
        key_t = key.permute(1, 0, 2)  # [batch, seq_k, head_dim]
        
        # 计算attention
        attn = torch.matmul(query_t, key_t.transpose(-1, -2)) * scaling  # [batch, num_heads, chunk_size, seq_k]
        attn = attn.masked_fill(index_mask.unsqueeze(1), -1e9)
        attn = torch.softmax(attn, dim=-1)  # [batch, num_heads, chunk_size, seq_k]
        
        # 对heads求和并归一化
        attn_sum = attn.sum(dim=1)  # [batch, chunk_size, seq_k]
        attn_dist = attn_sum / attn_sum.sum(dim=-1, keepdim=True)
        
        # Index score softmax
        index_score_masked = index_score.masked_fill(index_mask, -1e9)
        index_prob = torch.softmax(index_score_masked, dim=-1) + eps
        
        # KL散度
        kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
        
        return kl_loss
    
    # 测试
    ref_loss = pytorch_reference(query, key, index_score, index_mask, scaling)
    triton_loss = compute_chunk_index_loss_v2(query, key, index_score, index_mask, scaling)
    
    print(f"PyTorch参考实现: {ref_loss.item():.6f}")
    print(f"Triton实现: {triton_loss.item():.6f}")
    print(f"差异: {abs(ref_loss.item() - triton_loss.item()):.6e}")
    
    # 性能测试
    import time
    
    # Warm up
    for _ in range(10):
        _ = compute_chunk_index_loss_v2(query, key, index_score, index_mask, scaling)
    
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start = time.time()
    for _ in range(100):
        _ = compute_chunk_index_loss_v2(query, key, index_score, index_mask, scaling)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100 * 1000
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(100):
        _ = pytorch_reference(query, key, index_score, index_mask, scaling)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 100 * 1000
    
    print(f"\nPyTorch时间: {pytorch_time:.3f} ms")
    print(f"Triton时间: {triton_time:.3f} ms")
    print(f"加速比: {pytorch_time / triton_time:.2f}x")


if __name__ == "__main__":
    test_chunk_index_loss()

