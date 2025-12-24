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
# Fused Kernel: Sparse Attention + Loss (Online Softmax 优化版本)
# ============================================================================

@triton.jit
def _sparse_attn_loss_fused_kernel(
    Q_ptr, K_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    # 中间存储指针 (用于两遍扫描)
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
    stride_asb, stride_ass, stride_ask,  # AttnSum strides
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """
    Sparse Attention + Loss Kernel (Online Softmax 优化版本)
    
    使用 Online Softmax 算法，在整个 topk 上正确计算 softmax：
    
    对每个 head:
      Pass 1: 遍历所有 BLOCK_TOPK 块，使用 online 算法计算全局 max 和 sum
      Pass 2: 遍历所有 BLOCK_TOPK 块，使用全局 max/sum 计算归一化概率
    
    Online Softmax 核心公式:
      m_new = max(m_old, max(block))
      l_new = l_old * exp(m_old - m_new) + sum(exp(block - m_new))
      最终: p[i] = exp(x[i] - m_final) / l_final
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
    # Part 1: 对每个 head 使用 Online Softmax 计算 attention，累加到 attn_sum
    # =========================================================================
    
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        # -----------------------------------------------------------------
        # Pass 1: 计算全局 max 和 sum (Online Softmax)
        # -----------------------------------------------------------------
        m_global = NEG_INF  # running max
        l_global = 0.0      # running sum
        
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            # 加载 indices 并计算 causal mask
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)
            
            # 计算 QK
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
            # 修正旧的 sum，并加上新块的 exp
            l_global = l_global * tl.exp(m_global - m_new) + tl.sum(tl.exp(qk - m_new))
            m_global = m_new
        
        # 处理全 NEG_INF 情况
        m_global = tl.where(m_global == NEG_INF, 0.0, m_global)
        l_global = tl.where(l_global < 1e-9, 1.0, l_global)
        
        # -----------------------------------------------------------------
        # Pass 2: 使用全局 max/sum 计算归一化概率，累加到 attn_sum
        # -----------------------------------------------------------------
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
            
            # 使用全局 max/sum 归一化
            p = tl.exp(qk - m_global) / l_global
            p = tl.where(causal_mask_block, 0.0, p)
            
            # 累加到 attn_sum (第一个 head 写入，后续 head 累加)
            attn_sum_ptrs = attn_sum_base + offs_tk * stride_ask
            if h == 0:
                tl.store(attn_sum_ptrs, p, mask=tk_mask)
            else:
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
    
    # Triton Sparse版本 (传入 chunk_offset)
    tri = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    
    diff = abs(ref.item() - tri.item())
    passed = diff < 1e-3
    print(f"Accuracy - PyTorch(Full): {ref.item():.6f}, Triton(Sparse): {tri.item():.6f}, Diff: {diff:.6e}, Pass: {passed}")
    return passed


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
    
    # Test 1: Triton fused kernel (Sparse)
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    results['triton_sparse'] = triton_time
    
    # Test 2: PyTorch reference (Full) - 仅当 triton_only=False 时执行
    if not triton_only:
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
    if triton_only:
        print(f"  Triton Sparse fused:   {triton_time:.3f} ms")
    else:
        print(f"  PyTorch Full ref:      {pytorch_time:.3f} ms")
        print(f"  Triton Sparse fused:   {triton_time:.3f} ms (加速: {pytorch_time/triton_time:.2f}x)")
    
    return results



# ============================================================================
# 性能分析：拆分 Kernel 各部分
# ============================================================================

@triton.jit
def _profile_load_only_kernel(
    K_ptr, Indices_ptr, Output_ptr,
    batch_size, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
):
    """Profile: 只测试数据加载"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    offs_topk = tl.arange(0, topk)
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    
    acc = tl.zeros([topk], dtype=tl.float32)
    num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
    for d_idx in range(num_d_blocks):
        d_start = d_idx * BLOCK_D
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < head_dim
        
        k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(k_gathered, axis=1)
    
    tl.store(Output_ptr + pid, tl.sum(acc))


@triton.jit
def _profile_qk_only_kernel(
    Q_ptr, K_ptr, Indices_ptr, Output_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr, scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
):
    """Profile: 测试 QK 计算（不含 softmax）"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    
    offs_topk = tl.arange(0, topk)
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    
    acc = tl.zeros([topk], dtype=tl.float32)
    
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        qk = tl.zeros([topk], dtype=tl.float32)
        
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        acc += qk * scaling
    
    tl.store(Output_ptr + pid, tl.sum(acc))


@triton.jit
def _profile_qk_softmax_kernel(
    Q_ptr, K_ptr, Indices_ptr, Output_ptr,
    batch_size, num_heads, chunk_size, chunk_offset,
    head_dim: tl.constexpr, topk: tl.constexpr, scaling,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_ib, stride_is, stride_ik,
    BLOCK_D: tl.constexpr,
):
    """Profile: 测试 QK + Softmax（不含 KL）"""
    pid = tl.program_id(0)
    pid_batch = pid // chunk_size
    pid_row = pid % chunk_size
    NEG_INF = -1e9
    
    offs_topk = tl.arange(0, topk)
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    global_query_pos = chunk_offset + pid_row
    causal_mask = indices > global_query_pos
    
    attn_sum = tl.zeros([topk], dtype=tl.float32)
    
    for h in range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        qk = tl.zeros([topk], dtype=tl.float32)
        
        num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
        for d_idx in range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        qk = qk * scaling
        qk = tl.where(causal_mask, NEG_INF, qk)
        
        m = tl.max(qk)
        m = tl.where(m == NEG_INF, 0.0, m)
        p = tl.exp(qk - m)
        l = tl.sum(p)
        l = tl.where(l < 1e-9, 1.0, l)
        p = p / l
        p = tl.where(causal_mask, 0.0, p)
        
        attn_sum += p
    
    tl.store(Output_ptr + pid, tl.sum(attn_sum))


def profile_kernel_parts(
    batch_size=1, num_heads=16, chunk_size=4096, seq_len=8192,
    head_dim=512, topk=2048, num_warmup=3, num_iters=10
):
    """
    分步 Profile 各部分耗时
    
    测试内容:
    1. 只加载 K (gather 操作)
    2. QK 计算 (不含 softmax)
    3. QK + Softmax (不含 KL)
    4. 完整 kernel
    """
    import time
    
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    chunk_offset = seq_len - chunk_size
    
    print("=" * 70)
    print("Kernel 各部分耗时分析")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}")
    print(f"       seq={seq_len}, dim={head_dim}, topk={topk}")
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
    grid = (batch_size * chunk_size,)
    
    results = {}
    
    # Test 1: 只加载 K
    print("\n[1] 只加载 K (gather 操作)...")
    for _ in range(num_warmup):
        _profile_load_only_kernel[grid](
            key, indices, output,
            batch_size, chunk_size, chunk_offset, head_dim, topk,
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _profile_load_only_kernel[grid](
            key, indices, output,
            batch_size, chunk_size, chunk_offset, head_dim, topk,
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['load_k'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['load_k']:.3f} ms")
    
    # Test 2: QK 计算
    print("\n[2] QK 计算 (不含 softmax)...")
    for _ in range(num_warmup):
        _profile_qk_only_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
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
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['qk_only'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['qk_only']:.3f} ms")
    
    # Test 3: QK + Softmax
    print("\n[3] QK + Softmax (不含 KL)...")
    for _ in range(num_warmup):
        _profile_qk_softmax_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        _profile_qk_softmax_kernel[grid](
            query, key, indices, output,
            batch_size, num_heads, chunk_size, chunk_offset, head_dim, topk, scaling,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_D=BLOCK_D, num_warps=NUM_WARPS,
        )
    torch.cuda.synchronize()
    results['qk_softmax'] = (time.time() - start) / num_iters * 1000
    print(f"    耗时: {results['qk_softmax']:.3f} ms")
    
    # Test 4: 完整 kernel (Online Softmax 版本)
    print("\n[4] 完整 Kernel (Online Softmax)...")
    loss_per_row = torch.zeros(batch_size, chunk_size, device=device, dtype=torch.float32)
    attn_sum = torch.zeros(batch_size, chunk_size, topk, device=device, dtype=torch.float32)
    block_topk = min(BLOCK_TOPK, topk)
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
    print(f"  [1] 加载 K:                    {results['load_k']:.3f} ms ({results['load_k']/results['full']*100:.1f}%)")
    print(f"  [2] QK 计算 (含加载):          {results['qk_only']:.3f} ms ({results['qk_only']/results['full']*100:.1f}%)")
    print(f"  [3] QK + Softmax:              {results['qk_softmax']:.3f} ms ({results['qk_softmax']/results['full']*100:.1f}%)")
    print(f"  [4] 完整 Kernel:               {results['full']:.3f} ms (100%)")
    print("-" * 70)
    print(f"  增量: Softmax 开销:            {results['qk_softmax'] - results['qk_only']:.3f} ms")
    print(f"  增量: KL 开销:                 {results['full'] - results['qk_softmax']:.3f} ms")
    
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
