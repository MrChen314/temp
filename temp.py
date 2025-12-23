"""
Triton Fused Optimized - Sparse Attention Loss (H20 GPU优化版本)

针对H20 (Hopper架构, sm_90) 的优化:
1. 双重分块: BLOCK_TOPK + BLOCK_D，避免寄存器溢出
2. 两遍扫描 Online Softmax
3. 单个kernel完成 attention softmax + head sum + KL loss 计算
4. 无需中间tensor，减少显存使用
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# 固定配置常量
BLOCK_TOPK = 64   # topk 分块大小
BLOCK_D = 64      # head_dim 分块大小
NUM_STAGES = 2
NUM_WARPS = 4


# ============================================================================
# Fused Kernel: Sparse Attention + Loss (分块迭代版本)
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
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    分块迭代版本的 Sparse Attention + Loss Kernel
    
    关键优化：
    - 对 topk 维度分块 (BLOCK_TOPK)，避免 k_gathered[topk, BLOCK_D] 过大
    - 对 head_dim 维度分块 (BLOCK_D)
    - 使用两遍扫描计算 softmax：Pass1 计算 max，Pass2 计算 sum 和归一化
    
    内存使用分析 (配置: topk=512, head_dim=256, BLOCK_TOPK=64, BLOCK_D=64):
    - k_block: [64, 64] = 4K elements = 16KB ✓
    - 每线程约 64 寄存器，远低于 255 限制
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    NEG_INF = -1e9
    LOG2E: tl.constexpr = 1.44269504  # log2(e), 用于 exp2 优化
    
    # 基地址计算
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    
    # 预加载完整的 indices 和 causal_mask
    # 这是必须的，因为 softmax 需要完整的信息
    offs_topk = tl.arange(0, topk)
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    causal_mask = indices > pid_row
    
    # 最终累加结果
    attn_sum = tl.zeros([topk], dtype=tl.float32)
    
    # 预计算分块数量 (必须在循环外定义 constexpr)
    num_topk_blocks: tl.constexpr = (topk + BLOCK_TOPK - 1) // BLOCK_TOPK
    num_d_blocks: tl.constexpr = (head_dim + BLOCK_D - 1) // BLOCK_D
    
    # =========================================================================
    # Part 1: 累加所有heads的attention scores (分块版本)
    # =========================================================================
    for h in tl.static_range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        # =====================================================================
        # Pass 1: 分块计算 QK^T 并找全局 max
        # =====================================================================
        m_global = NEG_INF
        
        for topk_blk in tl.static_range(num_topk_blocks):
            topk_start = topk_blk * BLOCK_TOPK
            offs_topk_block = topk_start + tl.arange(0, BLOCK_TOPK)
            topk_mask = offs_topk_block < topk
            
            # 获取这个块的 indices
            indices_block = tl.load(
                idx_base + offs_topk_block * stride_ik, 
                mask=topk_mask, 
                other=0
            ).to(tl.int64)
            
            # 计算这个块的 qk
            qk_block = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            
            for d_blk in tl.static_range(num_d_blocks):
                d_start = d_blk * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                # 加载 Q chunk: [BLOCK_D]
                q = tl.load(
                    q_base + offs_d * stride_qd, 
                    mask=d_mask, 
                    other=0.0
                ).to(tl.float32)
                
                # 加载 K block: [BLOCK_TOPK, BLOCK_D] - 这是关键优化！
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_block = tl.load(
                    k_ptrs, 
                    mask=topk_mask[:, None] & d_mask[None, :], 
                    other=0.0
                ).to(tl.float32)
                
                # 向量化点积累加
                qk_block += tl.sum(q[None, :] * k_block, axis=1)
            
            # 应用 scaling 和 causal mask
            qk_block = qk_block * scaling
            causal_mask_block = indices_block > pid_row
            qk_block = tl.where(causal_mask_block, NEG_INF, qk_block)
            qk_block = tl.where(topk_mask, qk_block, NEG_INF)
            
            # 更新全局 max
            m_global = tl.maximum(m_global, tl.max(qk_block))
        
        # 处理全部被 mask 的情况
        m_global = tl.where(m_global == NEG_INF, 0.0, m_global)
        
        # =====================================================================
        # Pass 2: 计算 exp sum
        # =====================================================================
        l_global = 0.0
        
        for topk_blk in tl.static_range(num_topk_blocks):
            topk_start = topk_blk * BLOCK_TOPK
            offs_topk_block = topk_start + tl.arange(0, BLOCK_TOPK)
            topk_mask = offs_topk_block < topk
            
            indices_block = tl.load(
                idx_base + offs_topk_block * stride_ik, 
                mask=topk_mask, 
                other=0
            ).to(tl.int64)
            
            # 重新计算 qk_block
            qk_block = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            
            for d_blk in tl.static_range(num_d_blocks):
                d_start = d_blk * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                q = tl.load(
                    q_base + offs_d * stride_qd, 
                    mask=d_mask, 
                    other=0.0
                ).to(tl.float32)
                
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_block = tl.load(
                    k_ptrs, 
                    mask=topk_mask[:, None] & d_mask[None, :], 
                    other=0.0
                ).to(tl.float32)
                
                qk_block += tl.sum(q[None, :] * k_block, axis=1)
            
            qk_block = qk_block * scaling
            causal_mask_block = indices_block > pid_row
            qk_block = tl.where(causal_mask_block, NEG_INF, qk_block)
            qk_block = tl.where(topk_mask, qk_block, NEG_INF)
            
            # 计算 exp 并累加
            exp_block = tl.math.exp2((qk_block - m_global) * LOG2E)
            exp_block = tl.where(causal_mask_block | ~topk_mask, 0.0, exp_block)
            l_global += tl.sum(exp_block)
        
        l_global = tl.where(l_global < 1e-9, 1.0, l_global)
        
        # =====================================================================
        # Pass 3: 计算 softmax 并累加到 attn_sum
        # =====================================================================
        for topk_blk in tl.static_range(num_topk_blocks):
            topk_start = topk_blk * BLOCK_TOPK
            offs_topk_block = topk_start + tl.arange(0, BLOCK_TOPK)
            topk_mask = offs_topk_block < topk
            
            indices_block = tl.load(
                idx_base + offs_topk_block * stride_ik, 
                mask=topk_mask, 
                other=0
            ).to(tl.int64)
            
            # 重新计算 qk_block
            qk_block = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
            
            for d_blk in tl.static_range(num_d_blocks):
                d_start = d_blk * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim
                
                q = tl.load(
                    q_base + offs_d * stride_qd, 
                    mask=d_mask, 
                    other=0.0
                ).to(tl.float32)
                
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_block = tl.load(
                    k_ptrs, 
                    mask=topk_mask[:, None] & d_mask[None, :], 
                    other=0.0
                ).to(tl.float32)
                
                qk_block += tl.sum(q[None, :] * k_block, axis=1)
            
            qk_block = qk_block * scaling
            causal_mask_block = indices_block > pid_row
            qk_block = tl.where(causal_mask_block, NEG_INF, qk_block)
            
            # 计算归一化的 softmax
            p_block = tl.math.exp2((qk_block - m_global) * LOG2E) / l_global
            p_block = tl.where(causal_mask_block, 0.0, p_block)
            p_block = tl.where(topk_mask, p_block, 0.0)
            
            # 累加到 attn_sum 的对应位置
            # 使用完整的 offs_topk 数组和 mask 来更新
            update_mask = (offs_topk >= topk_start) & (offs_topk < topk_start + BLOCK_TOPK)
            # 由于 static_range，我们知道确切的位置
            attn_sum = tl.where(
                update_mask,
                attn_sum + tl.broadcast_to(p_block, [topk]),
                attn_sum
            )
    
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
    p_is = tl.math.exp2((is_val - m_is) * LOG2E)
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
# 简化版本：使用共享内存缓存 qk (减少重复计算)
# ============================================================================

@triton.jit
def _sparse_attn_loss_fused_kernel_v2(
    Q_ptr, K_ptr, IndexScore_ptr, Indices_ptr, Loss_ptr,
    batch_size, num_heads: tl.constexpr, seq_len,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    scaling,
    eps: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_isb, stride_iss, stride_isk,
    stride_ib, stride_is, stride_ik,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    V2版本：减少 BLOCK_D 大小，一次性处理 topk
    通过减小 BLOCK_D 来降低 k_gathered 的大小
    
    k_gathered: [topk, BLOCK_D] = [512, 32] = 16K elements ✓
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    NEG_INF = -1e9
    LOG2E: tl.constexpr = 1.44269504
    
    q_batch_base = Q_ptr + pid_batch * stride_qb
    k_batch_base = K_ptr + pid_batch * stride_kb
    idx_base = Indices_ptr + pid_batch * stride_ib + pid_row * stride_is
    is_base = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss
    
    offs_topk = tl.arange(0, topk)
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    causal_mask = indices > pid_row
    
    attn_sum = tl.zeros([topk], dtype=tl.float32)
    num_d_blocks: tl.constexpr = (head_dim + BLOCK_D - 1) // BLOCK_D
    
    for h in tl.static_range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs
        
        # 计算 QK^T - 小块 BLOCK_D
        qk = tl.zeros([topk], dtype=tl.float32)
        
        for d_idx in tl.static_range(num_d_blocks):
            d_start = d_idx * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim
            
            q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)
            
            # k_gathered: [topk, BLOCK_D] - BLOCK_D 较小时可以接受
            k_ptrs = k_batch_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_gathered = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
            
            qk += tl.sum(q[None, :] * k_gathered, axis=1)
        
        qk = qk * scaling
        qk = tl.where(causal_mask, NEG_INF, qk)
        
        # Softmax
        m = tl.max(qk)
        m = tl.where(m == NEG_INF, 0.0, m)
        p = tl.math.exp2((qk - m) * LOG2E)
        l = tl.sum(p)
        l = tl.where(l < 1e-9, 1.0, l)
        p = p / l
        p = tl.where(causal_mask, 0.0, p)
        
        attn_sum += p
    
    # 归一化
    attn_total = tl.sum(attn_sum)
    attn_total = tl.where(attn_total < eps, 1.0, attn_total)
    attn_dist = attn_sum / attn_total + eps
    
    # Index score softmax
    is_val = tl.load(is_base + offs_topk * stride_isk)
    is_val = tl.where(causal_mask, NEG_INF, is_val)
    
    m_is = tl.max(is_val)
    m_is = tl.where(m_is == NEG_INF, 0.0, m_is)
    p_is = tl.math.exp2((is_val - m_is) * LOG2E)
    s_is = tl.sum(p_is)
    s_is = tl.where(s_is < 1e-9, 1.0, s_is)
    index_prob = p_is / s_is + eps
    
    # KL 散度
    kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
    kl = tl.where(causal_mask, 0.0, kl)
    kl_sum = tl.sum(kl)
    
    tl.store(Loss_ptr + pid_batch * seq_len + pid_row, kl_sum)


# ============================================================================
# Wrapper函数
# ============================================================================

def compute_index_loss_sparse(query, key, index_score, indices, scaling, eps=1e-10, use_v2=True):
    """
    Sparse版本的完整loss计算 (分块迭代版本)
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, seq_len, head_dim]
        index_score: [batch, seq_len, topk]
        indices: [batch, seq_len, topk]
        scaling: attention scaling factor
        eps: 数值稳定epsilon
        use_v2: 使用V2简化版本 (推荐，更快)
    
    Returns:
        loss: 标量loss值
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    topk = indices.shape[-1]
    
    query = query.contiguous()
    key = key.contiguous()
    index_score = index_score.contiguous()
    indices = indices.contiguous().to(torch.int64)
    
    loss_per_row = torch.zeros(batch_size, seq_len, device=query.device, dtype=torch.float32)
    
    grid = (batch_size * seq_len,)
    
    if use_v2:
        # V2: 减小 BLOCK_D，一次处理完整 topk
        # 适用于 topk 不太大的情况
        block_d = 32  # 小的 BLOCK_D
        _sparse_attn_loss_fused_kernel_v2[grid](
            query, key, index_score, indices, loss_per_row,
            batch_size, num_heads, seq_len, head_dim, topk, scaling, eps,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_TOPK=BLOCK_TOPK,
            BLOCK_D=block_d,
            num_stages=NUM_STAGES, num_warps=NUM_WARPS,
        )
    else:
        # V1: 双重分块，适用于大 topk
        _sparse_attn_loss_fused_kernel[grid](
            query, key, index_score, indices, loss_per_row,
            batch_size, num_heads, seq_len, head_dim, topk, scaling, eps,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2),
            index_score.stride(0), index_score.stride(1), index_score.stride(2),
            indices.stride(0), indices.stride(1), indices.stride(2),
            BLOCK_TOPK=BLOCK_TOPK,
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
    
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    key_expanded = key.unsqueeze(2).expand(-1, -1, topk, -1)
    k_gathered = torch.gather(key_expanded, dim=1, index=indices_expanded)
    
    attn = torch.einsum('bhsd,bstd->bhst', query, k_gathered) * scaling
    
    row_indices = torch.arange(seq_len, device=query.device).view(1, 1, -1, 1)
    causal_mask = indices.unsqueeze(1) > row_indices
    attn = attn.masked_fill(causal_mask, -1e9)
    
    attn = torch.softmax(attn, dim=-1)
    
    attn_sum = attn.sum(dim=1)
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    causal_mask_2d = indices > torch.arange(seq_len, device=query.device).view(1, -1, 1)
    index_score_masked = index_score.masked_fill(causal_mask_2d, -1e9)
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

def test_full_accuracy(batch_size=1, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42, use_v2=True):
    """测试完整流程精度"""
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    ref = pytorch_reference_sparse(query, key, index_score, indices, scaling)
    tri = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_v2=use_v2)
    
    diff = abs(ref.item() - tri.item())
    passed = diff < 1e-3
    version = "V2" if use_v2 else "V1"
    print(f"[{version}] Accuracy - Ref: {ref.item():.6f}, Triton: {tri.item():.6f}, Diff: {diff:.6e}, Pass: {passed}")
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
    use_v2: bool = True,
):
    """性能测试"""
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("=" * 70)
    print("Sparse Triton Fused Kernel 性能测试 (分块迭代版本)")
    print("=" * 70)
    print(f"参数: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print(f"分块配置: BLOCK_TOPK={BLOCK_TOPK}, BLOCK_D={32 if use_v2 else BLOCK_D}")
    print(f"使用版本: {'V2 (简化版)' if use_v2 else 'V1 (双重分块)'}")
    print(f"理论复杂度: O(seq * topk * head_dim * num_heads) = O({seq_len * topk * head_dim * num_heads:,})")
    
    # 内存使用估算
    if use_v2:
        k_block_size = topk * 32  # [topk, BLOCK_D=32]
    else:
        k_block_size = BLOCK_TOPK * BLOCK_D  # [BLOCK_TOPK, BLOCK_D]
    print(f"K块大小: {k_block_size} elements = {k_block_size * 4 / 1024:.1f} KB")
    print("=" * 70)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, topk, device=device, dtype=torch.float32)
    indices = generate_topk_indices(batch_size, seq_len, topk, device)
    
    results = {}
    
    # Test 1: Triton fused kernel
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_v2=use_v2)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_v2=use_v2)
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
    print("精度测试 - V2 版本 (简化版)")
    print("=" * 70)
    
    print("\n[小规模测试]")
    test_full_accuracy(batch_size=1, num_heads=4, seq_len=64, head_dim=32, topk=16, use_v2=True)
    
    print("\n[中等规模测试]")
    test_full_accuracy(batch_size=1, num_heads=8, seq_len=256, head_dim=64, topk=64, use_v2=True)
    
    print("\n[大规模测试]")
    test_full_accuracy(batch_size=1, num_heads=16, seq_len=1024, head_dim=128, topk=256, use_v2=True)
    
    print("\n[目标配置测试]")
    test_full_accuracy(batch_size=1, num_heads=16, seq_len=4096, head_dim=256, topk=512, use_v2=True)
    
    print("\n")
    test_performance(
        batch_size=1,
        num_heads=16,
        seq_len=4096,
        head_dim=256,
        topk=512,
        num_warmup=2,
        num_benchmark=3,
        use_v2=True,
    )
