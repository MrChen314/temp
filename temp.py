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
from dataclasses import dataclass
from typing import List, Optional


# 固定配置常量
BLOCK_D = 128
BLOCK_TOPK = 256  # topk 分块大小
BLOCK_H = 16      # head 分块大小 (满足 tl.dot 的 N >= 16 要求)
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
    BLOCK_H: tl.constexpr,
):
    """
    Sparse Attention + Loss Kernel (Online Softmax + tl.dot 优化版本)
    
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
                qT = tl.load(q_ptrs, mask=d_block_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)  # [BLOCK_D, BLOCK_H]
                
                # 加载 k_gathered: [BLOCK_TOPK, BLOCK_D]
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d_block[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_block_mask[None, :], other=0.0).to(tl.float32)  # [BLOCK_TOPK, BLOCK_D]
                
                # 累加 QK: [BLOCK_TOPK, BLOCK_H] (float32精度)
                qk += tl.dot(k_gathered, qT)
            
            qk = qk * scaling
            
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
        # Pass 2: 使用全局 max/sum 计算归一化概率，累加到 attn_sum
        # -----------------------------------------------------------------
        for tk_idx in range(num_topk_blocks):
            tk_start = tk_idx * BLOCK_TOPK
            offs_tk = tk_start + tl.arange(0, BLOCK_TOPK)
            tk_mask = offs_tk < topk
            
            indices_block = tl.load(idx_base + offs_tk * stride_ik, mask=tk_mask, other=0).to(tl.int64)
            causal_mask_block = (indices_block > global_query_pos) | (~tk_mask)  # [BLOCK_TOPK]
            
            # 分块计算 QK (支持 head_dim > BLOCK_D)
            qk = tl.zeros([BLOCK_TOPK, BLOCK_H], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d_block = d_start + offs_d
                d_block_mask = offs_d_block < head_dim
                
                # 加载 qT: [BLOCK_D, BLOCK_H]
                q_ptrs = q_batch_base + (h_start + offs_h[None, :]) * stride_qh + offs_d_block[:, None] * stride_qd
                qT = tl.load(q_ptrs, mask=d_block_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)  # [BLOCK_D, BLOCK_H]
                
                # 加载 k_gathered: [BLOCK_TOPK, BLOCK_D]
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d_block[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_block_mask[None, :], other=0.0).to(tl.float32)  # [BLOCK_TOPK, BLOCK_D]
                
                # 累加 QK: [BLOCK_TOPK, BLOCK_H] (float32精度)
                qk += tl.dot(k_gathered, qT)
            
            qk = qk * scaling
            
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
        
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF).to(tl.float32)
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
        is_val = tl.load(is_base + offs_tk * stride_isk, mask=tk_mask, other=NEG_INF).to(tl.float32)
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
    
    # 选择合适的 block_topk (确保 >= 16 以满足 tl.dot 要求)
    if block_topk is None:
        block_topk = min(BLOCK_TOPK, topk)
    # 如果 topk < 16，需要 pad 到 16
    if block_topk < 16:
        block_topk = 16
    
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
    
    # 选择合适的 block_h (确保 >= 16 以满足 tl.dot 要求)
    block_h = min(BLOCK_H, num_heads)
    # 如果 num_heads < 16，需要 pad 到 16
    if block_h < 16:
        block_h = 16
    
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
        BLOCK_H=block_h,
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
# 测试配置和函数
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


def run_single_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """运行单个精度测试"""
    torch.manual_seed(config.seed)
    scaling = 1.0 / (config.head_dim ** 0.5)
    
    # query: [batch, num_heads, chunk_size, head_dim]
    query = torch.randn(config.batch_size, config.num_heads, config.chunk_size, 
                        config.head_dim, device=device, dtype=torch.bfloat16)
    # key: [batch, seq_len, head_dim]
    key = torch.randn(config.batch_size, config.seq_len, config.head_dim, 
                      device=device, dtype=torch.bfloat16)
    
    # Full版本: index_score是 [batch, chunk_size, seq_len]
    index_score_full = torch.randn(config.batch_size, config.chunk_size, config.seq_len, 
                                   device=device, dtype=torch.bfloat16)
    
    # 从index_score生成mask和indices
    chunk_offset = config.seq_len - config.chunk_size
    index_mask, topk_indices = generate_index_mask_from_score(
        index_score_full, config.topk, device, chunk_offset=chunk_offset)
    
    # 从full index_score中gather出sparse index_score给Triton kernel使用
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    # PyTorch Full版本参考
    ref = pytorch_reference(query, key, index_score_full, index_mask, scaling)
    
    # Triton Sparse版本 (传入 chunk_offset)
    tri = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, 
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


def test_full_accuracy(configs: List[TestConfig]):
    """
    批量运行精度测试
    
    Args:
        configs: 测试配置列表
    """
    print("\n" + "=" * 90)
    print("精度测试 (PyTorch Full vs Triton Sparse)")
    print("=" * 90)
    
    results = []
    for config in configs:
        result = run_single_accuracy_test(config)
        results.append(result)
    
    # 打印表格格式的结果
    print(f"\n{'Name':<15} {'Config':<50} {'PyTorch':<12} {'Triton':<12} {'RelDiff':<12} {'Pass':<6}")
    print("-" * 107)
    for r in results:
        print(f"{r['config'].name:<15} {str(r['config']):<50} "
              f"{r['ref']:<12.4f} {r['tri']:<12.4f} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    # 汇总
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 107)
    print(f"总计: {passed_count}/{len(results)} 通过")
    
    return results


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
    query = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device, dtype=torch.bfloat16)
    # key: [batch, seq_len, head_dim] - 完整序列的key (KV cache)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    # chunk_offset: 假设当前chunk从序列末尾开始
    chunk_offset = seq_len - chunk_size
    
    index_score_full = torch.randn(batch_size, chunk_size, seq_len, device=device, dtype=torch.bfloat16)
    index_mask, topk_indices = generate_index_mask_from_score(index_score_full, topk, device, chunk_offset=chunk_offset)
    index_score_sparse = torch.gather(index_score_full, dim=-1, index=topk_indices)
    
    results = {}
    memory_stats = {}
    
    # 记录基准显存 (输入数据分配后)
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    # Test 1: Triton fused kernel (Sparse) - 带显存监控
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    # 重置峰值统计，只测量 benchmark 期间的峰值
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    triton_peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    results['triton_sparse'] = triton_time
    memory_stats['triton_sparse'] = triton_peak_memory
    
    # Test 2: PyTorch reference (Full) - 仅当 triton_only=False 时执行
    if not triton_only:
        # 清理 Triton 的临时显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        for _ in range(num_warmup):
            _ = pytorch_reference(query, key, index_score_full, index_mask, scaling)
        torch.cuda.synchronize()
        
        # 重置峰值统计，只测量 benchmark 期间的峰值
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(num_benchmark):
            _ = pytorch_reference(query, key, index_score_full, index_mask, scaling)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / num_benchmark * 1000
        pytorch_peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        results['pytorch_full'] = pytorch_time
        memory_stats['pytorch_full'] = pytorch_peak_memory
    
    print(f"\n>>> 性能结果 (warmup={num_warmup}, iters={num_benchmark})")
    if triton_only:
        print(f"  Triton Sparse fused:   {triton_time:.3f} ms")
    else:
        print(f"  PyTorch Full ref:      {pytorch_time:.3f} ms")
        print(f"  Triton Sparse fused:   {triton_time:.3f} ms (加速: {pytorch_time/triton_time:.2f}x)")
    
    print(f"\n>>> 显存峰值")
    print(f"  基准显存 (输入数据):    {base_memory:.2f} GB")
    print(f"  Triton Sparse 峰值:    {memory_stats['triton_sparse']:.2f} GB (增量: {memory_stats['triton_sparse'] - base_memory:.2f} GB)")
    if not triton_only:
        print(f"  PyTorch Full 峰值:     {memory_stats['pytorch_full']:.2f} GB (增量: {memory_stats['pytorch_full'] - base_memory:.2f} GB)")
        print(f"  显存节省:              {memory_stats['pytorch_full'] - memory_stats['triton_sparse']:.2f} GB ({(1 - memory_stats['triton_sparse']/memory_stats['pytorch_full'])*100:.1f}%)")
    
    results['memory'] = memory_stats
    return results


if __name__ == "__main__":
    # 定义测试配置
    accuracy_configs = [
        TestConfig(name="小规模", batch_size=1, num_heads=4, chunk_size=32, seq_len=64, head_dim=32, topk=16),
        TestConfig(name="中等规模", batch_size=1, num_heads=8, chunk_size=128, seq_len=256, head_dim=64, topk=64),
        TestConfig(name="大规模", batch_size=1, num_heads=16, chunk_size=512, seq_len=1024, head_dim=128, topk=256),
        TestConfig(name="多batch", batch_size=4, num_heads=8, chunk_size=64, seq_len=128, head_dim=64, topk=32),
        TestConfig(name="大head_dim", batch_size=1, num_heads=8, chunk_size=128, seq_len=256, head_dim=256, topk=64),
        TestConfig(name="大规模", batch_size=1, num_heads=128, chunk_size=4096, seq_len=4096, head_dim=576, topk=2048),
        TestConfig(name="大规模", batch_size=1, num_heads=128, chunk_size=4096, seq_len=8192, head_dim=576, topk=2048),
        TestConfig(name="大规模", batch_size=1, num_heads=128, chunk_size=8192, seq_len=8192, head_dim=576, topk=2048),
    ]
    
    # 运行精度测试
    test_full_accuracy(accuracy_configs)
    
    # 性能测试
    print("\n")
    test_performance(
        batch_size=1,
        num_heads=128,
        chunk_size=4 * 1024,
        seq_len=8 * 1024,
        head_dim=576,
        topk=2048,
        num_warmup=1,
        num_benchmark=3,
        triton_only=False,
    )
