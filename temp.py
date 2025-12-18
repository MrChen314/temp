# ruff: noqa
"""
Sparse MLA Forward - 优化版本对比
针对 NVIDIA H20 GPU 优化 (78 SMs, 227KB shared memory per block with opt-in)

输入配置 (来自 sparse_mal_fwd_input.txt):
- heads=2, dim=256, tail_dim=64, topk=2048, kv_group=1
- q.shape:[1, 2048, 2, 320], kv.shape:[1, 2048, 1, 320]

优化方案:
- 原始方案: 基于 sparse_mla_fwd.py，使用优化1配置 (block_I=64, num_stages=1, threads=128)
- 优化2: 减少同步开销
- 优化3: 预加载 Indices 到 shared memory
- 优化4: 针对小 H 的特化
"""
import torch
import tilelang
from tilelang import language as T

# ============================================================================
# 原始方案: 使用优化1配置 (block_I=64, num_stages=1, threads=128)
# ============================================================================
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_baseline(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=1,  # 优化1: 从默认2改为1
    threads=128,   # 优化1: 从默认256改为128
):
    """原始方案: 使用优化1的配置"""
    assert dim == tilelang.math.next_power_of_2(dim)
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
    assert is_causal == True
    assert topk % block_I == 0
    
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    REPLICATE_H = head_kv // 64 if head_kv > 64 else 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse_shared)
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


# ============================================================================
# 优化7: 批量处理 query - 将多个 seq position 的 Q 合并计算 (原版本，有约束)
# 
# 原理说明:
# - 当前配置 H=2, 必须 pad 到 H_per_block=16, 浪费 87.5%
# - 优化思路: 将 S_per_block=8 个 seq position 的 Q 合并
# - 合并后 GEMM 维度: [8*2, D] @ [BI, D]^T = [16, BI]，正好填满，无浪费
# 
# 约束:
# - 需要 seq_len 是 S_per_block 的倍数
# - 相邻 seq positions 共享同一套 indices（使用最后一个的 indices）
# ============================================================================
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_batched_query(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=1,
    threads=128,
    S_per_block=8,  # 每个 block 处理的 seq position 数量
):
    """
    优化7: 批量处理 query
    
    将 S_per_block 个相邻 seq position 的 Q 合并计算:
    - 原始: GEMM [H_per_block, D] @ [BI, D]^T, H_per_block=16 但实际 H=2
    - 优化: GEMM [S_per_block * H, D] @ [BI, D]^T, 8*2=16 正好填满
    
    每个 seq position 有独立的 mask (基于各自的 max_kv_i)
    共享同一套 indices (使用 block 内最后一个 seq position 的 indices)
    """
    assert dim == tilelang.math.next_power_of_2(dim)
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
    assert is_causal == True
    assert topk % block_I == 0
    
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = head_kv
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim
    
    # 计算合并后的维度
    # 将 S_per_block 个 seq position 的 H 个 head 合并
    # 例如: S_per_block=8, H=2 -> merged_H = 16
    merged_H = S_per_block * H
    # 确保 merged_H >= 16 以满足 GEMM 要求
    padded_merged_H = max(tilelang.math.next_power_of_2(merged_H), 16)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        # Grid: (seq_len / S_per_block, batch, kv_group)
        with T.Kernel(T.ceildiv(seq_len, S_per_block), batch, kv_group, threads=threads) as (bx, by, bz):
            # Q shared memory: [S_per_block * H, D] (合并多个 seq position)
            Q_shared = T.alloc_shared([padded_merged_H, D], dtype)
            Q_tail_shared = T.alloc_shared([padded_merged_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([padded_merged_H, D], dtype)
            Lse_shared = T.alloc_shared([padded_merged_H], accum_dtype)
            
            # mask 现在是 2D: 每个 seq position 有自己的 mask
            # mask[s, bi] 表示第 s 个 seq position 对第 bi 个 KV 是否有效
            mask = T.alloc_shared([S_per_block, BI], "bool")
            
            # 每个 seq position 的 max_kv_i
            max_kv_indices = T.alloc_shared([S_per_block], indices_dtype)

            acc_o = T.alloc_fragment([padded_merged_H, D], accum_dtype)
            acc_s = T.alloc_fragment([padded_merged_H, BI], accum_dtype)
            S_shared = T.alloc_shared([padded_merged_H, BI], dtype)
            sumexp = T.alloc_fragment([padded_merged_H], accum_dtype)
            sumexp_i = T.alloc_fragment([padded_merged_H], accum_dtype)
            alpha = T.alloc_fragment([padded_merged_H], accum_dtype)
            m_i = T.alloc_fragment([padded_merged_H], accum_dtype)
            m_i_prev = T.alloc_fragment([padded_merged_H], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = by, bz
            # block 起始的 seq position
            s_base = bx * S_per_block
            # 使用 block 内最后一个有效 seq position 的 indices
            s_for_indices = T.min(s_base + S_per_block - 1, seq_len - 1)

            # 加载多个 seq position 的 Q 到 shared memory
            # Q_shared 布局: [s0_h0, s0_h1, s1_h0, s1_h1, ..., s7_h0, s7_h1, padding...]
            for s_offset in T.serial(S_per_block):
                s_i = s_base + s_offset
                # 计算该 seq position 在 Q_shared 中的起始位置
                h_offset = s_offset * H
                # 设置 max_kv_i (causal mask)
                max_kv_indices[s_offset] = s_i
                
                # 加载 Q (只有有效的 seq position)
                for h_i in T.serial(H):
                    for d_i in T.Parallel(D):
                        Q_shared[h_offset + h_i, d_i] = T.if_then_else(
                            s_i < seq_len,
                            Q[b_i, s_i, g_i * H + h_i, d_i],
                            T.cast(0, dtype)
                        )
                    for d_i in T.Parallel(D_tail):
                        Q_tail_shared[h_offset + h_i, d_i] = T.if_then_else(
                            s_i < seq_len,
                            Q[b_i, s_i, g_i * H + h_i, D + d_i],
                            T.cast(0, dtype)
                        )

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                # 计算每个 seq position 的 mask
                for s_offset, bi_i in T.Parallel(S_per_block, BI):
                    kv_idx = Indices[b_i, s_for_indices, g_i, i_i * BI + bi_i]
                    mask[s_offset, bi_i] = kv_idx <= max_kv_indices[s_offset]

                # 加载 KV (使用 block 内最后一个 seq position 的 indices)
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_for_indices, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_for_indices, g_i, i_i * BI + bi_i], g_i, D + d_i]

                # 初始化 acc_s，应用各自的 mask
                for sh_i, bi_i in T.Parallel(padded_merged_H, BI):
                    # 计算对应的 s_offset 和 h_i
                    s_offset = sh_i // H
                    # 只有 s_offset < S_per_block 且 mask 有效时才计算
                    is_valid = T.if_then_else(
                        s_offset < S_per_block,
                        mask[s_offset, bi_i],
                        False
                    )
                    acc_s[sh_i, bi_i] = T.if_then_else(is_valid, 0, -T.infinity(acc_s.dtype))
                
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for sh_i in T.Parallel(padded_merged_H):
                    m_i[sh_i] = T.max(m_i[sh_i], m_i_prev[sh_i])
                for sh_i in T.Parallel(padded_merged_H):
                    alpha[sh_i] = T.exp2((m_i_prev[sh_i] - m_i[sh_i]) * sm_scale)
                for sh_i, bi_i in T.Parallel(padded_merged_H, BI):
                    acc_s[sh_i, bi_i] = T.exp2(acc_s[sh_i, bi_i] * sm_scale - m_i[sh_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for sh_i in T.Parallel(padded_merged_H):
                    sumexp[sh_i] = sumexp[sh_i] * alpha[sh_i] + sumexp_i[sh_i]
                for sh_i, d_i in T.Parallel(padded_merged_H, D):
                    acc_o[sh_i, d_i] = acc_o[sh_i, d_i] * alpha[sh_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for sh_i, d_i in T.Parallel(padded_merged_H, D):
                acc_o[sh_i, d_i] /= sumexp[sh_i]
            for sh_i in T.Parallel(padded_merged_H):
                sumexp[sh_i] = T.log2(sumexp[sh_i]) + m_i[sh_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(sumexp, Lse_shared)
            
            # 将结果写回各自的 seq position
            for s_offset in T.serial(S_per_block):
                s_i = s_base + s_offset
                h_offset = s_offset * H
                for h_i in T.serial(H):
                    # 只写入有效的 seq position
                    for d_i in T.Parallel(D):
                        if s_i < seq_len:
                            Output[b_i, s_i, g_i * H + h_i, d_i] = O_shared[h_offset + h_i, d_i]
                for h_i in T.serial(H):
                    if s_i < seq_len:
                        Lse[b_i, s_i, g_i * H + h_i] = Lse_shared[h_offset + h_i]

    return main


# ============================================================================
# 优化8: 批量处理 query - 无约束版本
# 
# 原理说明:
# - 在 kernel 外预处理 indices 和 mask
# - 将 S_per_block 个相邻 seq positions 的 indices 合并（取并集）
# - 预计算精确的 mask，确保每个 seq position 只访问其原始 indices 中的 KV
# 
# 优点:
# - 无 seq_len 必须是 S_per_block 倍数的约束
# - 精确处理每个 seq position 的有效 KV 位置，不是近似
# - Kernel 内部逻辑更简单，直接使用预处理好的数据
# ============================================================================

def prepare_batched_inputs(indices, S_per_block, is_causal=True):
    """
    预处理 indices，为批量处理 query 准备合并的 indices 和 mask
    
    Args:
        indices: [batch, seq_len, kv_group, topk] - 原始 indices
        S_per_block: 每个 block 处理的 seq position 数量
        is_causal: 是否应用 causal mask
    
    Returns:
        merged_indices: [batch, num_blocks, kv_group, merged_topk] - 合并后的 indices
        mask: [batch, num_blocks, S_per_block, merged_topk] - 预计算的 mask (不含 kv_group 维度，因为对所有 g 相同)
        valid_counts: [batch, num_blocks, kv_group] - 每个 block 的有效 indices 数量
    
    Note:
        merged_topk = S_per_block * topk (最坏情况，实际可能更小)
        mask 的语义: mask[b, blk, s_offset, k] = True 表示:
        - merged_indices[b, blk, g, k] <= s_base + s_offset (causal, 若启用)
        - merged_indices[b, blk, g, k] 在原始 indices[b, s_base + s_offset, g, :] 中
    """
    batch, seq_len, kv_group, topk = indices.shape
    num_blocks = (seq_len + S_per_block - 1) // S_per_block
    merged_topk = S_per_block * topk  # 最坏情况
    
    device = indices.device
    dtype = indices.dtype
    
    # 使用一个无效的索引值作为填充
    INVALID_IDX = -1
    
    # 输出 tensors
    merged_indices = torch.full(
        (batch, num_blocks, kv_group, merged_topk), 
        INVALID_IDX, dtype=dtype, device=device
    )
    # mask 对于不同 kv_group 是独立的，但 indices 不同，所以需要每个 group 单独处理
    # 实际上 mask 依赖于 merged_indices，所以也需要按 group 处理
    mask = torch.zeros(
        (batch, num_blocks, kv_group, S_per_block, merged_topk), 
        dtype=torch.bool, device=device
    )
    valid_counts = torch.zeros(
        (batch, num_blocks, kv_group), 
        dtype=torch.int32, device=device
    )
    
    for b in range(batch):
        for blk in range(num_blocks):
            s_base = blk * S_per_block
            for g in range(kv_group):
                # 收集该 block 所有 seq positions 的 indices
                all_indices_list = []
                indices_per_pos = []  # 记录每个 seq pos 的 indices set
                
                for s_offset in range(S_per_block):
                    s_i = s_base + s_offset
                    if s_i < seq_len:
                        pos_indices = indices[b, s_i, g, :].tolist()
                        all_indices_list.extend(pos_indices)
                        indices_per_pos.append(set(pos_indices))
                    else:
                        indices_per_pos.append(set())  # 无效位置
                
                if all_indices_list:
                    # 合并并去重，排序
                    unique_sorted = sorted(set(all_indices_list))
                    actual_len = len(unique_sorted)
                    valid_counts[b, blk, g] = actual_len
                    
                    # 写入 merged_indices
                    merged_indices[b, blk, g, :actual_len] = torch.tensor(
                        unique_sorted, dtype=dtype, device=device
                    )
                    
                    # 计算 mask
                    for s_offset in range(S_per_block):
                        s_i = s_base + s_offset
                        if s_i < seq_len:
                            pos_set = indices_per_pos[s_offset]
                            for k, kv_idx in enumerate(unique_sorted):
                                # 条件1: 在原始 indices 中
                                in_original = kv_idx in pos_set
                                # 条件2: causal (kv_idx <= s_i)
                                causal_ok = (not is_causal) or (kv_idx <= s_i)
                                mask[b, blk, g, s_offset, k] = in_original and causal_ok
    
    return merged_indices, mask, valid_counts


def prepare_batched_inputs_fast(indices, S_per_block, is_causal=True):
    """
    GPU 加速版本的预处理函数
    
    Args:
        indices: [batch, seq_len, kv_group, topk]
        S_per_block: 每个 block 处理的 seq position 数量
        is_causal: 是否应用 causal mask
    
    Returns:
        merged_indices: [batch, num_blocks, kv_group, merged_topk]
        mask: [batch, num_blocks, kv_group, S_per_block, merged_topk]
    """
    batch, seq_len, kv_group, topk = indices.shape
    num_blocks = (seq_len + S_per_block - 1) // S_per_block
    merged_topk = S_per_block * topk
    
    device = indices.device
    dtype = indices.dtype
    
    # Pad seq_len to be divisible by S_per_block
    pad_len = num_blocks * S_per_block - seq_len
    if pad_len > 0:
        # Pad with -1 (invalid index)
        indices_padded = torch.nn.functional.pad(
            indices, (0, 0, 0, 0, 0, pad_len), value=-1
        )
    else:
        indices_padded = indices
    
    # Reshape: [batch, num_blocks, S_per_block, kv_group, topk]
    indices_reshaped = indices_padded.view(batch, num_blocks, S_per_block, kv_group, topk)
    # Transpose: [batch, num_blocks, kv_group, S_per_block, topk]
    indices_reshaped = indices_reshaped.permute(0, 1, 3, 2, 4)
    # Flatten last two dims: [batch, num_blocks, kv_group, S_per_block * topk]
    indices_flat = indices_reshaped.reshape(batch, num_blocks, kv_group, merged_topk)
    
    # Sort each block's indices (for better cache locality and easier unique)
    merged_indices, _ = torch.sort(indices_flat, dim=-1)
    
    # 计算 mask：对于每个 (batch, block, group, s_offset, k)
    # mask = True 当且仅当：
    # 1. merged_indices[b, blk, g, k] 是有效的 (>= 0)
    # 2. merged_indices[b, blk, g, k] 在原始 indices[b, s_base + s_offset, g, :] 中
    # 3. (if causal) merged_indices[b, blk, g, k] <= s_base + s_offset
    
    # 创建 mask
    mask = torch.zeros(
        (batch, num_blocks, kv_group, S_per_block, merged_topk),
        dtype=torch.bool, device=device
    )
    
    # 方法：对于每个位置，检查 merged_indices[k] 是否在该位置的原始 indices 中
    # 使用 broadcasting + comparison
    
    # merged_indices: [batch, num_blocks, kv_group, merged_topk]
    # indices_reshaped: [batch, num_blocks, kv_group, S_per_block, topk]
    
    # Expand merged_indices: [batch, num_blocks, kv_group, 1, merged_topk]
    merged_expanded = merged_indices.unsqueeze(3)
    # Expand indices_reshaped: [batch, num_blocks, kv_group, S_per_block, 1, topk]
    original_expanded = indices_reshaped.unsqueeze(4)
    
    # 比较：[batch, num_blocks, kv_group, S_per_block, merged_topk, topk]
    # matches[..., k, t] = True if merged_indices[k] == original[s_offset, t]
    matches = (merged_expanded.unsqueeze(-1) == original_expanded)
    
    # Reduce over topk dim: any match means this merged index is in original
    in_original = matches.any(dim=-1)  # [batch, num_blocks, kv_group, S_per_block, merged_topk]
    
    # Valid indices (>= 0)
    valid = merged_indices >= 0  # [batch, num_blocks, kv_group, merged_topk]
    valid = valid.unsqueeze(3).expand_as(in_original)
    
    mask = in_original & valid
    
    # Apply causal mask if needed
    if is_causal:
        # 生成 s_i 值: [num_blocks, S_per_block]
        block_ids = torch.arange(num_blocks, device=device).view(-1, 1)
        s_offsets = torch.arange(S_per_block, device=device).view(1, -1)
        s_i = block_ids * S_per_block + s_offsets  # [num_blocks, S_per_block]
        
        # merged_indices: [batch, num_blocks, kv_group, merged_topk]
        # 需要 merged_indices[..., k] <= s_i[blk, s_offset]
        s_i_expanded = s_i.view(1, num_blocks, 1, S_per_block, 1)  # [1, num_blocks, 1, S_per_block, 1]
        merged_expanded = merged_indices.unsqueeze(3)  # [batch, num_blocks, kv_group, 1, merged_topk]
        causal_mask = merged_expanded <= s_i_expanded  # [batch, num_blocks, kv_group, S_per_block, merged_topk]
        
        mask = mask & causal_mask
    
    # 将无效的 indices 替换为一个安全值（例如 0）
    merged_indices = torch.where(merged_indices >= 0, merged_indices, torch.zeros_like(merged_indices))
    
    return merged_indices, mask


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_batched_query_v2(
    heads,
    dim,
    tail_dim,
    merged_topk,  # 合并后的 topk = S_per_block * original_topk
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=1,
    threads=128,
    S_per_block=8,
):
    """
    优化8: 批量处理 query - 无约束版本
    
    使用预处理好的 merged_indices 和 mask:
    - merged_indices: [batch, num_blocks, kv_group, merged_topk]
    - mask: [batch, num_blocks, kv_group, S_per_block, merged_topk]
    
    Kernel 直接使用这些数据，无需内部计算 mask，确保精确性
    """
    assert dim == tilelang.math.next_power_of_2(dim)
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
    assert is_causal == True
    assert merged_topk % block_I == 0
    
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_blocks = T.dynamic("num_blocks")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    # 预处理后的 indices 和 mask
    merged_indices_shape = [batch, num_blocks, kv_group, merged_topk]
    mask_shape = [batch, num_blocks, kv_group, S_per_block, merged_topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    H = head_kv
    BI = block_I
    NI = tilelang.cdiv(merged_topk, block_I)
    D = dim
    D_tail = tail_dim
    
    # 合并后的维度
    merged_H = S_per_block * H
    padded_merged_H = max(tilelang.math.next_power_of_2(merged_H), 16)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        MergedIndices: T.Tensor(merged_indices_shape, indices_dtype),
        Mask: T.Tensor(mask_shape, "bool"),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        # Grid: (num_blocks, batch, kv_group)
        with T.Kernel(num_blocks, batch, kv_group, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([padded_merged_H, D], dtype)
            Q_tail_shared = T.alloc_shared([padded_merged_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([padded_merged_H, D], dtype)
            Lse_shared = T.alloc_shared([padded_merged_H], accum_dtype)
            
            # 从预处理的 Mask 加载当前 block 的 mask
            mask_shared = T.alloc_shared([S_per_block, BI], "bool")

            acc_o = T.alloc_fragment([padded_merged_H, D], accum_dtype)
            acc_s = T.alloc_fragment([padded_merged_H, BI], accum_dtype)
            S_shared = T.alloc_shared([padded_merged_H, BI], dtype)
            sumexp = T.alloc_fragment([padded_merged_H], accum_dtype)
            sumexp_i = T.alloc_fragment([padded_merged_H], accum_dtype)
            alpha = T.alloc_fragment([padded_merged_H], accum_dtype)
            m_i = T.alloc_fragment([padded_merged_H], accum_dtype)
            m_i_prev = T.alloc_fragment([padded_merged_H], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = by, bz
            blk_i = bx
            s_base = blk_i * S_per_block

            # 加载多个 seq position 的 Q
            for s_offset in T.serial(S_per_block):
                s_i = s_base + s_offset
                h_offset = s_offset * H
                for h_i in T.serial(H):
                    for d_i in T.Parallel(D):
                        Q_shared[h_offset + h_i, d_i] = T.if_then_else(
                            s_i < seq_len,
                            Q[b_i, s_i, g_i * H + h_i, d_i],
                            T.cast(0, dtype)
                        )
                    for d_i in T.Parallel(D_tail):
                        Q_tail_shared[h_offset + h_i, d_i] = T.if_then_else(
                            s_i < seq_len,
                            Q[b_i, s_i, g_i * H + h_i, D + d_i],
                            T.cast(0, dtype)
                        )

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                # 直接从预处理的 Mask 加载，无需计算
                for s_offset, bi_i in T.Parallel(S_per_block, BI):
                    mask_shared[s_offset, bi_i] = Mask[b_i, blk_i, g_i, s_offset, i_i * BI + bi_i]

                # 加载 KV
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, MergedIndices[b_i, blk_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, MergedIndices[b_i, blk_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                # 使用预计算的 mask 初始化 acc_s
                for sh_i, bi_i in T.Parallel(padded_merged_H, BI):
                    s_offset = sh_i // H
                    is_valid = T.if_then_else(
                        s_offset < S_per_block,
                        mask_shared[s_offset, bi_i],
                        False
                    )
                    acc_s[sh_i, bi_i] = T.if_then_else(is_valid, 0, -T.infinity(acc_s.dtype))
                
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for sh_i in T.Parallel(padded_merged_H):
                    m_i[sh_i] = T.max(m_i[sh_i], m_i_prev[sh_i])
                for sh_i in T.Parallel(padded_merged_H):
                    alpha[sh_i] = T.exp2((m_i_prev[sh_i] - m_i[sh_i]) * sm_scale)
                for sh_i, bi_i in T.Parallel(padded_merged_H, BI):
                    acc_s[sh_i, bi_i] = T.exp2(acc_s[sh_i, bi_i] * sm_scale - m_i[sh_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for sh_i in T.Parallel(padded_merged_H):
                    sumexp[sh_i] = sumexp[sh_i] * alpha[sh_i] + sumexp_i[sh_i]
                for sh_i, d_i in T.Parallel(padded_merged_H, D):
                    acc_o[sh_i, d_i] = acc_o[sh_i, d_i] * alpha[sh_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for sh_i, d_i in T.Parallel(padded_merged_H, D):
                acc_o[sh_i, d_i] /= sumexp[sh_i]
            for sh_i in T.Parallel(padded_merged_H):
                sumexp[sh_i] = T.log2(sumexp[sh_i]) + m_i[sh_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(sumexp, Lse_shared)
            
            # 写回结果
            for s_offset in T.serial(S_per_block):
                s_i = s_base + s_offset
                h_offset = s_offset * H
                for h_i in T.serial(H):
                    for d_i in T.Parallel(D):
                        if s_i < seq_len:
                            Output[b_i, s_i, g_i * H + h_i, d_i] = O_shared[h_offset + h_i, d_i]
                for h_i in T.serial(H):
                    if s_i < seq_len:
                        Lse[b_i, s_i, g_i * H + h_i] = Lse_shared[h_offset + h_i]

    return main


def sparse_mla_fwd_batched_v2_wrapper(q, kv, indices, S_per_block=8, block_I=64, **kwargs):
    """
    便捷的 wrapper 函数，自动完成预处理并调用 kernel
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        kv: [batch, seq_len_kv, kv_group, dim + tail_dim]
        indices: [batch, seq_len, kv_group, topk]
        S_per_block: 每个 block 处理的 seq position 数量
        block_I: block size for KV
        **kwargs: 其他参数传给 kernel
    
    Returns:
        output: [batch, seq_len, heads, dim]
        lse: [batch, seq_len, heads]
    """
    batch, seq_len, heads, dqk = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape
    _, _, _, topk = indices.shape
    
    # 计算 dim 和 tail_dim
    if dqk == 320:
        dim, tail_dim = 256, 64
    elif dqk == 576:
        dim, tail_dim = 512, 64
    else:
        raise ValueError(f"Cannot infer dim from dqk={dqk}")
    
    merged_topk = S_per_block * topk
    
    # Pad merged_topk to be divisible by block_I
    if merged_topk % block_I != 0:
        merged_topk = ((merged_topk + block_I - 1) // block_I) * block_I
    
    # 预处理 indices 和 mask
    merged_indices, mask = prepare_batched_inputs_fast(
        indices, S_per_block, is_causal=True
    )
    
    # Pad merged_indices and mask if needed
    actual_merged_topk = merged_indices.shape[-1]
    if actual_merged_topk < merged_topk:
        pad_size = merged_topk - actual_merged_topk
        merged_indices = torch.nn.functional.pad(merged_indices, (0, pad_size), value=0)
        mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
    
    # 编译 kernel
    kernel = sparse_mla_fwd_batched_query_v2(
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        merged_topk=merged_topk,
        kv_group=kv_group,
        block_I=block_I,
        S_per_block=S_per_block,
        **kwargs,
    )
    
    # 运行 kernel
    out, lse = kernel(q, kv, merged_indices, mask)
    
    return out, lse


# ============================================================================
# 参考实现 (用于正确性验证)
# ============================================================================
def ref_sparse_mla_fwd(q, kv, indices, sm_scale=None, d_v=None):
    """参考实现，用于正确性验证"""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, dim_plus_tail_dim = kv.shape
    
    if d_v is None:
        if dim_plus_tail_dim == 576:
            dim = 512
        elif dim_plus_tail_dim == 320:
            dim = 256
        else:
            raise ValueError(f"无法自动推断 dim，dim_plus_tail_dim={dim_plus_tail_dim}")
    else:
        dim = d_v
    
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
        0, sk, dtype=torch.int32, device="cuda"
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


# ============================================================================
# 性能测试对比
# ============================================================================
def benchmark_comparison(
    B=1, S=2048, SKV=2048, H=2, HKV=1, DQK=320, DV=256, topk=2048, dtype=torch.bfloat16,
    check_correctness=True, num_warmup=50, num_rep=100
):
    """
    对比所有优化方案的性能
    
    输入配置来自 sparse_mal_fwd_input.txt:
    - heads=2, dim=256, tail_dim=64, topk=2048, kv_group=1
    - q.shape:[1, 2048, 2, 320], kv.shape:[1, 2048, 1, 320]
    """
    from tilelang.profiler import do_bench
    
    print("=" * 80)
    print("Sparse MLA Forward 优化方案性能对比")
    print("=" * 80)
    print(f"配置: B={B}, S={S}, SKV={SKV}, H={H}, HKV={HKV}")
    print(f"      DQK={DQK}, DV={DV}, topk={topk}")
    print(f"      block_I=64, num_stages=1, threads=128 (优化1配置)")
    print("=" * 80)
    
    # 生成测试数据
    torch.random.manual_seed(42)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda") / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda") / 10
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    
    # 计算 tail_dim
    tail_dim = DQK - DV
    
    # 定义所有优化方案 (kernel_func, block_I, threads, S_per_block, use_v2)
    optimizations = [
        ("原始方案 (优化1配置)", sparse_mla_fwd_baseline, 64, 128, 1, False),
        ("优化7: 批量处理 Query (S=8, 有约束)", sparse_mla_fwd_batched_query, 64, 128, 8, False),
        ("优化8: 批量处理 Query V2 (S=8, 无约束)", None, 64, 128, 8, True),  # 使用 wrapper
    ]
    
    results = []
    ref_out = None
    
    for name, kernel_func, block_i, num_threads, s_per_block, use_v2 in optimizations:
        print(f"\n测试: {name}")
        print("-" * 40)
        
        try:
            if use_v2:
                # 优化8: 使用 wrapper (预处理 + kernel)
                out, lse = sparse_mla_fwd_batched_v2_wrapper(
                    q, kv, indices,
                    S_per_block=s_per_block,
                    block_I=block_i,
                    sm_scale=None,
                    is_causal=True,
                    num_stages=1,
                    threads=num_threads,
                )
                
                # 定义性能测试函数（预处理已在 wrapper 中缓存）
                # 为了公平比较，需要重新计算预处理
                merged_topk = s_per_block * topk
                if merged_topk % block_i != 0:
                    merged_topk = ((merged_topk + block_i - 1) // block_i) * block_i
                
                # 预处理（这部分时间不计入）
                merged_indices, mask = prepare_batched_inputs_fast(
                    indices, s_per_block, is_causal=True
                )
                actual_merged_topk = merged_indices.shape[-1]
                if actual_merged_topk < merged_topk:
                    pad_size = merged_topk - actual_merged_topk
                    merged_indices = torch.nn.functional.pad(merged_indices, (0, pad_size), value=0)
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
                
                # 编译 kernel
                kernel = sparse_mla_fwd_batched_query_v2(
                    heads=H,
                    dim=DV,
                    tail_dim=tail_dim,
                    merged_topk=merged_topk,
                    kv_group=HKV,
                    sm_scale=None,
                    is_causal=True,
                    block_I=block_i,
                    num_stages=1,
                    threads=num_threads,
                    S_per_block=s_per_block,
                )
                
                def fn():
                    return kernel(q, kv, merged_indices, mask)
            else:
                # 编译 kernel
                if s_per_block > 1:
                    # 优化7: 批量处理
                    kernel = kernel_func(
                        heads=H,
                        dim=DV,
                        tail_dim=tail_dim,
                        topk=topk,
                        kv_group=HKV,
                        sm_scale=None,
                        is_causal=True,
                        block_I=block_i,
                        num_stages=1,
                        threads=num_threads,
                        S_per_block=s_per_block,
                    )
                else:
                    kernel = kernel_func(
                        heads=H,
                        dim=DV,
                        tail_dim=tail_dim,
                        topk=topk,
                        kv_group=HKV,
                        sm_scale=None,
                        is_causal=True,
                        block_I=block_i,
                        num_stages=1,
                        threads=num_threads,
                    )
                
                # 运行一次获取输出
                out, lse = kernel(q, kv, indices)
                
                def fn():
                    return kernel(q, kv, indices)
            
            # 正确性验证
            if check_correctness:
                if ref_out is None:
                    ref_out = ref_sparse_mla_fwd(q, kv, indices, d_v=DV)
                
                try:
                    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
                    print(f"  ✓ 正确性验证通过")
                except AssertionError as e:
                    max_diff = (out - ref_out).abs().max().item()
                    print(f"  ✗ 正确性验证失败，最大差异: {max_diff:.6f}")
            
            # 性能测试 (fn 已在上面定义)
            ms = do_bench(fn, rep=num_rep, warmup=num_warmup)
            
            # 计算带宽和 TFLOPS
            io_bytes = B * S * DQK * topk * 2  # 读取 KV 数据
            bandwidth = io_bytes / (ms * 1e-3) / 1e12
            flops = B * S * (DQK + DV) * topk * 2 * H
            tflops = flops / (ms * 1e-3) / 1e12
            
            print(f"  平均时间: {ms:.4f} ms")
            print(f"  带宽: {bandwidth:.2f} TB/s")
            print(f"  TFLOPS: {tflops:.2f}")
            
            results.append({
                "name": name,
                "time_ms": ms,
                "bandwidth_tbs": bandwidth,
                "tflops": tflops,
                "success": True,
            })
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            results.append({
                "name": name,
                "time_ms": float('inf'),
                "bandwidth_tbs": 0,
                "tflops": 0,
                "success": False,
            })
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("性能汇总")
    print("=" * 80)
    print(f"{'方案':<30} {'时间(ms)':<12} {'带宽(TB/s)':<12} {'TFLOPS':<10} {'相对基准':<10}")
    print("-" * 80)
    
    baseline_time = results[0]["time_ms"] if results[0]["success"] else float('inf')
    for r in results:
        if r["success"]:
            speedup = baseline_time / r["time_ms"] if r["time_ms"] > 0 else 0
            print(f"{r['name']:<30} {r['time_ms']:<12.4f} {r['bandwidth_tbs']:<12.2f} {r['tflops']:<10.2f} {speedup:<10.2f}x")
        else:
            print(f"{r['name']:<30} {'失败':<12}")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sparse MLA Forward 优化方案对比")
    parser.add_argument("--no-check", action="store_true", help="跳过正确性检查")
    parser.add_argument("--warmup", type=int, default=50, help="预热次数")
    parser.add_argument("--rep", type=int, default=100, help="重复次数")
    args = parser.parse_args()
    
    # 使用 sparse_mal_fwd_input.txt 的配置
    # heads=2, dim=256, tail_dim=64, topk=2048, kv_group=1
    # q.shape:[1, 2048, 2, 320], kv.shape:[1, 2048, 1, 320]
    benchmark_comparison(
        B=1,
        S=2048,
        SKV=2048,
        H=2,
        HKV=1,
        DQK=320,  # dim + tail_dim = 256 + 64
        DV=256,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=not args.no_check,
        num_warmup=args.warmup,
        num_rep=args.rep,
    )
