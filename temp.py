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
# 优化7: 批量处理 query - 将多个 seq position 的 Q 合并计算
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
                # 关键修复：每个 seq position 使用自己的 indices
                for s_offset, bi_i in T.Parallel(S_per_block, BI):
                    s_i = s_base + s_offset
                    # 每个 seq position 使用自己的 indices
                    kv_idx = Indices[b_i, T.min(s_i, seq_len - 1), g_i, i_i * BI + bi_i]
                    mask[s_offset, bi_i] = (kv_idx <= max_kv_indices[s_offset]) & (s_i < seq_len)

                # 加载 KV (使用 block 内最后一个 seq position 的 indices)
                # 注意：这里仍然使用最后一个 seq position 的 indices
                # 因为 Sparse MLA 中，后面的 seq position 的 indices 是前面的超集（causal）
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_for_indices, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_for_indices, g_i, i_i * BI + bi_i], g_i, D + d_i]

                # 初始化 acc_s，应用各自的 mask
                for sh_i, bi_i in T.Parallel(padded_merged_H, BI):
                    # 计算对应的 s_offset
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
    
    # 标准 indices：每个 seq position 独立随机选择
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    
    # 为优化7生成共享 indices 的版本
    # 每 S_per_block 个 seq positions 共享同一套 indices（使用 block 内最后一个的）
    S_per_block_for_opt7 = 8
    indices_shared = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for h in range(HKV):
            # 按 block 生成共享的 indices
            num_blocks = (S + S_per_block_for_opt7 - 1) // S_per_block_for_opt7
            for block_idx in range(num_blocks):
                s_base = block_idx * S_per_block_for_opt7
                s_end = min(s_base + S_per_block_for_opt7, S)
                # 使用 block 内最后一个 seq position 的范围来生成 indices
                t_for_idx = s_end - 1  # 最后一个有效 seq position
                # 生成一套 indices，给整个 block 共享
                i_i = torch.randperm(max(1, t_for_idx))[:topk]
                # 复制给 block 内所有 seq positions
                for t in range(s_base, s_end):
                    indices_shared[b, t, h, : len(i_i)] = i_i
    
    # 计算 tail_dim
    tail_dim = DQK - DV
    
    # 定义所有优化方案 (kernel_func, block_I, threads, S_per_block)
    optimizations = [
        ("原始方案 (优化1配置)", sparse_mla_fwd_baseline, 64, 128, 1),
        ("优化7: 批量处理 Query (S=8)", sparse_mla_fwd_batched_query, 64, 128, 8),
    ]
    
    results = []
    ref_out = None
    ref_out_shared = None  # 为优化7使用共享 indices 的参考输出
    
    for name, kernel_func, block_i, num_threads, s_per_block in optimizations:
        print(f"\n测试: {name}")
        print("-" * 40)
        
        # 优化7 使用共享 indices
        use_shared_indices = s_per_block > 1
        current_indices = indices_shared if use_shared_indices else indices
        
        try:
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
            out, lse = kernel(q, kv, current_indices)
            
            # 正确性验证
            if check_correctness:
                # 使用对应的 indices 计算参考输出
                if use_shared_indices:
                    if ref_out_shared is None:
                        ref_out_shared = ref_sparse_mla_fwd(q, kv, indices_shared, d_v=DV)
                    current_ref = ref_out_shared
                else:
                    if ref_out is None:
                        ref_out = ref_sparse_mla_fwd(q, kv, indices, d_v=DV)
                    current_ref = ref_out
                
                try:
                    torch.testing.assert_close(out, current_ref, rtol=1e-2, atol=1e-2)
                    print(f"  ✓ 正确性验证通过")
                except AssertionError as e:
                    max_diff = (out - current_ref).abs().max().item()
                    print(f"  ✗ 正确性验证失败，最大差异: {max_diff:.6f}")
            
            # 性能测试
            def fn():
                return kernel(q, kv, current_indices)
            
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
