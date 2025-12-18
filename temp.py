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
# 优化6: Warp Specialization（高级）
# 将线程分为三组：Producer（加载数据）、Consumer1（Q@K+softmax）、Consumer2（P@V）
# ============================================================================
@tilelang.jit(
    out_idx=[-2, -1],
    compile_flags=[
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
    ],
)
def sparse_mla_fwd_warp_specialized(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=1,  # 这里 num_stages 不再使用，因为我们手动实现流水线
    threads=384,   # 384线程: 128 (consumer1) + 128 (consumer2) + 128 (producer)
):
    """
    优化6: Warp Specialization
    - Producer (tx >= 256): 负责加载 KV 数据
    - Consumer 1 (tx < 128): 负责 Q@K + softmax + P@V_left
    - Consumer 2 (128 <= tx < 256): 负责 P@V_right
    
    通过 barrier 协调，实现计算与数据加载的重叠
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
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    REPLICATE_H = head_kv // 64 if head_kv > 64 else 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64
    
    # Producer 线程加载循环次数
    D_HALF_LOAD_ITERS = D // 128
    assert D // 2 == D_HALF_LOAD_ITERS * 64, f"D/2 必须是 64 的倍数, D={D}"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            # Q 相关 shared memory
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            
            # KV 双缓冲
            KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
            
            # O 复用 Q 的 shared memory
            O_shared_l = Q_shared_l
            O_shared_r = Q_shared_r
            
            # KV 有效性标记
            is_kv_valid = T.alloc_shared([BI], "bool", scope="shared")
            
            # Consumer 1 相关
            acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")
            alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            indices_local = T.alloc_local([1], indices_dtype)
            
            # Barriers for synchronization
            bar_q = T.alloc_barrier(arrive_count=384)
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)
            bar_k_0_free = T.alloc_barrier(arrive_count=256)
            bar_k_1_free = T.alloc_barrier(arrive_count=256)
            bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
            bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()

            # 所有线程加载 Q
            T.copy(Q[b_i, s_i, H0:H1, 0 : D // 2], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D // 2 : D], Q_shared_r)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
            T.barrier_arrive(bar_q)

            # ========== Consumer 1: Q@K + softmax + P@V_left ==========
            if tx < 128:
                T.set_max_nreg(240, 1)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))
                T.fill(acc_o_l, 0)
                T.barrier_wait(bar_q, 0)

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)
                    T.wait_wgmma(0)

                    if i_i != 0:
                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)
                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=-1)
                    T.wait_wgmma(0)

                    T.barrier_arrive(bar_sScale_and_sS_free)
                    T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)
                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_1_free[0])

                # Rescale
                for h_i in T.Parallel(H_per_block):
                    sum_exp_shared[h_i] = sumexp[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_l[h_i, d_i] /= sumexp[h_i]
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                T.copy(acc_o_l, O_shared_l)
                T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0 : D // 2])

            # ========== Consumer 2: P@V_right ==========
            elif tx >= 128 and tx < 256:
                T.set_max_nreg(168, 1)
                T.fill(acc_o_r, 0)
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                    T.barrier_arrive(bar_k_0_free[0])
                    T.barrier_arrive(bar_sScale_and_sS_free)

                    # Buffer 1
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                    T.barrier_arrive(bar_k_1_free[0])
                    if i_i != T.ceildiv(NI, 2) - 1:
                        T.barrier_arrive(bar_sScale_and_sS_free)

                # Rescale
                for h_i, d_i in T.Parallel(H_per_block, D // 2):
                    acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                T.copy(acc_o_r, O_shared_r)
                T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2 : D])
                
            # ========== Producer: 加载 KV 数据 ==========
            elif tx >= 256:
                T.set_max_nreg(80, 0)
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2) * BI + r * 16 + (tx - 256) // 8]
                        is_kv_valid[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                        if is_kv_valid[r * 16 + (tx - 256) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(D_HALF_LOAD_ITERS):
                                    for v in T.vectorized(8):
                                        KV_shared_0_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                                            b_i, indices_local[0], g_i, 64 * u + (tx - 256) % 8 * 8 + v
                                        ]
                                        KV_shared_0_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                                            b_i, indices_local[0], g_i, D // 2 + 64 * u + (tx - 256) % 8 * 8 + v
                                        ]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_0[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v] = KV[
                                        b_i, indices_local[0], g_i, D + (tx - 256) % 8 * 8 + v
                                    ]
                    T.cp_async_barrier_noinc(bar_k_0_ready[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8]
                        is_kv_valid[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                        if is_kv_valid[r * 16 + (tx - 256) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(D_HALF_LOAD_ITERS):
                                    for v in T.vectorized(8):
                                        KV_shared_1_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                                            b_i, indices_local[0], g_i, 64 * u + (tx - 256) % 8 * 8 + v
                                        ]
                                        KV_shared_1_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                                            b_i, indices_local[0], g_i, D // 2 + 64 * u + (tx - 256) % 8 * 8 + v
                                        ]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_1[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v] = KV[
                                        b_i, indices_local[0], g_i, D + (tx - 256) % 8 * 8 + v
                                    ]
                    T.cp_async_barrier_noinc(bar_k_1_ready[0])

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
    
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    
    # 计算 tail_dim
    tail_dim = DQK - DV
    
    # 定义所有优化方案
    optimizations = [
        ("原始方案 (优化1配置)", sparse_mla_fwd_baseline, 64, 128),
        ("优化6: Warp Specialization", sparse_mla_fwd_warp_specialized, 64, 384),
    ]
    
    results = []
    ref_out = None
    
    for name, kernel_func, block_i, num_threads in optimizations:
        print(f"\n测试: {name}")
        print("-" * 40)
        
        try:
            # 编译 kernel
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
            
            # 性能测试
            def fn():
                return kernel(q, kv, indices)
            
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
