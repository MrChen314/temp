# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tilelang.engine.callback import register_cuda_postproc_callback
import argparse
from dataclasses import dataclass
from typing import List


@tilelang.jit(
    out_idx=[-3, -2, -1],
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
def indexer_loss_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=0,
    threads=256,
):
    """
    IndexerLoss Forward Kernel with Pipeline Optimization (简化版)
    
    计算 sparse attention 的 m、l 和 qk 矩阵乘结果，用于后续的 attn_sum 计算。
    
    线程分组 (简化设计):
    - 线程1组 (tx < 128): 计算所有 heads 的 m、l 和 qk 矩阵乘结果
    - 线程3组 (tx >= 128): producer，负责加载 K 到 shared memory 双 buffer
    
    输出:
    - M_out: [batch, seq_len, heads] - 每个 head 的 max 值
    - L_out: [batch, seq_len, heads] - 每个 head 的 sum of exp 值
    - QK_out: [batch, seq_len, heads, topk] - 经过 sm_scale 缩放的 QK 结果
    """
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    m_out_shape = [batch, seq_len, heads]
    l_out_shape = [batch, seq_len, heads]
    qk_out_shape = [batch, seq_len, heads, topk]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    assert NI % 2 == 0, "NI should be a multiple of 2"
    D = dim
    D_tail = tail_dim
    KV_stride = kv_stride
    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor(1, indices_dtype),
        M_out: T.Tensor(m_out_shape, accum_dtype),  # type: ignore
        L_out: T.Tensor(l_out_shape, accum_dtype),  # type: ignore
        QK_out: T.Tensor(qk_out_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel((seq_len - kv_stride + 1 if CP0 else seq_len) * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            # Q shared memory - 所有线程共用
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            
            # K shared memory - 双 buffer
            KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
            
            # K validity mask - 双 buffer
            is_kv_valid_0 = T.alloc_shared([BI], "bool", scope="shared")
            is_kv_valid_1 = T.alloc_shared([BI], "bool", scope="shared")

            # 线程1组的局部变量 - 处理所有 heads
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            indices_local = T.alloc_local([1], indices_dtype)

            # Barriers - 简化版
            bar_q = T.alloc_barrier(arrive_count=256)
            # K buffer 同步
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)
            bar_k_0_free = T.alloc_barrier(arrive_count=128)
            bar_k_1_free = T.alloc_barrier(arrive_count=128)

            b_i, g_i = by, bz
            s_i = (bx + (KV_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (bx // REPLICATE_H + (KV_stride - 1 if CP0 else 0))
            q_i = q_start_index_s[0] + s_i
            max_kv_i = (q_i + 1 - KV_stride) // KV_stride

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()

            # 加载 Q 到 shared memory
            T.copy(Q[b_i, s_i, H0:H1, 0 : D // 2], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D // 2 : D], Q_shared_r)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
            T.barrier_arrive(bar_q)

            if tx < 128:
                # ================================================================
                # 线程1组: 计算所有 heads 的 m、l 和 qk 矩阵乘结果
                # ================================================================
                T.set_max_nreg(240, 1)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))
                T.barrier_wait(bar_q, 0)

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], 0, -T.infinity(acc_s.dtype))
                    # QK 矩阵乘 - 处理所有 heads
                    T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    # 写出 qk * sm_scale 结果
                    tk_start_0 = (i_i * 2) * BI
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        QK_out[b_i, s_i, H0 + h_i, tk_start_0 + bi_i] = acc_s[h_i, bi_i] * sm_scale

                    # Online softmax
                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], acc_s[h_i, bi_i], 0)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale) + sumexp_i[h_i]

                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    # 写出 qk * sm_scale 结果
                    tk_start_1 = (i_i * 2 + 1) * BI
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        QK_out[b_i, s_i, H0 + h_i, tk_start_1 + bi_i] = acc_s[h_i, bi_i] * sm_scale

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], acc_s[h_i, bi_i], 0)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale) + sumexp_i[h_i]

                    T.barrier_arrive(bar_k_1_free[0])

                # 写出最终的 m 和 l
                for h_i in T.Parallel(H_per_block):
                    M_out[b_i, s_i, H0 + h_i] = m_i[h_i]
                    L_out[b_i, s_i, H0 + h_i] = sumexp[h_i]

            elif tx >= 128:
                # ================================================================
                # 线程3组: Producer - 负责加载 K 到 shared memory
                # ================================================================
                T.set_max_nreg(80, 0)
                
                # 加载 K 给线程1组使用
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2) * BI + r * 16 + (tx - 128) // 8]
                        is_kv_valid_0[r * 16 + (tx - 128) // 8] = indices_local[0] <= max_kv_i
                        if is_kv_valid_0[r * 16 + (tx - 128) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        KV_shared_0_l[r * 16 + (tx - 128) // 8, 64 * u + (tx - 128) % 8 * 8 + v] = KV[
                                            b_i, indices_local[0], g_i, 64 * u + (tx - 128) % 8 * 8 + v
                                        ]
                                        KV_shared_0_r[r * 16 + (tx - 128) // 8, 64 * u + (tx - 128) % 8 * 8 + v] = KV[
                                            b_i, indices_local[0], g_i, D // 2 + 64 * u + (tx - 128) % 8 * 8 + v
                                        ]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_0[r * 16 + (tx - 128) // 8, (tx - 128) % 8 * 8 + v] = KV[
                                        b_i, indices_local[0], g_i, D + (tx - 128) % 8 * 8 + v
                                    ]
                    T.cp_async_barrier_noinc(bar_k_0_ready[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2 + 1) * BI + r * 16 + (tx - 128) // 8]
                        is_kv_valid_1[r * 16 + (tx - 128) // 8] = indices_local[0] <= max_kv_i
                        if is_kv_valid_1[r * 16 + (tx - 128) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        KV_shared_1_l[r * 16 + (tx - 128) // 8, 64 * u + (tx - 128) % 8 * 8 + v] = KV[
                                            b_i, indices_local[0], g_i, 64 * u + (tx - 128) % 8 * 8 + v
                                        ]
                                        KV_shared_1_r[r * 16 + (tx - 128) // 8, 64 * u + (tx - 128) % 8 * 8 + v] = KV[
                                            b_i, indices_local[0], g_i, D // 2 + 64 * u + (tx - 128) % 8 * 8 + v
                                        ]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_1[r * 16 + (tx - 128) // 8, (tx - 128) % 8 * 8 + v] = KV[
                                        b_i, indices_local[0], g_i, D + (tx - 128) % 8 * 8 + v
                                    ]
                    T.cp_async_barrier_noinc(bar_k_1_ready[0])

    return main

@tilelang.jit(out_idx=[-1])
def attn_sum_fwd(
    batch,
    seq_len,
    heads,
    topk,
    sm_scale_log2e,
    block_T=64,
    threads=128,
):
    """
    Attention Sum Forward Kernel
    
    计算 softmax 概率并沿 head 维度求和。
    
    输入:
    - QK_out: [batch, seq_len, heads, topk] - 已乘以 sm_scale 的 QK 结果
    - M_out: [batch, seq_len, heads] - 每个 head 的 max 值
    - L_out: [batch, seq_len, heads] - 每个 head 的 sum of exp2
    
    输出:
    - AttnSum: [batch, seq_len, topk] - 沿 head 维度求和的 attention 概率
    """
    accum_dtype = T.float32
    
    qk_shape = [batch, seq_len, heads, topk]
    m_shape = [batch, seq_len, heads]
    l_shape = [batch, seq_len, heads]
    attn_sum_shape = [batch, seq_len, topk]
    
    NT = tilelang.cdiv(topk, block_T)
    
    @T.prim_func
    def main(
        QK_out: T.Tensor(qk_shape, accum_dtype),  # type: ignore
        M_out: T.Tensor(m_shape, accum_dtype),  # type: ignore
        L_out: T.Tensor(l_shape, accum_dtype),  # type: ignore
        AttnSum: T.Tensor(attn_sum_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, batch, threads=threads) as (bx, by):
            # 分配本地内存
            qk_local = T.alloc_fragment([heads, block_T], accum_dtype)
            m_local = T.alloc_fragment([heads], accum_dtype)
            l_local = T.alloc_fragment([heads], accum_dtype)
            attn_sum_local = T.alloc_fragment([block_T], accum_dtype)
            
            # 加载 M 和 L (每个 seq position 只需加载一次)
            T.copy(M_out[by, bx, 0:heads], m_local)
            T.copy(L_out[by, bx, 0:heads], l_local)
            
            # 对 topk 分块处理
            for t_i in T.serial(NT):
                # 加载 QK block
                T.copy(QK_out[by, bx, 0:heads, t_i * block_T : (t_i + 1) * block_T], qk_local)
                
                # 计算 attn_prob = exp2(qk - m * sm_scale) / l
                for h_i, t_j in T.Parallel(heads, block_T):
                    qk_local[h_i, t_j] = T.exp2(qk_local[h_i, t_j] - m_local[h_i] * sm_scale_log2e) / l_local[h_i]
                
                # 沿 head 维度求和
                T.reduce_sum(qk_local, attn_sum_local, dim=0)
                
                # 写出结果
                T.copy(attn_sum_local, AttnSum[by, bx, t_i * block_T : (t_i + 1) * block_T])
    
    return main


@tilelang.jit(out_idx=[-1])
def indexer_loss_bwd(
    batch,
    chunk_size,
    topk,
    chunk_offset,
    kv_stride=1,
    eps=1e-10,
    block_T=64,
    threads=128,
):
    """
    IndexerLoss Backward Kernel
    
    计算 index_score 的梯度: grad = index_prob - attn_dist
    
    数学推导：
    Loss = KL(attn_dist || index_prob) = sum_k attn_dist_k * log(attn_dist_k / index_prob_k)
    
    由于 attn_dist 不依赖于 index_score（只依赖于 Q, K），所以：
    d Loss / d index_score_j = index_prob_j - attn_dist_j
    
    输入:
    - IndexScore: [batch, chunk_size, topk] - index 分数
    - Indices: [batch, chunk_size, topk] - 索引 (用于 causal mask)
    - AttnSum: [batch, chunk_size, topk] - 前向保存的 attn_sum
    
    输出:
    - dIndexScore: [batch, chunk_size, topk] - 梯度
    """
    accum_dtype = T.float32
    indices_dtype = T.int32
    
    is_shape = [batch, chunk_size, topk]
    indices_shape = [batch, chunk_size, topk]
    attn_sum_shape = [batch, chunk_size, topk]
    grad_shape = [batch, chunk_size, topk]
    
    NT = tilelang.cdiv(topk, block_T)
    NEG_INF = -(2**30)
    KV_stride = kv_stride
    
    @T.prim_func
    def main(
        IndexScore: T.Tensor(is_shape, accum_dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        AttnSum: T.Tensor(attn_sum_shape, accum_dtype),  # type: ignore
        dIndexScore: T.Tensor(grad_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(chunk_size, batch, threads=threads) as (bx, by):
            # 局部变量
            is_local = T.alloc_fragment([block_T], accum_dtype)
            indices_local = T.alloc_fragment([block_T], indices_dtype)
            attn_sum_local = T.alloc_fragment([block_T], accum_dtype)
            grad_local = T.alloc_fragment([block_T], accum_dtype)
            
            s_i = bx
            global_query_pos = chunk_offset + s_i
            max_kv_i = (global_query_pos + 1 - KV_stride) // KV_stride
            
            # =========================================================================
            # Step 1: 计算 attn_total (attn_sum 的总和，用于归一化成 attn_dist)
            # =========================================================================
            attn_total = T.alloc_fragment([1], accum_dtype)
            T.fill(attn_total, 0)
            
            for t_i in T.serial(NT):
                T.copy(AttnSum[by, bx, t_i * block_T : (t_i + 1) * block_T], attn_sum_local)
                for t_j in T.Parallel(block_T):
                    attn_total[0] = attn_total[0] + attn_sum_local[t_j]
            
            # 防止除零
            for i in T.Parallel(1):
                attn_total[i] = T.if_then_else(attn_total[i] < eps, 1.0, attn_total[i])
            
            # =========================================================================
            # Step 2: 对 index_score 做 Online Softmax 得到 index_prob
            # =========================================================================
            is_m_global = T.alloc_fragment([1], accum_dtype)
            is_l_global = T.alloc_fragment([1], accum_dtype)
            T.fill(is_m_global, NEG_INF)
            T.fill(is_l_global, 0)
            
            for t_i in T.serial(NT):
                T.copy(IndexScore[by, bx, t_i * block_T : (t_i + 1) * block_T], is_local)
                T.copy(Indices[by, bx, t_i * block_T : (t_i + 1) * block_T], indices_local)
                
                # 应用 causal mask
                for t_j in T.Parallel(block_T):
                    is_local[t_j] = T.if_then_else(indices_local[t_j] > max_kv_i, NEG_INF, is_local[t_j])
                
                # 找当前块的 max
                is_m_block = T.alloc_fragment([1], accum_dtype)
                T.fill(is_m_block, NEG_INF)
                for t_j in T.Parallel(block_T):
                    is_m_block[0] = T.max(is_m_block[0], is_local[t_j])
                
                # 更新全局 max 和 sum
                for i in T.Parallel(1):
                    is_m_new = T.max(is_m_global[i], is_m_block[i])
                    # 计算 exp 并累加
                    is_l_global[i] = is_l_global[i] * T.exp(is_m_global[i] - is_m_new)
                    is_m_global[i] = is_m_new
                
                for t_j in T.Parallel(block_T):
                    exp_val = T.exp(is_local[t_j] - is_m_global[0])
                    exp_val = T.if_then_else(indices_local[t_j] > max_kv_i, 0.0, exp_val)
                    is_l_global[0] = is_l_global[0] + exp_val
            
            # 防止除零
            for i in T.Parallel(1):
                is_m_global[i] = T.if_then_else(is_m_global[i] == NEG_INF, 0.0, is_m_global[i])
                is_l_global[i] = T.if_then_else(is_l_global[i] < 1e-9, 1.0, is_l_global[i])
            
            # =========================================================================
            # Step 3: 计算梯度 grad = index_prob - attn_dist
            # =========================================================================
            for t_i in T.serial(NT):
                T.copy(IndexScore[by, bx, t_i * block_T : (t_i + 1) * block_T], is_local)
                T.copy(Indices[by, bx, t_i * block_T : (t_i + 1) * block_T], indices_local)
                T.copy(AttnSum[by, bx, t_i * block_T : (t_i + 1) * block_T], attn_sum_local)
                
                for t_j in T.Parallel(block_T):
                    # 应用 causal mask
                    is_val = T.if_then_else(indices_local[t_j] > max_kv_i, NEG_INF, is_local[t_j])
                    # 计算 index_prob
                    index_prob = T.exp(is_val - is_m_global[0]) / is_l_global[0]
                    # 计算 attn_dist
                    attn_dist = attn_sum_local[t_j] / attn_total[0]
                    # 梯度 = index_prob - attn_dist
                    grad = index_prob - attn_dist
                    grad_local[t_j] = T.if_then_else(indices_local[t_j] > max_kv_i, 0.0, grad)
                
                # 写出梯度
                T.copy(grad_local, dIndexScore[by, bx, t_i * block_T : (t_i + 1) * block_T])
    
    return main


class IndexerLossFunctionOpt(torch.autograd.Function):
    """
    自定义 autograd Function，实现稀疏注意力索引损失的前向和反向传播 (TileLang 版本)
    
    前向传播:
        1. 计算 sparse attention (Q @ K[indices]) 并应用 online softmax
        2. 将所有 head 的 attention 累加到 attn_sum
        3. 返回 dummy loss (实际 loss 计算在反向传播中不需要)
    
    反向传播:
        由于 attn_dist 不依赖于 index_score（只依赖于 Q, K），所以：
        grad_index_score = index_prob - attn_dist
    """
    
    @staticmethod
    def forward(ctx, q, kv, index_score, indices, chunk_offset=0, kv_stride=1, sm_scale=None, eps=1e-10):
        """
        前向传播 - 计算 attn_sum
        
        Args:
            q: [batch, chunk_size, heads, dim + tail_dim]
            kv: [batch, seq_len_kv, kv_group, dim + tail_dim]
            index_score: [batch, chunk_size, topk] - 稀疏版本的 index 分数
            indices: [batch, chunk_size, kv_group, topk] - 每个 query 选择的 topk 个 key 索引
            chunk_offset: 当前 chunk 在完整序列中的起始位置 (用于 causal mask)
            kv_stride: KV stride
            sm_scale: attention scaling factor
            eps: 数值稳定 epsilon
        
        Returns:
            loss: dummy loss 值 (供 loss.backward() 调用)
        """
        batch, chunk_size, heads, dim_plus_tail_dim = q.shape
        _, seq_len_kv, kv_group, _ = kv.shape
        topk = indices.shape[-1]
        
        # 计算 attn_sum
        attn_sum = indexer_loss_fwd_interface(q, kv, indices, chunk_offset, kv_stride, sm_scale)
        
        # 返回 dummy loss (用于触发 backward)
        dummy_loss = torch.tensor(0.0, device=q.device, dtype=torch.float32, requires_grad=True)
        
        # 保存反向传播需要的张量
        # indices 需要转换为 [batch, chunk_size, topk] 格式
        indices_2d = indices[:, :, 0, :]  # 假设 kv_group=1
        ctx.save_for_backward(index_score, indices_2d, attn_sum)
        ctx.chunk_offset = chunk_offset
        ctx.kv_stride = kv_stride
        ctx.eps = eps
        ctx.batch_size = batch
        
        return dummy_loss
    
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
        kv_stride = ctx.kv_stride
        eps = ctx.eps
        batch_size = ctx.batch_size
        
        batch, chunk_size, topk = index_score.shape
        
        # 编译反向 kernel
        bwd_kernel = indexer_loss_bwd(batch, chunk_size, topk, chunk_offset, kv_stride, eps)
        
        # 运行反向 kernel
        grad_index_score = bwd_kernel(index_score, indices, attn_sum)
        
        # 乘以上游梯度并除以 batch_size (对应前向 loss = sum / batch_size)
        grad_index_score = grad_index_score * grad_output / batch_size
        
        # 返回对应输入的梯度
        # (q, kv, index_score, indices, chunk_offset, kv_stride, sm_scale, eps)
        return None, None, grad_index_score, None, None, None, None, None


def indexer_loss_opt(q, kv, index_score, indices, chunk_offset=0, kv_stride=1, sm_scale=None, eps=1e-10):
    """
    计算稀疏注意力索引损失 (Sparse Attention Index Loss) - TileLang 版本
    
    该函数使用 TileLang kernel 实现高效的稀疏注意力计算，支持自动梯度反向传播。
    
    核心思想:
        1. 计算 sparse attention: softmax(Q @ K[indices] * scaling)
        2. 将所有 head 的 attention 求和得到 attn_sum
        3. 反向传播时计算 grad = index_prob - attn_dist
    
    Args:
        q: [batch, chunk_size, heads, dim + tail_dim] - 当前 chunk 的 query
        kv: [batch, seq_len_kv, kv_group, dim + tail_dim] - 完整的 KV
        index_score: [batch, chunk_size, topk] - 稀疏版本的 index 分数，需要梯度
        indices: [batch, chunk_size, kv_group, topk] - 每个 query 选择的 topk 个 key 索引
        chunk_offset: 当前 chunk 在完整序列中的起始位置 (用于 causal mask)，默认为 0
        kv_stride: KV stride，默认为 1
        sm_scale: attention scaling factor
        eps: 数值稳定 epsilon，默认为 1e-10
    
    Returns:
        loss: dummy loss 值 (供 loss.backward() 调用)
    
    Example:
        >>> batch, heads, chunk_size, head_dim = 1, 128, 256, 576
        >>> seq_len, topk = 1024, 128
        >>> q = torch.randn(batch, chunk_size, heads, head_dim, device='cuda', dtype=torch.bfloat16)
        >>> kv = torch.randn(batch, seq_len, 1, head_dim, device='cuda', dtype=torch.bfloat16)
        >>> index_score = torch.randn(batch, chunk_size, topk, device='cuda', requires_grad=True)
        >>> indices = torch.randint(0, seq_len, (batch, chunk_size, 1, topk), device='cuda', dtype=torch.int32)
        >>> loss = indexer_loss_opt(q, kv, index_score, indices, chunk_offset=seq_len - chunk_size)
        >>> loss.backward()  # index_score.grad 现在包含梯度
    """
    return IndexerLossFunctionOpt.apply(q, kv, index_score, indices, chunk_offset, kv_stride, sm_scale, eps)


def indexer_loss_fwd_interface(
    q, kv, indices, chunk_offset=0, kv_stride=1, sm_scale=None, is_casual=True, return_kernel=False, print_kernel=False
):
    """
    IndexerLoss Forward Interface
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        kv: [batch, seq_len_kv, kv_group, dim + tail_dim]
        indices: [batch, seq_len, kv_group, topk]
        chunk_offset: 当前 chunk 在完整序列中的起始位置 (用于 causal mask)
        kv_stride: KV stride
        sm_scale: attention scaling factor
        is_casual: whether to use causal mask
        return_kernel: whether to return compiled kernel
        print_kernel: whether to print kernel source
    
    Returns:
        attn_sum: [batch, seq_len, topk] - 所有 head 的 attention 概率求和
    """
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = 512

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    if chunk_offset != 0:
        assert chunk_offset > kv_stride, (
            "If it is because each cp has too short length, you should fix the logic involving CP0 (cp_rank == 0), to make sure q with pos < KV_Stride - 1 is masked (or you may just ignore how this is handled if nan in these q's Out would not effect others, which is reported to be likely to happen by wangding)"
        )
    CP0 = chunk_offset == 0

    # 计算 kernel 中使用的 sm_scale（包含 log2(e) 因子）
    if sm_scale is None:
        sm_scale_log2e = (1.0 / dim_plus_tail_dim) ** 0.5 * 1.44269504
    else:
        sm_scale_log2e = sm_scale * 1.44269504

    kernel = indexer_loss_fwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, kv_group, sm_scale, is_casual, CP0)
    if print_kernel:
        print(kernel.get_kernel_source())
    
    # kernel 返回 M_out, L_out, QK_out
    m_out, l_out, qk_out = kernel(q, kv, indices, torch.tensor([chunk_offset], dtype=torch.int32, device="cuda"))
    
    # 使用 tilelang kernel 计算 attn_sum
    attn_sum_kernel = attn_sum_fwd(batch, seq_len, heads, topk, sm_scale_log2e)
    if print_kernel:
        print(attn_sum_kernel.get_kernel_source())
    attn_sum = attn_sum_kernel(qk_out, m_out, l_out)
    
    if return_kernel:
        return (kernel, attn_sum_kernel), attn_sum
    if chunk_offset == 0 and kv_stride > 1:
        attn_sum[:, : kv_stride - 1, :] = 0
    return attn_sum


def ref_indexer_loss_fwd_interface(q, kv, indices, chunk_offset=0, kv_stride=1, sm_scale=None, is_casual=True):
    """
    PyTorch Reference Implementation for IndexerLoss Forward
    
    计算 sparse attention 的累加分布 AttnSum。
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        kv: [batch, seq_len_kv, kv_group, dim + tail_dim]
        indices: [batch, seq_len, kv_group, topk]
        chunk_offset: 当前 chunk 在完整序列中的起始位置 (用于 causal mask)
        kv_stride: KV stride
        sm_scale: attention scaling factor
        is_casual: whether to use causal mask
    
    Returns:
        attn_sum: [batch, seq_len, topk] - 所有 head 的 attention 概率求和
    """
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)  # [batch, kv_group, seq_len, topk]
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape
    if chunk_offset is None:
        chunk_offset = sk * kv_stride - sq

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv  # [batch, seq_len_kv, kv_group, dim + tail_dim]

    g_index = g
    h_index = h // g
    
    # Causal mask for compressed KV
    compressed_casual_mask = torch.arange(chunk_offset, sq + chunk_offset, dtype=torch.int32, device="cuda").view(
        -1, 1
    ) >= torch.arange(kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32, device="cuda").view(1, -1)

    # Sparse mask based on indices
    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)  # [b, g, 1, sq, sk]

    q = q.view(b, sq, g, -1, dim_q)  # [b, sq, g, h_per_g, dim_q]
    
    # Compute attention scores
    score = torch.einsum("bmghd,bngd->bghmn", q, k)  # [b, g, h_per_g, sq, sk]
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)  # [b, g, h_per_g, sq, sk]
    
    # Sum over all heads: [b, g, h_per_g, sq, sk] -> [b, sq, sk]
    p_sum = p.sum(dim=(1, 2))  # [b, sq, sk]
    
    # Gather to get attn_sum for topk indices
    # indices: [batch, kv_group, seq_len, topk]
    # 需要将 p_sum 按 indices gather
    indices_for_gather = indices[:, 0, :, :]  # [batch, seq_len, topk] (假设 kv_group=1)
    attn_sum = torch.gather(p_sum, dim=-1, index=indices_for_gather.long())  # [batch, seq_len, topk]
    
    return attn_sum


def test_indexer_loss_fwd_pipelined(
    B=1, S=4096, SKV=8192, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16, chunk_offset=1024, check_correctness=True
):
    """
    Test IndexerLoss Forward Kernel
    """
    KV_stride = 1

    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(False) / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(False) / 10
    chunk_offset_t = torch.tensor([chunk_offset], dtype=torch.int32, device="cuda")

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(min(max(1, ((t + chunk_offset) // KV_stride)), SKV))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    kernels, tl_attn_sum = indexer_loss_fwd_interface(q, kv, indices, chunk_offset, KV_stride, return_kernel=True, print_kernel=True)
    indexer_kernel, attn_sum_kernel = kernels

    def fn():
        # indexer_kernel 返回 m_out, l_out, qk_out
        m_out, l_out, qk_out = indexer_kernel(q, kv, indices, chunk_offset_t)
        # 使用 attn_sum_kernel 计算 attn_sum
        attn_sum = attn_sum_kernel(qk_out, m_out, l_out)
        if chunk_offset == 0 and KV_stride > 1:
            attn_sum[:, : KV_stride - 1, :] = 0
        return attn_sum

    if check_correctness:
        ref_attn_sum = ref_indexer_loss_fwd_interface(q, kv, indices, chunk_offset, KV_stride)
        print(f"tl_attn_sum shape: {tl_attn_sum.shape}")
        print(f"ref_attn_sum shape: {ref_attn_sum.shape}")
        print(f"tl_attn_sum sample: {tl_attn_sum[0, 0, :10]}")
        print(f"ref_attn_sum sample: {ref_attn_sum[0, 0, :10]}")
        
        try:
            torch.testing.assert_close(tl_attn_sum, ref_attn_sum.to(tl_attn_sum.dtype), rtol=1e-2, atol=1e-2)
            print("Correctness test PASSED!")
        except AssertionError as e:
            print(f"Correctness test FAILED: {e}")
            max_diff = (tl_attn_sum - ref_attn_sum.to(tl_attn_sum.dtype)).abs().max()
            print(f"Max difference: {max_diff}")

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=10,
        warmup=10,
    )
    print(f"Average time: {ms:.3f} ms")
    print(f"fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print(f"fwd tflops = ", (B * S * DQK * topk * 2 * H) / (ms * 1e-3) / 1e12)


# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class TestConfig:
    """测试配置"""
    name: str
    batch_size: int = 1
    num_heads: int = 128
    chunk_size: int = 256
    seq_len: int = 256
    head_dim: int = 576  # tilelang 版本固定为 576
    topk: int = 128
    seed: int = 42
    
    def __str__(self):
        return (f"batch={self.batch_size}, heads={self.num_heads}, "
                f"chunk={self.chunk_size}, seq={self.seq_len}, "
                f"dim={self.head_dim}, topk={self.topk}")


# ============================================================================
# 前向精度测试 (比较 attn_sum)
# ============================================================================

def run_fwd_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """运行单个前向精度测试 - 比较 attn_sum"""
    torch.manual_seed(config.seed)
    
    # tilelang 版本使用固定的 dim=576
    head_dim = 576
    
    # 生成随机输入
    q = torch.randn(config.batch_size, config.chunk_size, config.num_heads, head_dim, 
                    device=device, dtype=torch.bfloat16) / 10
    kv = torch.randn(config.batch_size, config.seq_len, 1, head_dim, 
                     device=device, dtype=torch.bfloat16) / 10
    
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    
    # 计算 chunk_offset (类似于 triton 版本中的 chunk_offset = seq_len - chunk_size)
    chunk_offset = config.seq_len - config.chunk_size
    kv_stride = 1
    
    # 生成 indices
    indices = torch.full((config.batch_size, config.chunk_size, 1, config.topk), 
                         config.seq_len, dtype=torch.int32, device=device)
    for b in range(config.batch_size):
        for t in range(config.chunk_size):
            max_valid_idx = min(max(1, ((t + chunk_offset) // kv_stride)), config.seq_len)
            i_i = torch.randperm(max_valid_idx, device=device)[:config.topk]
            indices[b, t, 0, : len(i_i)] = i_i.to(torch.int32)
    
    # TileLang kernel: 计算 attn_sum
    tl_attn_sum = indexer_loss_fwd_interface(q, kv, indices, chunk_offset, kv_stride)
    
    # PyTorch 参考: 计算 attn_sum
    ref_attn_sum = ref_indexer_loss_fwd_interface(q, kv, indices, chunk_offset, kv_stride)
    
    # 比较 attn_sum
    abs_diff = (ref_attn_sum - tl_attn_sum.to(ref_attn_sum.dtype)).abs().max().item()
    rel_diff = abs_diff / (ref_attn_sum.abs().max().item() + 1e-10)
    passed = rel_diff < 1e-2  # 1% 相对误差
    
    return {
        'config': config,
        'ref_max': ref_attn_sum.abs().max().item(),
        'tl_max': tl_attn_sum.abs().max().item(),
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'passed': passed
    }


def test_fwd_accuracy(configs: List[TestConfig]):
    """批量运行前向精度测试 - 比较 attn_sum"""
    print("\n" + "=" * 110)
    print("前向精度测试 (PyTorch attn_sum vs TileLang attn_sum)")
    print("=" * 110)
    
    results = []
    for config in configs:
        try:
            result = run_fwd_accuracy_test(config)
            results.append(result)
        except Exception as e:
            print(f"跳过测试 {config.name}: {e}")
            continue
    
    print(f"\n{'Name':<12} {'Config':<60} {'RefMax':<12} {'TLMax':<12} {'RelDiff':<12} {'Pass':<6}")
    print("-" * 114)
    for r in results:
        print(f"{r['config'].name:<12} {str(r['config']):<60} "
              f"{r['ref_max']:<12.6f} {r['tl_max']:<12.6f} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 114)
    print(f"前向测试 (attn_sum): {passed_count}/{len(results)} 通过")
    
    return results


# ============================================================================
# 反向精度测试
# ============================================================================

def pytorch_reference_bwd(q, kv, index_score_full, indices, chunk_offset, kv_stride=1, sm_scale=None):
    """
    PyTorch 参考实现: 使用 autograd 计算反向传播
    
    Args:
        q: [batch, chunk_size, heads, dim + tail_dim]
        kv: [batch, seq_len_kv, kv_group, dim + tail_dim]
        index_score_full: [batch, chunk_size, topk] - 需要梯度
        indices: [batch, chunk_size, kv_group, topk]
        chunk_offset: chunk 偏移
        kv_stride: KV stride
        sm_scale: attention scaling factor
    
    Returns:
        grad_index_score: [batch, chunk_size, topk] - 梯度
    """
    if not index_score_full.requires_grad:
        index_score_full = index_score_full.detach().clone().requires_grad_(True)
    
    eps = 1e-10
    
    # 计算 attn_sum
    attn_sum = ref_indexer_loss_fwd_interface(q, kv, indices, chunk_offset, kv_stride, sm_scale)
    
    # 计算 attn_dist
    attn_total = attn_sum.sum(dim=-1, keepdim=True) + eps
    attn_dist = attn_sum / attn_total
    
    # 计算 index_prob (对 index_score 做 softmax)
    batch, chunk_size, topk = index_score_full.shape
    indices_2d = indices[:, :, 0, :]  # [batch, chunk_size, topk]
    
    # 创建 causal mask
    query_positions = chunk_offset + torch.arange(chunk_size, device=q.device).view(-1, 1)
    max_kv_i = (query_positions + 1 - kv_stride) // kv_stride  # [chunk_size, 1]
    causal_mask = indices_2d > max_kv_i.unsqueeze(0)  # [batch, chunk_size, topk]
    
    index_score_masked = index_score_full.masked_fill(causal_mask, float('-inf'))
    index_prob = torch.softmax(index_score_masked, dim=-1)
    
    # KL 散度的梯度
    # Loss = sum_k attn_dist_k * log(attn_dist_k / index_prob_k)
    # dL/d(index_score_j) = index_prob_j - attn_dist_j
    grad = index_prob - attn_dist.to(index_prob.dtype)
    grad = grad.masked_fill(causal_mask, 0.0)
    
    return grad


def run_bwd_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """运行单个反向精度测试"""
    torch.manual_seed(config.seed)
    
    head_dim = 576
    
    # 生成随机输入
    q = torch.randn(config.batch_size, config.chunk_size, config.num_heads, head_dim, 
                    device=device, dtype=torch.bfloat16) / 10
    kv = torch.randn(config.batch_size, config.seq_len, 1, head_dim, 
                     device=device, dtype=torch.bfloat16) / 10
    
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    
    chunk_offset = config.seq_len - config.chunk_size
    kv_stride = 1
    
    # 生成 indices
    indices = torch.full((config.batch_size, config.chunk_size, 1, config.topk), 
                         config.seq_len, dtype=torch.int32, device=device)
    for b in range(config.batch_size):
        for t in range(config.chunk_size):
            max_valid_idx = min(max(1, ((t + chunk_offset) // kv_stride)), config.seq_len)
            i_i = torch.randperm(max_valid_idx, device=device)[:config.topk]
            indices[b, t, 0, : len(i_i)] = i_i.to(torch.int32)
    
    # 生成 index_score
    index_score = torch.randn(config.batch_size, config.chunk_size, config.topk, 
                              device=device, dtype=torch.float32)
    
    # PyTorch 参考: 计算梯度
    ref_grad = pytorch_reference_bwd(q, kv, index_score.clone(), indices, chunk_offset, kv_stride)
    
    # TileLang 版本: 使用自定义 backward
    index_score_tl = index_score.clone().requires_grad_(True)
    loss = indexer_loss_opt(q, kv, index_score_tl, indices, chunk_offset, kv_stride)
    loss.backward()
    tl_grad = index_score_tl.grad
    
    # 比较梯度
    abs_diff = (ref_grad - tl_grad).abs().max().item()
    rel_diff = abs_diff / (ref_grad.abs().max().item() + 1e-10)
    passed = rel_diff < 1e-2  # 1% 相对误差
    
    return {
        'config': config,
        'ref_grad_max': ref_grad.abs().max().item(),
        'tl_grad_max': tl_grad.abs().max().item(),
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
        'passed': passed
    }


def test_bwd_accuracy(configs: List[TestConfig]):
    """批量运行反向精度测试"""
    print("\n" + "=" * 110)
    print("反向精度测试 (PyTorch autograd vs TileLang kernel)")
    print("=" * 110)
    
    results = []
    for config in configs:
        try:
            result = run_bwd_accuracy_test(config)
            results.append(result)
        except Exception as e:
            print(f"跳过测试 {config.name}: {e}")
            continue
    
    print(f"\n{'Name':<12} {'Config':<60} {'RefMax':<12} {'TLMax':<12} {'RelDiff':<12} {'Pass':<6}")
    print("-" * 114)
    for r in results:
        print(f"{r['config'].name:<12} {str(r['config']):<60} "
              f"{r['ref_grad_max']:<12.2e} {r['tl_grad_max']:<12.2e} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 114)
    print(f"反向测试: {passed_count}/{len(results)} 通过")
    
    return results


# ============================================================================
# 前向+反向性能对比测试
# ============================================================================

def test_fwd_bwd_performance(
    batch_size: int = 1,
    num_heads: int = 128,
    chunk_size: int = 1024,
    seq_len: int = 2048,
    head_dim: int = 576,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 5,
    num_benchmark: int = 20,
):
    """前向+反向性能测试"""
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    
    print("\n" + "=" * 80)
    print("前向+反向性能测试 (TileLang)")
    print("=" * 80)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print("=" * 80)
    
    q = torch.randn(batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.bfloat16) / 10
    kv = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16) / 10
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    
    chunk_offset = seq_len - chunk_size
    kv_stride = 1
    
    # 生成 indices
    indices = torch.full((batch_size, chunk_size, 1, topk), seq_len, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for t in range(chunk_size):
            max_valid_idx = min(max(1, ((t + chunk_offset) // kv_stride)), seq_len)
            i_i = torch.randperm(max_valid_idx, device=device)[:topk]
            indices[b, t, 0, : len(i_i)] = i_i.to(torch.int32)
    
    index_score = torch.randn(batch_size, chunk_size, topk, device=device, dtype=torch.float32)
    
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated() / (1024**3)
    
    # 仅前向测试
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = indexer_loss_fwd_interface(q, kv, indices, chunk_offset, kv_stride)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        _ = indexer_loss_fwd_interface(q, kv, indices, chunk_offset, kv_stride)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start) / num_benchmark * 1000
    fwd_peak = torch.cuda.max_memory_allocated() / (1024**3)
    
    # 前向+反向测试
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        index_score_copy = index_score.clone().requires_grad_(True)
        loss = indexer_loss_opt(q, kv, index_score_copy, indices, chunk_offset, kv_stride)
        loss.backward()
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        index_score_copy = index_score.clone().requires_grad_(True)
        loss = indexer_loss_opt(q, kv, index_score_copy, indices, chunk_offset, kv_stride)
        loss.backward()
    torch.cuda.synchronize()
    fwd_bwd_time = (time.time() - start) / num_benchmark * 1000
    fwd_bwd_peak = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"\n>>> 性能 (warmup={num_warmup}, iters={num_benchmark})")
    print(f"  仅前向:     {fwd_time:.3f} ms")
    print(f"  前向+反向:  {fwd_bwd_time:.3f} ms")
    print(f"  反向开销:   {fwd_bwd_time - fwd_time:.3f} ms")
    
    print(f"\n>>> 显存峰值")
    print(f"  基准显存:   {base_memory:.2f} GB")
    print(f"  前向峰值:   {fwd_peak:.2f} GB (增量: {fwd_peak - base_memory:.2f} GB)")
    print(f"  前向+反向峰值: {fwd_bwd_peak:.2f} GB (增量: {fwd_bwd_peak - base_memory:.2f} GB)")
    
    # 计算 TFLOPS
    flops_fwd = batch_size * chunk_size * head_dim * topk * 2 * num_heads
    print(f"\n>>> 计算吞吐量")
    print(f"  前向 TFLOPS: {flops_fwd / (fwd_time * 1e-3) / 1e12:.2f}")
    print(f"  前向+反向 TFLOPS: {flops_fwd * 2 / (fwd_bwd_time * 1e-3) / 1e12:.2f}")
    
    return {
        'fwd_time': fwd_time,
        'fwd_bwd_time': fwd_bwd_time,
        'fwd_peak': fwd_peak,
        'fwd_bwd_peak': fwd_bwd_peak,
    }


# ============================================================================
# 主测试入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_correctness", action="store_true")
    parser.add_argument("--test_accuracy", action="store_true", help="运行前向精度测试")
    parser.add_argument("--test_bwd_accuracy", action="store_true", help="运行反向精度测试")
    parser.add_argument("--test_performance", action="store_true", help="运行前向+反向性能测试")
    args = parser.parse_args()
    
    # 精度测试配置 (tilelang 版本使用固定的 dim=576, topk 需要是 64 的倍数)
    accuracy_configs = [
        TestConfig(name="小规模", batch_size=1, num_heads=16, chunk_size=256, seq_len=512, head_dim=576, topk=128),
        TestConfig(name="中等规模", batch_size=1, num_heads=64, chunk_size=512, seq_len=1024, head_dim=576, topk=256),
        TestConfig(name="大规模", batch_size=1, num_heads=128, chunk_size=1024, seq_len=2048, head_dim=576, topk=512),
        TestConfig(name="多batch", batch_size=2, num_heads=64, chunk_size=256, seq_len=512, head_dim=576, topk=128),
        TestConfig(name="大topk", batch_size=1, num_heads=128, chunk_size=512, seq_len=2048, head_dim=576, topk=1024),
    ]
    
    if args.test_accuracy:
        # ========== 前向精度测试 ==========
        test_fwd_accuracy(accuracy_configs)
    elif args.test_bwd_accuracy:
        # ========== 反向精度测试 ==========
        test_bwd_accuracy(accuracy_configs)
    elif args.test_performance:
        # ========== 前向+反向性能测试 ==========
        test_fwd_bwd_performance(
            batch_size=1,
            num_heads=128,
            chunk_size=1024,
            seq_len=2048,
            head_dim=576,
            topk=512,
            num_warmup=3,
            num_benchmark=10,
        )
    elif args.test_correctness:
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 1, 1024, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
        test_indexer_loss_fwd_pipelined(B, S, SKV, H, HKV, DQK, DV, topk, dtype, check_correctness=True)
    else:
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 1, 4096, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
        test_indexer_loss_fwd_pipelined(B, S, SKV, H, HKV, DQK, DV, topk, dtype, check_correctness=False)
