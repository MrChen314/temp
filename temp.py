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
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            
            KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
            
            is_kv_valid_0 = T.alloc_shared([BI], "bool", scope="shared")
            is_kv_valid_1 = T.alloc_shared([BI], "bool", scope="shared")

            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            indices_local = T.alloc_local([1], indices_dtype)

            bar_q = T.alloc_barrier(arrive_count=256)
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

            T.copy(Q[b_i, s_i, H0:H1, 0 : D // 2], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D // 2 : D], Q_shared_r)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
            T.barrier_arrive(bar_q)

            if tx < 128:
                T.set_max_nreg(240, 1)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))
                T.barrier_wait(bar_q, 0)

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

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

                for h_i in T.Parallel(H_per_block):
                    M_out[b_i, s_i, H0 + h_i] = m_i[h_i]
                    L_out[b_i, s_i, H0 + h_i] = sumexp[h_i]

            elif tx >= 128:
                T.set_max_nreg(80, 0)
                
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
            qk_local = T.alloc_fragment([heads, block_T], accum_dtype)
            m_local = T.alloc_fragment([heads], accum_dtype)
            l_local = T.alloc_fragment([heads], accum_dtype)
            attn_sum_local = T.alloc_fragment([block_T], accum_dtype)
            
            T.copy(M_out[by, bx, 0:heads], m_local)
            T.copy(L_out[by, bx, 0:heads], l_local)
            
            for t_i in T.serial(NT):
                T.copy(QK_out[by, bx, 0:heads, t_i * block_T : (t_i + 1) * block_T], qk_local)
                
                for h_i, t_j in T.Parallel(heads, block_T):
                    qk_local[h_i, t_j] = T.exp2(qk_local[h_i, t_j] - m_local[h_i] * sm_scale_log2e) / l_local[h_i]
                
                T.reduce_sum(qk_local, attn_sum_local, dim=0)
                T.copy(attn_sum_local, AttnSum[by, bx, t_i * block_T : (t_i + 1) * block_T])
    
    return main


def indexer_loss_fwd_interface(
    q, kv, indices, chunk_offset=0, sm_scale=None, kv_stride=1, is_casual=True
):
    """
    IndexerLoss Forward Interface
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        kv: [batch, seq_len_kv, kv_group, dim + tail_dim]
        indices: [batch, seq_len, kv_group, topk]
        chunk_offset: the starting position of the current chunk within the full sequence (used for causal mask)
        sm_scale: attention scaling factor
        kv_stride: KV stride
        is_casual: whether to use causal mask
    
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

    if sm_scale is None:
        sm_scale_log2e = (1.0 / dim_plus_tail_dim) ** 0.5 * 1.44269504
    else:
        sm_scale_log2e = sm_scale * 1.44269504
    CP0 = chunk_offset == 0

    kernel = indexer_loss_fwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, kv_group, sm_scale, is_casual, CP0)
    m_out, l_out, qk_out = kernel(q, kv, indices, torch.tensor([chunk_offset], dtype=torch.int32, device="cuda"))
    attn_sum_kernel = attn_sum_fwd(batch, seq_len, heads, topk, sm_scale_log2e)
    attn_sum = attn_sum_kernel(qk_out, m_out, l_out)

    return attn_sum
