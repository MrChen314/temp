# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tilelang.engine.callback import register_cuda_postproc_callback
import argparse


@tilelang.jit(
    out_idx=[-1],
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
    threads=384,
):
    """
    IndexerLoss Forward Kernel with Pipeline Optimization
    
    计算 sparse attention 的累加分布 AttnSum，用于后续的 loss 计算。
    
    线程分组:
    - 线程1组 (tx < 128): 处理前半 heads，两遍扫描计算 m/l 和 attn_sum
    - 线程2组 (128 <= tx < 256): 处理后半 heads，两遍扫描计算 m/l 和 attn_sum  
    - 线程3组 (tx >= 256): producer，负责加载 K 到 shared memory 双 buffer
    
    输出:
    - AttnSum: [batch, seq_len, topk] - 所有 head 的 attention 概率求和
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
    attn_sum_shape = [batch, seq_len, topk]
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
    # 每组线程处理一半的 heads
    H_half = H_per_block // 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor(1, indices_dtype),
        AttnSum: T.Tensor(attn_sum_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel((seq_len - kv_stride + 1 if CP0 else seq_len) * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            # Q shared memory - 两组线程共用
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
            
            # K validity mask
            is_kv_valid_0 = T.alloc_shared([BI], "bool", scope="shared")
            is_kv_valid_1 = T.alloc_shared([BI], "bool", scope="shared")
            
            # Indices shared memory for second pass
            indices_shared_0 = T.alloc_shared([BI], indices_dtype)
            indices_shared_1 = T.alloc_shared([BI], indices_dtype)

            # 线程1组的局部变量
            acc_s_1 = T.alloc_fragment([H_half, BI], accum_dtype)
            sumexp_1 = T.alloc_fragment([H_half], accum_dtype)
            sumexp_i_1 = T.alloc_fragment([H_half], accum_dtype)
            m_i_1 = T.alloc_fragment([H_half], accum_dtype)
            m_i_prev_1 = T.alloc_fragment([H_half], accum_dtype)
            attn_sum_local_1 = T.alloc_fragment([BI], accum_dtype)
            
            # 线程2组的局部变量
            acc_s_2 = T.alloc_fragment([H_half, BI], accum_dtype)
            sumexp_2 = T.alloc_fragment([H_half], accum_dtype)
            sumexp_i_2 = T.alloc_fragment([H_half], accum_dtype)
            m_i_2 = T.alloc_fragment([H_half], accum_dtype)
            m_i_prev_2 = T.alloc_fragment([H_half], accum_dtype)
            attn_sum_local_2 = T.alloc_fragment([BI], accum_dtype)
            
            # 共享的 attn_sum 用于最终合并
            attn_sum_shared = T.alloc_shared([BI], accum_dtype)
            
            # 用于存储最终的 m 和 l (第一遍扫描结束后)
            m_final_shared_1 = T.alloc_shared([H_half], accum_dtype)
            l_final_shared_1 = T.alloc_shared([H_half], accum_dtype)
            m_final_shared_2 = T.alloc_shared([H_half], accum_dtype)
            l_final_shared_2 = T.alloc_shared([H_half], accum_dtype)

            indices_local = T.alloc_local([1], indices_dtype)

            # Barriers
            bar_q = T.alloc_barrier(arrive_count=384)
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)
            bar_k_0_free = T.alloc_barrier(arrive_count=256)
            bar_k_1_free = T.alloc_barrier(arrive_count=256)
            bar_pass1_done = T.alloc_barrier(arrive_count=256)
            bar_pass2_k_0_ready = T.alloc_barrier(arrive_count=128)
            bar_pass2_k_1_ready = T.alloc_barrier(arrive_count=128)
            bar_pass2_k_0_free = T.alloc_barrier(arrive_count=256)
            bar_pass2_k_1_free = T.alloc_barrier(arrive_count=256)
            # 线程1组写完后 arrive，线程2组 wait
            bar_merge_ready = T.alloc_barrier(arrive_count=128)
            # 线程2组读完后 arrive，线程1组 wait (下一轮才能继续写)
            bar_merge_done = T.alloc_barrier(arrive_count=128)

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
                # 线程1组: 处理前半 heads [0, H_half)
                # ================================================================
                T.set_max_nreg(240, 1)
                T.fill(sumexp_1, 0)
                T.fill(m_i_1, -(2**30))
                T.barrier_wait(bar_q, 0)

                # ================================================================
                # Pass 1: 计算 m_global 和 l_global
                # ================================================================
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], 0, -T.infinity(acc_s_1.dtype))
                    # QK 矩阵乘 - 只用前半 heads
                    T.gemm(Q_shared_l[0:H_half, :], KV_shared_0_l, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r[0:H_half, :], KV_shared_0_r, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared[0:H_half, :], K_tail_shared_0, acc_s_1, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    # Online softmax
                    T.copy(m_i_1, m_i_prev_1)
                    T.reduce_max(acc_s_1, m_i_1, dim=1, clear=False)
                    for h_i in T.Parallel(H_half):
                        m_i_1[h_i] = T.max(m_i_1[h_i], m_i_prev_1[h_i])
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.exp2(acc_s_1[h_i, bi_i] * sm_scale - m_i_1[h_i] * sm_scale)
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], acc_s_1[h_i, bi_i], 0)
                    T.reduce_sum(acc_s_1, sumexp_i_1, dim=1)
                    for h_i in T.Parallel(H_half):
                        sumexp_1[h_i] = sumexp_1[h_i] * T.exp2((m_i_prev_1[h_i] - m_i_1[h_i]) * sm_scale) + sumexp_i_1[h_i]

                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], 0, -T.infinity(acc_s_1.dtype))
                    T.gemm(Q_shared_l[0:H_half, :], KV_shared_1_l, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r[0:H_half, :], KV_shared_1_r, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared[0:H_half, :], K_tail_shared_1, acc_s_1, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    T.copy(m_i_1, m_i_prev_1)
                    T.reduce_max(acc_s_1, m_i_1, dim=1, clear=False)
                    for h_i in T.Parallel(H_half):
                        m_i_1[h_i] = T.max(m_i_1[h_i], m_i_prev_1[h_i])
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.exp2(acc_s_1[h_i, bi_i] * sm_scale - m_i_1[h_i] * sm_scale)
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], acc_s_1[h_i, bi_i], 0)
                    T.reduce_sum(acc_s_1, sumexp_i_1, dim=1)
                    for h_i in T.Parallel(H_half):
                        sumexp_1[h_i] = sumexp_1[h_i] * T.exp2((m_i_prev_1[h_i] - m_i_1[h_i]) * sm_scale) + sumexp_i_1[h_i]

                    T.barrier_arrive(bar_k_1_free[0])

                # 保存最终的 m 和 l
                for h_i in T.Parallel(H_half):
                    m_final_shared_1[h_i] = m_i_1[h_i]
                    l_final_shared_1[h_i] = sumexp_1[h_i]
                
                T.barrier_arrive(bar_pass1_done)
                T.barrier_wait(bar_pass1_done, 0)

                # ================================================================
                # Pass 2: 使用 m_global/l_global 计算归一化 attention 并累加
                # ================================================================
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.fill(attn_sum_local_1, 0)
                    T.barrier_wait(bar_pass2_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], 0, -T.infinity(acc_s_1.dtype))
                    T.gemm(Q_shared_l[0:H_half, :], KV_shared_0_l, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r[0:H_half, :], KV_shared_0_r, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared[0:H_half, :], K_tail_shared_0, acc_s_1, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    # 使用最终的 m/l 归一化
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.exp2(acc_s_1[h_i, bi_i] * sm_scale - m_final_shared_1[h_i] * sm_scale) / l_final_shared_1[h_i]
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], acc_s_1[h_i, bi_i], 0)
                    
                    # Head sum 并累加到 attn_sum_local
                    for bi_i in T.Parallel(BI):
                        for h_i in T.serial(H_half):
                            attn_sum_local_1[bi_i] += acc_s_1[h_i, bi_i]
                    
                    # 写入 shared memory 供线程2组合并
                    for bi_i in T.Parallel(BI):
                        attn_sum_shared[bi_i] = attn_sum_local_1[bi_i]
                    
                    # 通知线程2组可以读取
                    T.barrier_arrive(bar_merge_ready[0])
                    # 等待线程2组读取完成后才能继续下一次写入
                    T.barrier_wait(bar_merge_done[0], (i_i * 2) & 1)

                    T.barrier_arrive(bar_pass2_k_0_free[0])

                    # Buffer 1
                    T.fill(attn_sum_local_1, 0)
                    T.barrier_wait(bar_pass2_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], 0, -T.infinity(acc_s_1.dtype))
                    T.gemm(Q_shared_l[0:H_half, :], KV_shared_1_l, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r[0:H_half, :], KV_shared_1_r, acc_s_1, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared[0:H_half, :], K_tail_shared_1, acc_s_1, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.exp2(acc_s_1[h_i, bi_i] * sm_scale - m_final_shared_1[h_i] * sm_scale) / l_final_shared_1[h_i]
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_1[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], acc_s_1[h_i, bi_i], 0)
                    
                    for bi_i in T.Parallel(BI):
                        for h_i in T.serial(H_half):
                            attn_sum_local_1[bi_i] += acc_s_1[h_i, bi_i]
                    
                    # 写入 shared memory 供线程2组合并
                    for bi_i in T.Parallel(BI):
                        attn_sum_shared[bi_i] = attn_sum_local_1[bi_i]
                    
                    # 通知线程2组可以读取
                    T.barrier_arrive(bar_merge_ready[0])
                    # 等待线程2组读取完成后才能继续下一次写入
                    T.barrier_wait(bar_merge_done[0], (i_i * 2 + 1) & 1)

                    T.barrier_arrive(bar_pass2_k_1_free[0])

            elif tx >= 128 and tx < 256:
                # ================================================================
                # 线程2组: 处理后半 heads [H_half, H_per_block)
                # ================================================================
                T.set_max_nreg(240, 1)
                T.fill(sumexp_2, 0)
                T.fill(m_i_2, -(2**30))
                T.barrier_wait(bar_q, 0)

                # ================================================================
                # Pass 1: 计算 m_global 和 l_global
                # ================================================================
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], 0, -T.infinity(acc_s_2.dtype))
                    # QK 矩阵乘 - 只用后半 heads
                    T.gemm(Q_shared_l[H_half:H_per_block, :], KV_shared_0_l, acc_s_2, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r[H_half:H_per_block, :], KV_shared_0_r, acc_s_2, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared[H_half:H_per_block, :], K_tail_shared_0, acc_s_2, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    T.copy(m_i_2, m_i_prev_2)
                    T.reduce_max(acc_s_2, m_i_2, dim=1, clear=False)
                    for h_i in T.Parallel(H_half):
                        m_i_2[h_i] = T.max(m_i_2[h_i], m_i_prev_2[h_i])
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.exp2(acc_s_2[h_i, bi_i] * sm_scale - m_i_2[h_i] * sm_scale)
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], acc_s_2[h_i, bi_i], 0)
                    T.reduce_sum(acc_s_2, sumexp_i_2, dim=1)
                    for h_i in T.Parallel(H_half):
                        sumexp_2[h_i] = sumexp_2[h_i] * T.exp2((m_i_prev_2[h_i] - m_i_2[h_i]) * sm_scale) + sumexp_i_2[h_i]

                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], 0, -T.infinity(acc_s_2.dtype))
                    T.gemm(Q_shared_l[H_half:H_per_block, :], KV_shared_1_l, acc_s_2, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r[H_half:H_per_block, :], KV_shared_1_r, acc_s_2, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared[H_half:H_per_block, :], K_tail_shared_1, acc_s_2, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    T.copy(m_i_2, m_i_prev_2)
                    T.reduce_max(acc_s_2, m_i_2, dim=1, clear=False)
                    for h_i in T.Parallel(H_half):
                        m_i_2[h_i] = T.max(m_i_2[h_i], m_i_prev_2[h_i])
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.exp2(acc_s_2[h_i, bi_i] * sm_scale - m_i_2[h_i] * sm_scale)
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], acc_s_2[h_i, bi_i], 0)
                    T.reduce_sum(acc_s_2, sumexp_i_2, dim=1)
                    for h_i in T.Parallel(H_half):
                        sumexp_2[h_i] = sumexp_2[h_i] * T.exp2((m_i_prev_2[h_i] - m_i_2[h_i]) * sm_scale) + sumexp_i_2[h_i]

                    T.barrier_arrive(bar_k_1_free[0])

                # 保存最终的 m 和 l
                for h_i in T.Parallel(H_half):
                    m_final_shared_2[h_i] = m_i_2[h_i]
                    l_final_shared_2[h_i] = sumexp_2[h_i]
                
                T.barrier_arrive(bar_pass1_done)
                T.barrier_wait(bar_pass1_done, 0)

                # ================================================================
                # Pass 2: 使用 m_global/l_global 计算归一化 attention 并累加
                # ================================================================
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.fill(attn_sum_local_2, 0)
                    T.barrier_wait(bar_pass2_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], 0, -T.infinity(acc_s_2.dtype))
                    T.gemm(Q_shared_l[H_half:H_per_block, :], KV_shared_0_l, acc_s_2, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r[H_half:H_per_block, :], KV_shared_0_r, acc_s_2, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared[H_half:H_per_block, :], K_tail_shared_0, acc_s_2, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.exp2(acc_s_2[h_i, bi_i] * sm_scale - m_final_shared_2[h_i] * sm_scale) / l_final_shared_2[h_i]
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.if_then_else(is_kv_valid_0[bi_i], acc_s_2[h_i, bi_i], 0)
                    
                    for bi_i in T.Parallel(BI):
                        for h_i in T.serial(H_half):
                            attn_sum_local_2[bi_i] += acc_s_2[h_i, bi_i]
                    
                    # 等待线程1组写入 shared memory
                    T.barrier_wait(bar_merge_ready[0], (i_i * 2) & 1)
                    
                    # 合并两组结果并写出
                    tk_start_0 = (i_i * 2) * BI
                    for bi_i in T.Parallel(BI):
                        AttnSum[b_i, s_i, tk_start_0 + bi_i] = attn_sum_shared[bi_i] + attn_sum_local_2[bi_i]
                    
                    # 通知线程1组可以继续
                    T.barrier_arrive(bar_merge_done[0])

                    T.barrier_arrive(bar_pass2_k_0_free[0])

                    # Buffer 1
                    T.fill(attn_sum_local_2, 0)
                    T.barrier_wait(bar_pass2_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], 0, -T.infinity(acc_s_2.dtype))
                    T.gemm(Q_shared_l[H_half:H_per_block, :], KV_shared_1_l, acc_s_2, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r[H_half:H_per_block, :], KV_shared_1_r, acc_s_2, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared[H_half:H_per_block, :], K_tail_shared_1, acc_s_2, transpose_B=True, wg_wait=-1)

                    T.wait_wgmma(0)

                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.exp2(acc_s_2[h_i, bi_i] * sm_scale - m_final_shared_2[h_i] * sm_scale) / l_final_shared_2[h_i]
                    for h_i, bi_i in T.Parallel(H_half, BI):
                        acc_s_2[h_i, bi_i] = T.if_then_else(is_kv_valid_1[bi_i], acc_s_2[h_i, bi_i], 0)
                    
                    for bi_i in T.Parallel(BI):
                        for h_i in T.serial(H_half):
                            attn_sum_local_2[bi_i] += acc_s_2[h_i, bi_i]
                    
                    # 等待线程1组写入 shared memory
                    T.barrier_wait(bar_merge_ready[0], (i_i * 2 + 1) & 1)
                    
                    # 合并两组结果并写出
                    tk_start_1 = (i_i * 2 + 1) * BI
                    for bi_i in T.Parallel(BI):
                        AttnSum[b_i, s_i, tk_start_1 + bi_i] = attn_sum_shared[bi_i] + attn_sum_local_2[bi_i]
                    
                    # 通知线程1组可以继续
                    T.barrier_arrive(bar_merge_done[0])

                    T.barrier_arrive(bar_pass2_k_1_free[0])

            elif tx >= 256:
                # ================================================================
                # 线程3组: Producer - 负责加载 K 到 shared memory
                # ================================================================
                T.set_max_nreg(80, 0)
                
                # Pass 1: 加载 K 给第一遍扫描
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2) * BI + r * 16 + (tx - 256) // 8]
                        indices_shared_0[r * 16 + (tx - 256) // 8] = indices_local[0]
                        is_kv_valid_0[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                        if is_kv_valid_0[r * 16 + (tx - 256) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
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
                        indices_shared_1[r * 16 + (tx - 256) // 8] = indices_local[0]
                        is_kv_valid_1[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                        if is_kv_valid_1[r * 16 + (tx - 256) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
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

                # 等待 Pass 1 完成
                T.barrier_arrive(bar_pass1_done)
                T.barrier_wait(bar_pass1_done, 0)

                # Pass 2: 重新加载 K 给第二遍扫描
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_pass2_k_0_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2) * BI + r * 16 + (tx - 256) // 8]
                        is_kv_valid_0[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                        if is_kv_valid_0[r * 16 + (tx - 256) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
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
                    T.cp_async_barrier_noinc(bar_pass2_k_0_ready[0])

                    # Buffer 1
                    T.barrier_wait(bar_pass2_k_1_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8]
                        is_kv_valid_1[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                        if is_kv_valid_1[r * 16 + (tx - 256) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
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
                    T.cp_async_barrier_noinc(bar_pass2_k_1_ready[0])

    return main


def indexer_loss_fwd_interface(
    q, kv, indices, q_start_index_s, kv_stride, sm_scale=None, is_casual=True, return_kernel=False, print_kernel=False
):
    """
    IndexerLoss Forward Interface
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        kv: [batch, seq_len_kv, kv_group, dim + tail_dim]
        indices: [batch, seq_len, kv_group, topk]
        q_start_index_s: chunk offset
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

    if q_start_index_s != 0:
        assert q_start_index_s > kv_stride, (
            "If it is because each cp has too short length, you should fix the logic involving CP0 (cp_rank == 0), to make sure q with pos < KV_Stride - 1 is masked (or you may just ignore how this is handled if nan in these q's Out would not effect others, which is reported to be likely to happen by wangding)"
        )
    CP0 = q_start_index_s == 0

    kernel = indexer_loss_fwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, kv_group, sm_scale, is_casual, CP0)
    if print_kernel:
        print(kernel.get_kernel_source())
    attn_sum = kernel(q, kv, indices, torch.tensor([q_start_index_s], dtype=torch.int32, device="cuda"))
    if return_kernel:
        return kernel, attn_sum
    if q_start_index_s == 0 and kv_stride > 1:
        attn_sum[:, : kv_stride - 1, :] = 0
    return attn_sum


def ref_indexer_loss_fwd_interface(q, kv, indices, q_start_index_s, kv_stride=1, sm_scale=None, is_casual=True):
    """
    PyTorch Reference Implementation for IndexerLoss Forward
    
    计算 sparse attention 的累加分布 AttnSum。
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        kv: [batch, seq_len_kv, kv_group, dim + tail_dim]
        indices: [batch, seq_len, kv_group, topk]
        q_start_index_s: chunk offset
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
    if q_start_index_s is None:
        q_start_index_s = sk * kv_stride - sq

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv  # [batch, seq_len_kv, kv_group, dim + tail_dim]

    g_index = g
    h_index = h // g
    
    # Causal mask for compressed KV
    compressed_casual_mask = torch.arange(q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device="cuda").view(
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
    B=1, S=4096, SKV=8192, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16, q_start_s_index=1024, check_correctness=True
):
    """
    Test IndexerLoss Forward Kernel
    """
    KV_stride = 1

    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(False) / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(False) / 10
    q_start_s_index_t = torch.tensor([q_start_s_index], dtype=torch.int32, device="cuda")

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(min(max(1, ((t + q_start_s_index) // KV_stride)), SKV))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    kernel, tl_attn_sum = indexer_loss_fwd_interface(q, kv, indices, q_start_s_index, KV_stride, return_kernel=True, print_kernel=True)

    def fn():
        attn_sum = kernel(q, kv, indices, q_start_s_index_t)
        if q_start_s_index == 0 and KV_stride > 1:
            attn_sum[:, : KV_stride - 1, :] = 0
        return attn_sum

    if check_correctness:
        ref_attn_sum = ref_indexer_loss_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_correctness", action="store_true")
    args = parser.parse_args()
    if args.test_correctness:
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 1, 1024, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
    else:
        B, S, SKV, H, HKV, DQK, DV, topk, dtype = 1, 4096, 8192, 128, 1, 576, 512, 2048, torch.bfloat16
    test_indexer_loss_fwd_pipelined(B, S, SKV, H, HKV, DQK, DV, topk, dtype, check_correctness=args.test_correctness)
