# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tilelang.engine.callback import register_cuda_postproc_callback
import argparse


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

    # 计算 kernel 中使用的 sm_scale（包含 log2(e) 因子）
    if sm_scale is None:
        sm_scale_log2e = (1.0 / dim_plus_tail_dim) ** 0.5 * 1.44269504
    else:
        sm_scale_log2e = sm_scale * 1.44269504

    kernel = indexer_loss_fwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, kv_group, sm_scale, is_casual, CP0)
    if print_kernel:
        print(kernel.get_kernel_source())
    
    # kernel 返回 M_out, L_out, QK_out
    m_out, l_out, qk_out = kernel(q, kv, indices, torch.tensor([q_start_index_s], dtype=torch.int32, device="cuda"))
    
    # 根据 m, l, qk 计算 attn_sum
    # qk_out: [batch, seq_len, heads, topk] - 已乘以 sm_scale_log2e
    # m_out: [batch, seq_len, heads] - max 值
    # l_out: [batch, seq_len, heads] - sum of exp2
    # attn_prob = exp2(qk_out - m_out * sm_scale_log2e) / l_out
    attn_prob = torch.exp2(qk_out - m_out.unsqueeze(-1) * sm_scale_log2e) / l_out.unsqueeze(-1)
    
    # 沿 head 维度求和
    attn_sum = attn_prob.sum(dim=2)  # [batch, seq_len, topk]
    
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

    # 计算 sm_scale（与接口中相同）
    sm_scale_log2e = (1.0 / DQK) ** 0.5 * 1.44269504

    def fn():
        # kernel 返回 m_out, l_out, qk_out
        m_out, l_out, qk_out = kernel(q, kv, indices, q_start_s_index_t)
        # 计算 attn_sum
        attn_prob = torch.exp2(qk_out - m_out.unsqueeze(-1) * sm_scale_log2e) / l_out.unsqueeze(-1)
        attn_sum = attn_prob.sum(dim=2)
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
