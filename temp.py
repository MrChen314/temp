# ruff: noqa
"""
Sparse MLA Forward with T.print Debug
参考 study/lecture4/2调试工具/README.md 添加调试语句
"""
import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_debug(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
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

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float32"

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

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
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
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
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            # ====== DEBUG: 打印加载的Q ======
            if bx == 0 and by == 0 and bz == 0:
                T.print(b_i, msg="[DEBUG] b_i:")
                T.print(s_i, msg="[DEBUG] s_i:")
                T.print(H0, msg="[DEBUG] H0:")
                T.print(H1, msg="[DEBUG] H1:")
                T.print(Q_shared, msg="[DEBUG] Q_shared after copy:")
                T.print(Q_tail_shared, msg="[DEBUG] Q_tail_shared after copy:")

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                # ====== DEBUG: 打印加载的KV (只在第一个iteration) ======
                if bx == 0 and by == 0 and bz == 0 and i_i == 0:
                    T.print(i_i, msg="[DEBUG] i_i (iteration):")
                    T.print(KV_shared, msg="[DEBUG] KV_shared after copy:")
                    T.print(K_tail_shared, msg="[DEBUG] K_tail_shared after copy:")
                    T.print(mask, msg="[DEBUG] mask:")

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # ====== DEBUG: 打印attention score (只在第一个iteration) ======
                if bx == 0 and by == 0 and bz == 0 and i_i == 0:
                    T.print(acc_s, msg="[DEBUG] acc_s after GEMM (QK^T):")

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]

                # ====== DEBUG: 打印softmax中间值 (只在第一个iteration) ======
                if bx == 0 and by == 0 and bz == 0 and i_i == 0:
                    T.print(m_i, msg="[DEBUG] m_i (row max):")
                    T.print(alpha, msg="[DEBUG] alpha (rescale factor):")
                    T.print(sumexp, msg="[DEBUG] sumexp (sum of exp):")
                    T.print(acc_s, msg="[DEBUG] acc_s after softmax (P):")

                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                # ====== DEBUG: 打印累积输出 (只在第一个iteration) ======
                if bx == 0 and by == 0 and bz == 0 and i_i == 0:
                    T.print(acc_o, msg="[DEBUG] acc_o after PV GEMM (first iter):")

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            # ====== DEBUG: 打印最终输出 ======
            if bx == 0 and by == 0 and bz == 0:
                T.print(acc_o, msg="[DEBUG] acc_o (final output after rescale):")
                T.print(sumexp, msg="[DEBUG] sumexp (final LSE):")

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mla_fwd_debug_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512, block_I=64, num_stages=2, threads=256):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = sparse_mla_fwd_debug(
        heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual, block_I=block_I, num_stages=num_stages, threads=threads
    )
    out, lse = kernel(q, kv, indices)
    return out, lse


def test_sparse_mla_fwd_debug(
    B=1,
    S=4,  # 使用较小的S以减少输出
    SKV=8,
    H=16,  # 使用较小的H以减少输出
    HKV=1,
    DQK=576,
    DV=512,
    topk=64,  # 使用较小的topk以减少输出
    dtype=torch.bfloat16,
    block_I=64,
    num_stages=2,
    threads=256,
):
    """
    测试带调试信息的 sparse_mla_fwd
    使用较小的参数以减少调试输出量
    """
    print("=" * 60)
    print("Testing sparse_mla_fwd_debug with small parameters")
    print(f"B={B}, S={S}, SKV={SKV}, H={H}, HKV={HKV}")
    print(f"DQK={DQK}, DV={DV}, topk={topk}")
    print("=" * 60)
    
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(False)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(False)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                valid_indices = min(topk, max(1, t))
                i_i = torch.randperm(max(1, t), device="cpu")[:valid_indices]
                indices[b, t, h, :len(i_i)] = i_i.to("cuda")

    print("\n[INFO] Running kernel with T.print debug statements...")
    print("=" * 60)
    
    tl_out, tl_lse = sparse_mla_fwd_debug_interface(
        q, kv, indices, 
        block_I=block_I, 
        num_stages=num_stages, 
        threads=threads
    )
    
    print("=" * 60)
    print("[INFO] Kernel execution completed!")
    print(f"Output shape: {tl_out.shape}")
    print(f"LSE shape: {tl_lse.shape}")
    print(f"Output (first 4 elements): {tl_out[0, 0, 0, :4]}")
    print(f"LSE (first 4 elements): {tl_lse[0, 0, :4]}")


if __name__ == "__main__":
    test_sparse_mla_fwd_debug()
