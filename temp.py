# ruff: noqa
"""
性能测试脚本: 对比 sparse_mla_bwd 实现与 PyTorch 参考实现
- sparse_mla_bwd: tilelang 实现
- torch ref: PyTorch 参考实现 (autograd)
"""
import torch
import tilelang
from tilelang import language as T
from typing import Callable


# ============================================================================
# 前向 Kernel (用于反向传播的依赖)
# ============================================================================
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_kernel(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim)
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
    assert is_causal == True
    assert topk % block_I == 0
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

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

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

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

            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


# ============================================================================
# 反向 Kernels (来自 sparse_mla_bwd.py)
# ============================================================================
@tilelang.jit(out_idx=[-1])
def preprocess_kernel(
    B,
    S,
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel_func(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
                T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel_func


@tilelang.jit(out_idx=[-1])
def postprocess_kernel(
    B,
    S_kv,
    D,
    D_tail,
    kv_group=1,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    dkv_shape = [B, S_kv, kv_group, D + D_tail]

    @T.prim_func
    def postprocess_kernel_func(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, B, threads=threads) as (bx, by, bz):
            T.copy(
                dKV[bz, bx * block_N : (bx + 1) * block_N, by, :],
                dKV_out[bz, bx * block_N : (bx + 1) * block_N, by, :],
            )

    return postprocess_kernel_func


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def bwd_kernel(
    B,
    S,
    S_kv,
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=0,
    threads=256,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert is_causal == True
    assert topk % block_size == 0
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504

    H_kv = H // kv_group
    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mla_bwd_kernel_func(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(S, B, kv_group, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([padded_H, D], dtype)
            Q_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dO_shared = T.alloc_shared([padded_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([padded_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([padded_H, BS], dtype)
            dQ_shared = T.alloc_shared([padded_H, D], dtype)
            dQ_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)

            acc_p = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([padded_H, D], accum_dtype)
            acc_dq_tail = T.alloc_fragment([padded_H, D_tail], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

            max_kv_i = s_i

            T.copy(Q[by, s_i, bz * padded_H : (bz + 1) * padded_H, :D], Q_shared)
            T.copy(Q[by, s_i, bz * padded_H : (bz + 1) * padded_H, D:], Q_tail_shared)
            T.copy(dO[by, s_i, bz * padded_H : (bz + 1) * padded_H, :D], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            T.annotate_layout(
                {
                    dQ_shared: tilelang.layout.make_swizzled_layout(dQ_shared),
                    dQ_tail_shared: tilelang.layout.make_swizzled_layout(dQ_tail_shared),
                }
            )

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, bz, i_i * BS + bi_i] <= max_kv_i

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz, i_i * BS + bi_i], bz, d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz, i_i * BS + bi_i], bz, D + d_i]
                T.gemm(Q_tail_shared, KV_tail_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * padded_H + h_i])

                T.copy(acc_p, P_shared_cast)

                T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * padded_H + h_i]) * sm_scale

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(dP_shared_cast, Q_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                T.clear(acc_dkv_tail)
                T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        if bi_i < BS // split_store:
                            acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[by, Indices[by, s_i, bz, i_i * BS + bi_i + s * (BS // split_store)], bz, d_i * 4],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

                    for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                        T.atomic_addx4(
                            dKV[by, Indices[by, s_i, bz, i_i * BS + bi_i + s * (BS // split_store)], bz, D + d_i * 4],
                            acc_dkv_tail_shared[bi_i, d_i * 4],
                        )

            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * padded_H : (bz + 1) * padded_H, :D])
            T.copy(dQ_tail_shared, dQ[by, s_i, bz * padded_H : (bz + 1) * padded_H, D:])

    return sparse_mla_bwd_kernel_func


# ============================================================================
# 组合接口
# ============================================================================
def sparse_mla_fwd(q, kv, indices, sm_scale=None, d_v=512):
    """前向传播接口"""
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape
    dim = d_v
    tail_dim = dim_plus_tail_dim - dim
    topk = indices.shape[-1]

    kernel = sparse_mla_fwd_kernel(
        heads=heads,
        dim=dim,
        tail_dim=tail_dim,
        topk=topk,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=True,
    )
    out, lse = kernel(q, kv, indices)
    return out, lse


def sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=None, d_v=512):
    """反向传播接口"""
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = kv.shape
    D = d_v
    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]

    # Get kernels
    preprocess_k = preprocess_kernel(B, S, H, D)
    bwd_k = bwd_kernel(B, S, S_kv, H, D, D_tail, topk, kv_group, sm_scale, is_causal=True)
    postprocess_k = postprocess_kernel(B, S_kv, D, D_tail, kv_group)

    delta = preprocess_k(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_k(q, kv, do, indices, lse, delta, dkv)
    dkv = postprocess_k(dkv)

    return dq, dkv


# ============================================================================
# PyTorch 参考实现
# ============================================================================
def ref_sparse_mla_fwd(q, kv, indices, sm_scale=None, d_v=512):
    """PyTorch 前向参考实现"""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, dim_plus_tail_dim = kv.shape

    dim = d_v
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g

    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device=q.device).view(-1, 1) >= torch.arange(
        0, sk, dtype=torch.int32, device=q.device
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


def ref_sparse_mla_bwd(q, kv, do, indices, sm_scale=None, d_v=512):
    """PyTorch 反向参考实现 (使用 autograd)"""
    q = q.detach().clone().float().requires_grad_(True)
    kv = kv.detach().clone().float().requires_grad_(True)
    do = do.float()
    
    # 前向传播
    indices_t = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, dim_plus_tail_dim = kv.shape

    dim = d_v
    k = kv
    v = kv[..., :dim]

    g_index = g
    h_index = h // g

    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device=q.device).view(-1, 1) >= torch.arange(
        0, sk, dtype=torch.int32, device=q.device
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices_t.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask = mask.view(b, g_index, 1, sq, sk)

    q_view = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q_view, k)
    sm_scale_val = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale_val)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p, v)
    o = o.reshape(b, sq, h, dim)

    # 反向传播
    o.backward(do)

    return q.grad.to(torch.bfloat16), kv.grad.to(torch.bfloat16)


# ============================================================================
# 性能测试函数
# ============================================================================
def test_bwd_performance(
    B=1,
    S=4096,
    SKV=4096,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    check_correctness=True,
    warmup=50,
    rep=100,
):
    """
    测试 sparse_mla_bwd 的性能
    """
    print("=" * 80)
    print("Sparse MLA Backward 性能对比测试")
    print("=" * 80)
    print(f"配置: B={B}, S={S}, SKV={SKV}, H={H}, HKV={HKV}")
    print(f"      DQK={DQK}, DV={DV}, topk={topk}")
    print(f"      warmup={warmup}, rep={rep}")
    print("=" * 80)

    # 生成测试数据
    torch.random.manual_seed(42)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda") / 10
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda") / 10
    do = torch.randn((B, S, H, DV), dtype=dtype, device="cuda") / 10
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    do.clamp_(-10, 10)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t), device="cuda")[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    tail_dim = DQK - DV

    results = []

    # ========================================================================
    # 1. TileLang 实现
    # ========================================================================
    print("\n[1] TileLang 实现 (sparse_mla_bwd)")
    print("-" * 40)
    try:
        # 先进行前向传播获取 o 和 lse
        tl_out, tl_lse = sparse_mla_fwd(q, kv, indices, d_v=DV)
        
        # 运行一次反向传播
        tl_dq, tl_dkv = sparse_mla_bwd(q, kv, tl_out, do, indices, tl_lse, d_v=DV)

        # 正确性验证
        if check_correctness:
            ref_dq, ref_dkv = ref_sparse_mla_bwd(q, kv, do, indices, d_v=DV)
            try:
                torch.testing.assert_close(tl_dq, ref_dq, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(tl_dkv, ref_dkv, rtol=1e-2, atol=1e-2)
                print("  ✓ 正确性验证通过")
            except AssertionError as e:
                max_diff_dq = (tl_dq - ref_dq).abs().max().item()
                max_diff_dkv = (tl_dkv - ref_dkv).abs().max().item()
                print(f"  ✗ 正确性验证失败，dQ 最大差异: {max_diff_dq:.6f}, dKV 最大差异: {max_diff_dkv:.6f}")

        # 性能测试
        def fn_tl():
            return sparse_mla_bwd(q, kv, tl_out, do, indices, tl_lse, d_v=DV)

        from tilelang.profiler import do_bench
        ms_tl = do_bench(fn_tl, rep=rep, warmup=warmup)

        results.append({
            "name": "TileLang 实现 (sparse_mla_bwd)",
            "time_ms": ms_tl,
            "success": True,
        })
        print(f"  平均时间: {ms_tl:.4f} ms")

    except Exception as e:
        import traceback
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        results.append({"name": "TileLang 实现", "time_ms": float('inf'), "success": False})

    # ========================================================================
    # 2. PyTorch 参考实现
    # ========================================================================
    print("\n[2] PyTorch 参考实现 (autograd)")
    print("-" * 40)
    try:
        # 运行一次
        ref_dq, ref_dkv = ref_sparse_mla_bwd(q, kv, do, indices, d_v=DV)
        print("  ✓ 参考实现运行成功")

        # 性能测试
        def fn_ref():
            return ref_sparse_mla_bwd(q, kv, do, indices, d_v=DV)

        from tilelang.profiler import do_bench
        ms_ref = do_bench(fn_ref, rep=min(rep, 10), warmup=min(warmup, 5))  # ref 较慢

        results.append({
            "name": "PyTorch 参考实现 (autograd)",
            "time_ms": ms_ref,
            "success": True,
        })
        print(f"  平均时间: {ms_ref:.4f} ms")

    except Exception as e:
        import traceback
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        results.append({"name": "PyTorch 参考实现", "time_ms": float('inf'), "success": False})

    # ========================================================================
    # 汇总结果
    # ========================================================================
    print("\n" + "=" * 80)
    print("性能汇总")
    print("=" * 80)

    # 计算理论带宽和 TFLOPS
    # BWD FLOPS: 大约是 FWD 的 2.5-3 倍
    per_token_flop = 2 * sum([
        H * DV * topk,      # dO @ V^T
        H * DQK * topk,     # dP @ K
        H * DQK * topk,     # dP^T @ Q
        H * DQK * topk,     # P^T @ dO
        H * DV * topk,      # S @ V
    ])
    flops = per_token_flop * S
    io_bytes = B * S * max(DQK * 2, DQK + DV) * topk * 2

    print(f"{'方案':<45} {'时间(ms)':<12} {'带宽(TB/s)':<12} {'TFLOPS':<10} {'加速比':<10}")
    print("-" * 95)

    # 以参考实现为基准
    ref_time = next((r["time_ms"] for r in results if "参考" in r["name"] and r["success"]), float('inf'))

    for r in results:
        if r["success"]:
            bandwidth = io_bytes / (r["time_ms"] * 1e-3) / 1e12
            tflops = flops / (r["time_ms"] * 1e-3) / 1e12
            speedup = ref_time / r["time_ms"] if r["time_ms"] > 0 else 0
            print(f"{r['name']:<45} {r['time_ms']:<12.4f} {bandwidth:<12.2f} {tflops:<10.2f} {speedup:<10.2f}x")
        else:
            print(f"{r['name']:<45} {'失败':<12}")

    print("=" * 80)

    # 打印最快版本
    successful = [r for r in results if r["success"]]
    if successful:
        fastest = min(successful, key=lambda x: x["time_ms"])
        print(f"\n🏆 最快版本: {fastest['name']} ({fastest['time_ms']:.4f} ms)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sparse MLA Backward 性能测试")
    parser.add_argument("--no-check", action="store_true", help="跳过正确性检查")
    parser.add_argument("--warmup", type=int, default=50, help="预热次数")
    parser.add_argument("--rep", type=int, default=100, help="重复次数")
    parser.add_argument("--S", type=int, default=4096, help="Sequence length")
    parser.add_argument("--SKV", type=int, default=4096, help="KV sequence length")
    parser.add_argument("--H", type=int, default=128, help="Number of heads")
    args = parser.parse_args()

    test_bwd_performance(
        B=1,
        S=args.S,
        SKV=args.SKV,
        H=args.H,
        HKV=1,
        DQK=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=not args.no_check,
        warmup=args.warmup,
        rep=args.rep,
    )
