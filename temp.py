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
    seq_len_k,
    heads,
    dim,
    tail_dim,
    topk,
    k_stride,
    k_group=1,
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
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong k to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    head_k = heads // k_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    k_shape = [batch, seq_len_k, k_group, dim + tail_dim]
    indices_shape = [batch, seq_len, k_group, topk]
    m_out_shape = [batch, seq_len, heads]
    l_out_shape = [batch, seq_len, heads]
    qk_out_shape = [batch, seq_len, heads, topk]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = k_group
    H = head_k
    padded_H = max(tilelang.math.next_power_of_2(head_k), 16)
    if padded_H != H:
        assert k_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when k_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    assert NI % 2 == 0, "NI should be a multiple of 2"
    D = dim
    D_tail = tail_dim
    K_stride = k_stride
    if head_k > 64:
        assert head_k % 64 == 0, "head_k should be a multiple of 64"
        REPLICATE_H = head_k // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        q_start_index_s: T.Tensor(1, indices_dtype),
        M_out: T.Tensor(m_out_shape, accum_dtype),  # type: ignore
        L_out: T.Tensor(l_out_shape, accum_dtype),  # type: ignore
        QK_out: T.Tensor(qk_out_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel((seq_len - k_stride + 1 if CP0 else seq_len) * REPLICATE_H, batch, k_group, threads=threads) as (bx, by, bz):
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            
            K_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            K_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            K_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            K_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
            
            is_k_valid_0 = T.alloc_shared([BI], "bool", scope="shared")
            is_k_valid_1 = T.alloc_shared([BI], "bool", scope="shared")

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
            s_i = (bx + (K_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (bx // REPLICATE_H + (K_stride - 1 if CP0 else 0))
            q_i = q_start_index_s[0] + s_i
            max_k_i = (q_i + 1 - K_stride) // K_stride

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
                        acc_s[h_i, bi_i] = T.if_then_else(is_k_valid_0[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, K_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, K_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
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
                        acc_s[h_i, bi_i] = T.if_then_else(is_k_valid_0[bi_i], acc_s[h_i, bi_i], 0)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale) + sumexp_i[h_i]

                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_k_valid_1[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, K_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, K_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
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
                        acc_s[h_i, bi_i] = T.if_then_else(is_k_valid_1[bi_i], acc_s[h_i, bi_i], 0)
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
                        is_k_valid_0[r * 16 + (tx - 128) // 8] = indices_local[0] <= max_k_i
                        if is_k_valid_0[r * 16 + (tx - 128) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        K_shared_0_l[r * 16 + (tx - 128) // 8, 64 * u + (tx - 128) % 8 * 8 + v] = K[
                                            b_i, indices_local[0], g_i, 64 * u + (tx - 128) % 8 * 8 + v
                                        ]
                                        K_shared_0_r[r * 16 + (tx - 128) // 8, 64 * u + (tx - 128) % 8 * 8 + v] = K[
                                            b_i, indices_local[0], g_i, D // 2 + 64 * u + (tx - 128) % 8 * 8 + v
                                        ]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_0[r * 16 + (tx - 128) // 8, (tx - 128) % 8 * 8 + v] = K[
                                        b_i, indices_local[0], g_i, D + (tx - 128) % 8 * 8 + v
                                    ]
                    T.cp_async_barrier_noinc(bar_k_0_ready[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local[0] = Indices[b_i, s_i, g_i, (i_i * 2 + 1) * BI + r * 16 + (tx - 128) // 8]
                        is_k_valid_1[r * 16 + (tx - 128) // 8] = indices_local[0] <= max_k_i
                        if is_k_valid_1[r * 16 + (tx - 128) // 8]:
                            with T.attr("default", "async_scope", 1):
                                for u in T.serial(4):
                                    for v in T.vectorized(8):
                                        K_shared_1_l[r * 16 + (tx - 128) // 8, 64 * u + (tx - 128) % 8 * 8 + v] = K[
                                            b_i, indices_local[0], g_i, 64 * u + (tx - 128) % 8 * 8 + v
                                        ]
                                        K_shared_1_r[r * 16 + (tx - 128) // 8, 64 * u + (tx - 128) % 8 * 8 + v] = K[
                                            b_i, indices_local[0], g_i, D // 2 + 64 * u + (tx - 128) % 8 * 8 + v
                                        ]
                            with T.attr("default", "async_scope", 1):
                                for v in T.vectorized(8):
                                    K_tail_shared_1[r * 16 + (tx - 128) // 8, (tx - 128) % 8 * 8 + v] = K[
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


def calc_attn_dist(
    q, k, indices, chunk_offset=0, sm_scale=None, k_stride=1, is_casual=True, eps=1e-10
):
    """
    Calculate Attention Distribution
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        k: [batch, seq_len_k, k_group, dim + tail_dim]
        indices: [batch, seq_len, k_group, topk]
        chunk_offset: the starting position of the current chunk within the full sequence (used for causal mask)
        sm_scale: attention scaling factor
        k_stride: K stride
        is_casual: whether to use causal mask
        eps: epsilon for numerical stability
    
    Returns:
        attn_dist: [batch, seq_len, topk] - 归一化的 attention 分布
    """
    assert q.is_contiguous() and k.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_k, k_group, _ = k.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = 512

    assert k.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert k.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, k_group, topk)

    if sm_scale is None:
        sm_scale_log2e = (1.0 / dim_plus_tail_dim) ** 0.5 * 1.44269504
    else:
        sm_scale_log2e = sm_scale * 1.44269504
    CP0 = chunk_offset == 0

    kernel = indexer_loss_fwd(batch, seq_len, seq_len_k, heads, dim, tail_dim, topk, k_stride, k_group, sm_scale, is_casual, CP0)
    m_out, l_out, qk_out = kernel(q, k, indices, torch.tensor([chunk_offset], dtype=torch.int32, device="cuda"))
    attn_sum_kernel = attn_sum_fwd(batch, seq_len, heads, topk, sm_scale_log2e)
    attn_sum = attn_sum_kernel(qk_out, m_out, l_out)

    # 归一化得到 attn_dist
    attn_total = attn_sum.sum(dim=-1, keepdim=True) + eps
    attn_dist = attn_sum / attn_total

    return attn_dist


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
# PyTorch 参考实现
# ============================================================================

def ref_calc_attn_dist(q, k, indices, chunk_offset=0, k_stride=1, sm_scale=None, is_casual=True, eps=1e-10):
    """
    PyTorch Reference Implementation for calc_attn_dist
    
    计算 sparse attention 的归一化分布 attn_dist。
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        k: [batch, seq_len_k, k_group, dim + tail_dim]
        indices: [batch, seq_len, k_group, topk]
        chunk_offset: 当前 chunk 在完整序列中的起始位置 (用于 causal mask)
        k_stride: K stride
        sm_scale: attention scaling factor
        is_casual: whether to use causal mask
        eps: epsilon for numerical stability
    
    Returns:
        attn_dist: [batch, seq_len, topk] - 归一化的 attention 分布
    """
    q = q.float()
    k = k.float()
    indices_transposed = indices.transpose(1, 2)  # [batch, k_group, seq_len, topk]
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = k.shape
    if chunk_offset is None:
        chunk_offset = sk * k_stride - sq

    assert k.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512

    g_index = g
    h_index = h // g
    
    # Causal mask for compressed K
    compressed_casual_mask = torch.arange(chunk_offset, sq + chunk_offset, dtype=torch.int32, device=q.device).view(
        -1, 1
    ) >= torch.arange(k_stride - 1, sk * k_stride, k_stride, dtype=torch.int32, device=q.device).view(1, -1)

    # Sparse mask based on indices
    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices_transposed.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : k_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)  # [b, g, 1, sq, sk]

    q_reshaped = q.view(b, sq, g, -1, dim_q)  # [b, sq, g, h_per_g, dim_q]
    
    # Compute attention scores
    score = torch.einsum("bmghd,bngd->bghmn", q_reshaped, k)  # [b, g, h_per_g, sq, sk]
    sm_scale_val = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale_val)
    p = score.softmax(dim=-1)  # [b, g, h_per_g, sq, sk]
    
    # Sum over all heads: [b, g, h_per_g, sq, sk] -> [b, sq, sk]
    p_sum = p.sum(dim=(1, 2))  # [b, sq, sk]
    
    # Gather to get attn_sum for topk indices
    # indices_transposed: [batch, k_group, seq_len, topk]
    # 需要将 p_sum 按 indices gather
    indices_for_gather = indices_transposed[:, 0, :, :]  # [batch, seq_len, topk] (假设 k_group=1)
    attn_sum = torch.gather(p_sum, dim=-1, index=indices_for_gather.long())  # [batch, seq_len, topk]
    
    # 归一化得到 attn_dist
    attn_total = attn_sum.sum(dim=-1, keepdim=True) + eps
    attn_dist = attn_sum / attn_total
    
    return attn_dist


# ============================================================================
# 前向精度测试
# ============================================================================

def run_fwd_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """运行单个前向精度测试 - 比较 attn_dist"""
    torch.manual_seed(config.seed)
    
    # tilelang 版本使用固定的 dim=576
    head_dim = 576
    
    # 生成随机输入
    q = torch.randn(config.batch_size, config.chunk_size, config.num_heads, head_dim, 
                    device=device, dtype=torch.bfloat16) / 10
    k = torch.randn(config.batch_size, config.seq_len, 1, head_dim, 
                    device=device, dtype=torch.bfloat16) / 10
    
    q.clamp_(-10, 10)
    k.clamp_(-10, 10)
    
    # 计算 chunk_offset
    chunk_offset = config.seq_len - config.chunk_size
    k_stride = 1
    
    # 生成 indices
    indices = torch.full((config.batch_size, config.chunk_size, 1, config.topk), 
                         config.seq_len, dtype=torch.int32, device=device)
    for b in range(config.batch_size):
        for t in range(config.chunk_size):
            max_valid_idx = min(max(1, ((t + chunk_offset) // k_stride)), config.seq_len)
            i_i = torch.randperm(max_valid_idx, device=device)[:config.topk]
            indices[b, t, 0, : len(i_i)] = i_i.to(torch.int32)
    
    # TileLang kernel: 计算 attn_dist
    tl_attn_dist = calc_attn_dist(q, k, indices, chunk_offset, k_stride=k_stride)
    
    # PyTorch 参考: 计算 attn_dist
    ref_attn_dist = ref_calc_attn_dist(q, k, indices, chunk_offset, k_stride=k_stride)
    
    # 比较结果
    def calc_diff(a, b):
        abs_diff = torch.abs(a - b)
        max_diff = abs_diff.max().item()
        rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item() * 100
        return max_diff, rel_diff
    
    max_diff, rel_diff = calc_diff(ref_attn_dist, tl_attn_dist.to(ref_attn_dist.dtype))
    passed = rel_diff < 1e-3  # 相对误差 < 0.001%
    
    return {
        'config': config,
        'ref_max': ref_attn_dist.abs().max().item(),
        'tl_max': tl_attn_dist.abs().max().item(),
        'max_diff': max_diff,
        'rel_diff': rel_diff,
        'passed': passed
    }


def test_fwd_accuracy(configs: List[TestConfig]):
    """批量运行前向精度测试 - 比较 attn_dist"""
    print("\n" + "=" * 110)
    print("前向精度测试 (PyTorch attn_dist vs TileLang attn_dist)")
    print("=" * 110)
    
    results = []
    for config in configs:
        try:
            result = run_fwd_accuracy_test(config)
            results.append(result)
        except Exception as e:
            print(f"跳过测试 {config.name}: {e}")
            continue
    
    print(f"\n{'Name':<12} {'Config':<55} {'MaxDiff':<12} {'RelDiff(%)':<12} {'Pass':<6}")
    print("-" * 97)
    for r in results:
        print(f"{r['config'].name:<12} {str(r['config']):<55} "
              f"{r['max_diff']:<12.2e} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 97)
    print(f"前向测试 (attn_dist): {passed_count}/{len(results)} 通过")
    
    return results


# ============================================================================
# 前向性能测试
# ============================================================================

def test_fwd_performance(
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
    """前向性能测试"""
    import time
    
    torch.manual_seed(seed)
    device = 'cuda'
    
    print("\n" + "=" * 80)
    print("前向性能测试 (TileLang calc_attn_dist)")
    print("=" * 80)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print("=" * 80)
    
    q = torch.randn(batch_size, chunk_size, num_heads, head_dim, device=device, dtype=torch.bfloat16) / 10
    k = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16) / 10
    q.clamp_(-10, 10)
    k.clamp_(-10, 10)
    
    chunk_offset = seq_len - chunk_size
    k_stride = 1
    
    # 生成 indices
    indices = torch.full((batch_size, chunk_size, 1, topk), seq_len, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for t in range(chunk_size):
            max_valid_idx = min(max(1, ((t + chunk_offset) // k_stride)), seq_len)
            i_i = torch.randperm(max_valid_idx, device=device)[:topk]
            indices[b, t, 0, : len(i_i)] = i_i.to(torch.int32)
    
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated() / (1024**3)
    
    # 仅前向测试
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = calc_attn_dist(q, k, indices, chunk_offset, k_stride=k_stride)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        _ = calc_attn_dist(q, k, indices, chunk_offset, k_stride=k_stride)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start) / num_benchmark * 1000
    fwd_peak = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"\n>>> 性能 (warmup={num_warmup}, iters={num_benchmark})")
    print(f"  前向时间:   {fwd_time:.3f} ms")
    
    print(f"\n>>> 显存峰值")
    print(f"  基准显存:   {base_memory:.2f} GB")
    print(f"  前向峰值:   {fwd_peak:.2f} GB (增量: {fwd_peak - base_memory:.2f} GB)")
    
    # 计算 TFLOPS
    flops_fwd = batch_size * chunk_size * head_dim * topk * 2 * num_heads
    print(f"\n>>> 计算吞吐量")
    print(f"  前向 TFLOPS: {flops_fwd / (fwd_time * 1e-3) / 1e12:.2f}")
    
    # IO bandwidth
    io_bytes = batch_size * chunk_size * head_dim * topk * 2  # bf16
    print(f"  IO 带宽: {io_bytes / (fwd_time * 1e-3) / 1e12:.2f} TB/s")
    
    return {
        'fwd_time': fwd_time,
        'fwd_peak': fwd_peak,
    }
