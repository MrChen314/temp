# ruff: noqa
import torch
import tilelang
from tilelang import language as T
from tilelang.engine.callback import register_cuda_postproc_callback
import argparse
from dataclasses import dataclass
from typing import List


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
    """
    IndexerLoss Forward Kernel with Pipeline Optimization
    
    Compute sparse attention sum using pre-computed LSE (log-sum-exp).
    
    Thread Groups (Simplified Design):
    - Thread Group 1 (tx < 128): Compute qk matmul and attn_sum for all heads
    - Thread Group 3 (tx >= 128): Producer, responsible for loading K to shared memory with double buffering
    
    Inputs:
    - Q: [batch, seq_len, heads, dim + tail_dim] - Query
    - K: [batch, seq_len_k, k_group, dim + tail_dim] - Key
    - Indices: [batch, seq_len, k_group, topk] - Sparse indices
    - q_start_index_s: scalar - Starting position of current chunk
    - Lse: [batch, seq_len, heads] - Pre-computed log-sum-exp from full attention
    
    Outputs:
    - AttnSum: [batch, seq_len, topk] - Attention probabilities summed over head dimension
    """
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
    lse_shape = [batch, seq_len, heads]
    attn_sum_shape = [batch, seq_len, topk]
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
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore - Pre-computed LSE
        AttnSum: T.Tensor(attn_sum_shape, accum_dtype),  # type: ignore - Output
    ):
        with T.Kernel((seq_len - k_stride + 1 if CP0 else seq_len) * REPLICATE_H, batch, k_group, threads=threads) as (bx, by, bz):
            # Q shared memory - shared by all threads
            Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            
            # K shared memory - double buffer
            K_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
            K_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
            K_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
            K_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
            K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
            K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
            
            # K validity mask - double buffer
            is_k_valid_0 = T.alloc_shared([BI], "bool", scope="shared")
            is_k_valid_1 = T.alloc_shared([BI], "bool", scope="shared")

            # Local variables for thread group 1 - process all heads
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            lse_local = T.alloc_fragment([H_per_block], accum_dtype)
            attn_sum_local = T.alloc_fragment([BI], accum_dtype)

            indices_local = T.alloc_local([1], indices_dtype)

            # Barriers - simplified
            bar_q = T.alloc_barrier(arrive_count=256)
            # K buffer synchronization
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

            # Load Q to shared memory
            T.copy(Q[b_i, s_i, H0:H1, 0 : D // 2], Q_shared_l)
            T.copy(Q[b_i, s_i, H0:H1, D // 2 : D], Q_shared_r)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
            T.barrier_arrive(bar_q)

            if tx < 128:
                # ================================================================
                # Thread Group 1: Compute qk matmul and attn_sum for all heads
                # ================================================================
                T.set_max_nreg(240, 1)
                
                # Load LSE for this position
                T.copy(Lse[b_i, s_i, H0:H1], lse_local)
                
                T.barrier_wait(bar_q, 0)

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_k_valid_0[bi_i], 0, -T.infinity(acc_s.dtype))
                    # QK matmul - process all heads
                    T.gemm(Q_shared_l, K_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, K_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=0)

                    # Compute attn_prob = exp2(qk * sm_scale - lse) and sum over heads
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - lse_local[h_i])
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_k_valid_0[bi_i], acc_s[h_i, bi_i], 0)
                    
                    # Sum along head dimension
                    T.reduce_sum(acc_s, attn_sum_local, dim=0)
                    
                    # Write attn_sum for buffer 0
                    tk_start_0 = (i_i * 2) * BI
                    T.copy(attn_sum_local, AttnSum[b_i, s_i, tk_start_0 : tk_start_0 + BI])

                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_k_valid_1[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared_l, K_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_shared_r, K_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                    T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=0)

                    # Compute attn_prob = exp2(qk * sm_scale - lse) and sum over heads
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - lse_local[h_i])
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(is_k_valid_1[bi_i], acc_s[h_i, bi_i], 0)
                    
                    # Sum along head dimension
                    T.reduce_sum(acc_s, attn_sum_local, dim=0)
                    
                    # Write attn_sum for buffer 1
                    tk_start_1 = (i_i * 2 + 1) * BI
                    T.copy(attn_sum_local, AttnSum[b_i, s_i, tk_start_1 : tk_start_1 + BI])

                    T.barrier_arrive(bar_k_1_free[0])

            elif tx >= 128:
                # ================================================================
                # Thread Group 3: Producer - Load K to shared memory
                # ================================================================
                T.set_max_nreg(80, 0)
                
                # Load K for thread group 1
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


def calc_attn_dist(
    q, k, indices, lse, chunk_offset=0, sm_scale=None, k_stride=1, is_casual=True, eps=1e-10
):
    """
    Calculate Attention Distribution using pre-computed LSE
    
    Args:
        q: [batch, chunk_size, heads, dim + tail_dim]
        k: [batch, seq_len_k, k_group, dim + tail_dim]
        indices: [batch, chunk_size, k_group, topk]
        lse: [batch, chunk_size, heads] - Pre-computed LSE for the chunk
        chunk_offset: starting position of current chunk in full sequence (for causal mask)
        sm_scale: attention scaling factor
        k_stride: K stride
        is_casual: whether to use causal mask
        eps: epsilon for numerical stability
    
    Returns:
        attn_dist: [batch, chunk_size, topk] - normalized attention distribution
    """
    assert q.is_contiguous() and k.is_contiguous() and indices.is_contiguous() and lse.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_k, k_group, _ = k.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = 512

    assert k.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert k.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, k_group, topk)
    assert lse.shape == (batch, seq_len, heads), f"lse shape mismatch: {lse.shape} vs {(batch, seq_len, heads)}"

    CP0 = chunk_offset == 0

    kernel = indexer_loss_fwd(batch, seq_len, seq_len_k, heads, dim, tail_dim, topk, k_stride, k_group, sm_scale, is_casual, CP0)
    attn_sum = kernel(q, k, indices, torch.tensor([chunk_offset], dtype=torch.int32, device="cuda"), lse)

    # Normalize to get attn_dist
    attn_total = attn_sum.sum(dim=-1, keepdim=True) + eps
    attn_dist = attn_sum / attn_total

    return attn_dist


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class TestConfig:
    """Test Configuration"""
    name: str
    batch_size: int = 1
    num_heads: int = 128
    chunk_size: int = 256
    seq_len: int = 256
    head_dim: int = 576  # tilelang version fixed at 576
    topk: int = 128
    seed: int = 42
    
    def __str__(self):
        return (f"batch={self.batch_size}, heads={self.num_heads}, "
                f"chunk={self.chunk_size}, seq={self.seq_len}, "
                f"dim={self.head_dim}, topk={self.topk}")


# ============================================================================
# PyTorch Reference Implementation
# ============================================================================

def ref_calc_attn_dist(q, k, indices, chunk_offset=0, k_stride=1, sm_scale=None, is_casual=True, eps=1e-10):
    """
    PyTorch Reference Implementation for calc_attn_dist
    
    Compute normalized sparse attention distribution attn_dist.
    
    Args:
        q: [batch, seq_len, heads, dim + tail_dim]
        k: [batch, seq_len_k, k_group, dim + tail_dim]
        indices: [batch, seq_len, k_group, topk]
        chunk_offset: starting position of current chunk in full sequence (for causal mask)
        k_stride: K stride
        sm_scale: attention scaling factor
        is_casual: whether to use causal mask
        eps: epsilon for numerical stability
    
    Returns:
        attn_dist: [batch, seq_len, topk] - normalized attention distribution
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
    indices_for_gather = indices_transposed[:, 0, :, :]  # [batch, seq_len, topk] (assume k_group=1)
    attn_sum = torch.gather(p_sum, dim=-1, index=indices_for_gather.long())  # [batch, seq_len, topk]
    
    # Normalize to get attn_dist
    attn_total = attn_sum.sum(dim=-1, keepdim=True) + eps
    attn_dist = attn_sum / attn_total
    
    return attn_dist


# ============================================================================
# Forward Accuracy Test
# ============================================================================

def run_fwd_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """
    Run single forward accuracy test - compare attn_dist
    """
    from sparse_mla_fwd import sparse_mla_fwd_interface
    
    torch.manual_seed(config.seed)
    
    # tilelang version uses fixed dim=576
    head_dim = 576
    
    # Generate random inputs
    q = torch.randn(config.batch_size, config.seq_len, config.num_heads, head_dim, 
                    device=device, dtype=torch.bfloat16) / 10
    k = torch.randn(config.batch_size, config.seq_len, 1, head_dim, 
                    device=device, dtype=torch.bfloat16) / 10
    
    q.clamp_(-10, 10)
    k.clamp_(-10, 10)
    
    # Compute chunk_offset
    chunk_offset = config.seq_len - config.chunk_size
    k_stride = 1
    
    # Generate indices for full sequence
    indices_full = torch.full((config.batch_size, config.seq_len, 1, config.topk), 
                              config.seq_len, dtype=torch.int32, device=device)
    for b in range(config.batch_size):
        for t in range(config.seq_len):
            max_valid_idx = min(max(1, ((t) // k_stride)), config.seq_len)
            i_i = torch.randperm(max_valid_idx, device=device)[:config.topk]
            indices_full[b, t, 0, : len(i_i)] = i_i.to(torch.int32)
    
    # Use sparse_mla_fwd to compute full LSE
    _, full_lse = sparse_mla_fwd_interface(q, k, indices_full)
    
    # Extract chunk-specific data
    q_chunk = q[:, chunk_offset:, :, :].contiguous()
    indices_chunk = indices_full[:, chunk_offset:, :, :].contiguous()
    lse_chunk = full_lse[:, chunk_offset:, :].contiguous()
    
    # TileLang kernel: compute attn_dist using pre-computed LSE
    tl_attn_dist = calc_attn_dist(q_chunk, k, indices_chunk, lse_chunk, chunk_offset, k_stride=k_stride)
    
    # PyTorch reference: compute attn_dist
    ref_attn_dist = ref_calc_attn_dist(q_chunk, k, indices_chunk, chunk_offset, k_stride=k_stride)
    
    # Compare results
    def calc_diff(a, b):
        abs_diff = torch.abs(a - b)
        max_diff = abs_diff.max().item()
        rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item() * 100
        return max_diff, rel_diff
    
    max_diff, rel_diff = calc_diff(ref_attn_dist, tl_attn_dist.to(ref_attn_dist.dtype))
    passed = rel_diff < 1e-3  # relative error < 0.001%
    
    return {
        'config': config,
        'ref_max': ref_attn_dist.abs().max().item(),
        'tl_max': tl_attn_dist.abs().max().item(),
        'max_diff': max_diff,
        'rel_diff': rel_diff,
        'passed': passed
    }


def test_fwd_accuracy(configs: List[TestConfig]):
    """
    Batch run forward accuracy tests - compare attn_dist
    """
    print("\n" + "=" * 110)
    print("Forward Accuracy Test (PyTorch attn_dist vs TileLang attn_dist with pre-computed LSE)")
    print("=" * 110)
    
    results = []
    for config in configs:
        try:
            result = run_fwd_accuracy_test(config)
            results.append(result)
        except Exception as e:
            print(f"Skip test {config.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'Name':<12} {'Config':<55} {'MaxDiff':<12} {'RelDiff(%)':<12} {'Pass':<6}")
    print("-" * 97)
    for r in results:
        print(f"{r['config'].name:<12} {str(r['config']):<55} "
              f"{r['max_diff']:<12.2e} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 97)
    print(f"Forward Test (attn_dist): {passed_count}/{len(results)} passed")
    
    return results


# ============================================================================
# Forward Performance Test
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
    """
    Forward performance test
    """
    import time
    from sparse_mla_fwd import sparse_mla_fwd_interface
    
    torch.manual_seed(seed)
    device = 'cuda'
    
    print("\n" + "=" * 80)
    print("Forward Performance Test (TileLang calc_attn_dist with pre-computed LSE)")
    print("=" * 80)
    print(f"Parameters: batch={batch_size}, heads={num_heads}, chunk={chunk_size}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print("=" * 80)
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16) / 10
    k = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16) / 10
    q.clamp_(-10, 10)
    k.clamp_(-10, 10)
    
    chunk_offset = seq_len - chunk_size
    k_stride = 1
    
    # Generate indices for full sequence
    indices_full = torch.full((batch_size, seq_len, 1, topk), seq_len, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for t in range(seq_len):
            max_valid_idx = min(max(1, ((t) // k_stride)), seq_len)
            i_i = torch.randperm(max_valid_idx, device=device)[:topk]
            indices_full[b, t, 0, : len(i_i)] = i_i.to(torch.int32)
    
    # Use sparse_mla_fwd to compute full LSE
    _, full_lse = sparse_mla_fwd_interface(q, k, indices_full)
    
    # Extract chunk-specific data
    q_chunk = q[:, chunk_offset:, :, :].contiguous()
    indices_chunk = indices_full[:, chunk_offset:, :, :].contiguous()
    lse_chunk = full_lse[:, chunk_offset:, :].contiguous()
    
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated() / (1024**3)
    
    # Forward only test (indexer_loss_fwd kernel only)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = calc_attn_dist(q_chunk, k, indices_chunk, lse_chunk, chunk_offset, k_stride=k_stride)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        _ = calc_attn_dist(q_chunk, k, indices_chunk, lse_chunk, chunk_offset, k_stride=k_stride)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start) / num_benchmark * 1000
    fwd_peak = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"\n>>> Performance (warmup={num_warmup}, iters={num_benchmark})")
    print(f"  Forward time (indexer_loss only): {fwd_time:.3f} ms")
    
    print(f"\n>>> Memory Peak")
    print(f"  Base memory:    {base_memory:.2f} GB")
    print(f"  Forward peak:   {fwd_peak:.2f} GB (increment: {fwd_peak - base_memory:.2f} GB)")
    
    # Compute TFLOPS
    flops_fwd = batch_size * chunk_size * head_dim * topk * 2 * num_heads
    print(f"\n>>> Compute Throughput")
    print(f"  Forward TFLOPS: {flops_fwd / (fwd_time * 1e-3) / 1e12:.2f}")
    
    # IO bandwidth
    io_bytes = batch_size * chunk_size * head_dim * topk * 2  # bf16
    print(f"  IO Bandwidth: {io_bytes / (fwd_time * 1e-3) / 1e12:.2f} TB/s")
    
    return {
        'fwd_time': fwd_time,
        'fwd_peak': fwd_peak,
    }


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main function for running accuracy and performance tests
    """
    parser = argparse.ArgumentParser(description='TileLang IndexerLoss Forward Kernel Tests')
    parser.add_argument('--test_accuracy', action='store_true', help='Run forward accuracy tests')
    parser.add_argument('--test_performance', action='store_true', help='Run forward performance tests')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=128, help='Number of attention heads')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Chunk size')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--head_dim', type=int, default=576, help='Head dimension')
    parser.add_argument('--topk', type=int, default=512, help='Top-k value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=10, help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    # Default test configurations (tilelang version uses fixed dim=576, topk must be multiple of 64)
    accuracy_configs = [
        TestConfig(name="Small", batch_size=1, num_heads=16, chunk_size=256, seq_len=512, head_dim=576, topk=128),
        TestConfig(name="Medium", batch_size=1, num_heads=64, chunk_size=512, seq_len=1024, head_dim=576, topk=256),
        TestConfig(name="Large", batch_size=1, num_heads=128, chunk_size=1024, seq_len=2048, head_dim=576, topk=512),
        TestConfig(name="MultiBatch", batch_size=2, num_heads=128, chunk_size=2048, seq_len=4096, head_dim=576, topk=2048),
        TestConfig(name="LargeTopK", batch_size=1, num_heads=128, chunk_size=512, seq_len=2048, head_dim=576, topk=1024),
        TestConfig(name="Production", batch_size=1, num_heads=128, chunk_size=8192, seq_len=131072, head_dim=576, topk=2048),
    ]
    
    # Performance test configurations for different scales
    performance_configs = [
        {"name": "Production", "batch_size": 1, "num_heads": 128, "chunk_size": 8192, "seq_len": 131072, "head_dim": 576, "topk": 2048},
    ]
    
    if args.test_accuracy:
        # Run forward accuracy tests
        test_fwd_accuracy(accuracy_configs)
        
    elif args.test_performance:
        # Run forward performance tests
        print("\n" + "=" * 100)
        print("Forward Performance Tests at Different Scales")
        print("=" * 100)
        
        all_results = []
        for config_dict in performance_configs:
            try:
                result = test_fwd_performance(
                    batch_size=config_dict['batch_size'],
                    num_heads=config_dict['num_heads'],
                    chunk_size=config_dict['chunk_size'],
                    seq_len=config_dict['seq_len'],
                    head_dim=config_dict['head_dim'],
                    topk=config_dict['topk'],
                    seed=args.seed,
                    num_warmup=args.warmup,
                    num_benchmark=args.iters,
                )
                result['name'] = config_dict['name']
                all_results.append(result)
            except Exception as e:
                print(f"Skip performance test {config_dict['name']}: {e}")
                continue
        
        # Print summary
        print("\n" + "=" * 100)
        print("Performance Test Summary")
        print("=" * 100)
        print(f"\n{'Name':<15} {'Config':<55} {'Time(ms)':<12} {'Peak(GB)':<12}")
        print("-" * 94)
        for r in all_results:
            config_str = (f"batch={r.get('batch_size', 1) if 'batch_size' in r else config_dict['batch_size']}, "
                         f"heads={r.get('num_heads', 128) if 'num_heads' in r else config_dict['num_heads']}, "
                         f"chunk={r.get('chunk_size', 1024) if 'chunk_size' in r else config_dict['chunk_size']}, "
                         f"seq={r.get('seq_len', 2048) if 'seq_len' in r else config_dict['seq_len']}, "
                         f"dim={r.get('head_dim', 576) if 'head_dim' in r else config_dict['head_dim']}, "
                         f"topk={r.get('topk', 512) if 'topk' in r else config_dict['topk']}")
            print(f"{r['name']:<15} {config_str:<55} {r['fwd_time']:<12.3f} {r['fwd_peak']:<12.2f}")
        
        print("-" * 94)
        print(f"Total tests: {len(all_results)}")
        
    else:
        # Default: Run both accuracy and performance tests
        print("=" * 80)
        print("Running Default Tests (Accuracy + Performance)")
        print("=" * 80)
        
        # Accuracy tests
        test_fwd_accuracy(accuracy_configs)
        
        # Performance tests
        print("\n")
        test_fwd_performance(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            chunk_size=args.chunk_size,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            topk=args.topk,
            seed=args.seed,
            num_warmup=args.warmup,
            num_benchmark=args.iters,
        )


if __name__ == "__main__":
    main()
