#!/usr/bin/env python3
"""
Test script for mla_bwd kernel.
Validates dQ and dKV precision against a PyTorch reference for sequence length > 1.
"""

import torch
import mla_bwd_cuda
import flash_mla


def calc_diff(a: torch.Tensor, b: torch.Tensor):
    abs_diff = torch.abs(a - b)
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
    return max_diff, rel_diff


def build_reference(
    q: torch.Tensor,        # [S, B_H, D_Q]
    kv: torch.Tensor,       # [S_KV, D_K]
    dO: torch.Tensor,       # [S, B_H, D_V]
    lse: torch.Tensor,      # [S, B_H]
    O: torch.Tensor,        # [S, B_H, D_V]
    indices: torch.Tensor,  # [S, B_TOPK]
    chunk_s: int = 8,
):
    S, B_H, D_Q = q.shape
    S_KV, D_K = kv.shape
    B_TOPK = indices.shape[1]
    D_V = dO.shape[2]
    D_ROPE = D_Q - D_V

    all_k = kv.float()
    all_v = kv[:, :D_V].float()
    safe_indices = indices.clamp(0, S_KV - 1).to(torch.long)
    in_range_mask = (indices >= 0) & (indices < S_KV)

    sm_scale = 1.0 / (D_Q ** 0.5)
    scale_log2e = sm_scale * 1.44269504
    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)

    dQ_ref = torch.empty((S, B_H, D_Q), device=q.device, dtype=torch.float32)
    dKV_ref = torch.zeros((S_KV, D_K), device=q.device, dtype=torch.float32)

    for s0 in range(0, S, chunk_s):
        s1 = min(s0 + chunk_s, S)
        cs = s1 - s0

        q_chunk = q[s0:s1].float()
        dO_chunk = dO[s0:s1].float()
        O_chunk = O[s0:s1].float()
        lse_chunk = lse[s0:s1]
        indices_chunk = safe_indices[s0:s1]
        in_range_chunk = in_range_mask[s0:s1]
        q_pos = torch.arange(s0, s1, device=q.device, dtype=torch.int32).unsqueeze(1)
        causal_chunk = indices[s0:s1] <= q_pos
        valid_chunk = in_range_chunk & causal_chunk

        gather_idx = indices_chunk.reshape(-1)
        k = all_k.index_select(0, gather_idx).view(cs, B_TOPK, D_K)
        v = all_v.index_select(0, gather_idx).view(cs, B_TOPK, D_V)

        P_ref = torch.matmul(q_chunk, k.transpose(-2, -1))
        P_ref = torch.where(valid_chunk.unsqueeze(1), P_ref, neg_inf)
        dP_ref = torch.matmul(dO_chunk, v.transpose(-2, -1))
        delta_ref = (O_chunk * dO_chunk).sum(dim=-1)

        s_ref = torch.exp2(P_ref * scale_log2e - lse_chunk.unsqueeze(-1))
        ds_ref = s_ref * (dP_ref - delta_ref.unsqueeze(-1)) * sm_scale
        ds_bf16 = ds_ref.to(torch.bfloat16)

        dQ_ref[s0:s1] = torch.matmul(ds_bf16.float(), k)

        dV_ref = torch.matmul(s_ref.to(torch.bfloat16).float().transpose(-2, -1), dO_chunk)
        dK_nope_ref = torch.matmul(ds_bf16.float().transpose(-2, -1), q_chunk[:, :, :D_V])
        dK_rope_ref = torch.matmul(ds_bf16.float().transpose(-2, -1), q_chunk[:, :, D_V:])
        dKV_topk_ref = torch.cat([dV_ref + dK_nope_ref, dK_rope_ref], dim=-1)

        if valid_chunk.all():
            dKV_ref.index_add_(0, indices_chunk.reshape(-1), dKV_topk_ref.reshape(-1, D_K))
        else:
            valid_indices = indices_chunk[valid_chunk]
            valid_dkv = dKV_topk_ref[valid_chunk]
            if valid_indices.numel() > 0:
                dKV_ref.index_add_(0, valid_indices, valid_dkv)

    assert dQ_ref.shape == (S, B_H, D_Q)
    assert dKV_ref.shape == (S_KV, D_K)
    assert D_ROPE == 64
    return dQ_ref, dKV_ref


def run_one_case(
    name: str,
    q: torch.Tensor,
    kv: torch.Tensor,
    dO: torch.Tensor,
    lse: torch.Tensor,
    O: torch.Tensor,
    indices: torch.Tensor,
):
    dQ_ref, dKV_ref = build_reference(q, kv, dO, lse, O, indices)
    dQ_cuda, dKV_cuda = mla_bwd_cuda.mla_bwd(q, kv, dO, lse, O, indices)

    D_Q = q.shape[2]
    sm_scale = 1.0 / (D_Q ** 0.5)
    # flash_mla.flash_mla_sparse_bwd expects:
    #   kv: [s_kv, h_kv, d_qk] (3D)
    #   indices: [s_q, h_kv, topk] (3D)
    kv_3d = kv.unsqueeze(1)  # [S_KV, D_K] -> [S_KV, 1, D_K]
    indices_3d = indices.unsqueeze(1)  # [S, B_TOPK] -> [S, 1, B_TOPK]
    dQ_cuda_flash, dKV_cuda_3d_flash = flash_mla.flash_mla_sparse_bwd(
        q, kv_3d, O, dO, indices_3d, lse / 1.44269504,
        # q, kv_3d, O, dO, indices_3d, lse,
        sm_scale=sm_scale,
    )
    dKV_cuda_flash = dKV_cuda_3d_flash.squeeze(1)  # [S_KV, 1, D_K] -> [S_KV, D_K]
    torch.cuda.synchronize()

    dQ_max_diff, dQ_rel_diff = calc_diff(dQ_cuda, dQ_ref.bfloat16())
    dKV_max_diff, dKV_rel_diff = calc_diff(dKV_cuda, dKV_ref)

    # Compare flash_mla results with reference (by dimension ranges)
    # Range 0-256
    dQ_flash_max_diff_0_256, dQ_flash_rel_diff_0_256 = calc_diff(dQ_cuda_flash[:, :, :256], dQ_ref[:, :, :256].bfloat16())
    dKV_flash_max_diff_0_256, dKV_flash_rel_diff_0_256 = calc_diff(dKV_cuda_flash[:, :256], dKV_ref[:, :256].bfloat16())
    # Range 256-512
    dQ_flash_max_diff_256_512, dQ_flash_rel_diff_256_512 = calc_diff(dQ_cuda_flash[:, :, 256:512], dQ_ref[:, :, 256:512].bfloat16())
    dKV_flash_max_diff_256_512, dKV_flash_rel_diff_256_512 = calc_diff(dKV_cuda_flash[:, 256:512], dKV_ref[:, 256:512].bfloat16())
    # Range 512-576
    dQ_flash_max_diff_512_576, dQ_flash_rel_diff_512_576 = calc_diff(dQ_cuda_flash[:, :, 512:576], dQ_ref[:, :, 512:576].bfloat16())
    dKV_flash_max_diff_512_576, dKV_flash_rel_diff_512_576 = calc_diff(dKV_cuda_flash[:, 512:576], dKV_ref[:, 512:576].bfloat16())
    # Full range
    dQ_flash_max_diff, dQ_flash_rel_diff = calc_diff(dQ_cuda_flash, dQ_ref.bfloat16())
    dKV_flash_max_diff, dKV_flash_rel_diff = calc_diff(dKV_cuda_flash, dKV_ref.bfloat16())

    print(f"{name}:")
    print(f"  dQ       max_diff={dQ_max_diff:.6e}, rel_diff={dQ_rel_diff:.6e}")
    print(f"  dKV      max_diff={dKV_max_diff:.6e}, rel_diff={dKV_rel_diff:.6e}")
    print(f"  dQ_flash [0:256]   max_diff={dQ_flash_max_diff_0_256:.6e}, rel_diff={dQ_flash_rel_diff_0_256:.6e}")
    print(f"  dKV_flash[0:256]   max_diff={dKV_flash_max_diff_0_256:.6e}, rel_diff={dKV_flash_rel_diff_0_256:.6e}")
    print(f"  dQ_flash [256:512] max_diff={dQ_flash_max_diff_256_512:.6e}, rel_diff={dQ_flash_rel_diff_256_512:.6e}")
    print(f"  dKV_flash[256:512] max_diff={dKV_flash_max_diff_256_512:.6e}, rel_diff={dKV_flash_rel_diff_256_512:.6e}")
    print(f"  dQ_flash [512:576] max_diff={dQ_flash_max_diff_512_576:.6e}, rel_diff={dQ_flash_rel_diff_512_576:.6e}")
    print(f"  dKV_flash[512:576] max_diff={dKV_flash_max_diff_512_576:.6e}, rel_diff={dKV_flash_rel_diff_512_576:.6e}")
    print(f"  dQ_flash [full]    max_diff={dQ_flash_max_diff:.6e}, rel_diff={dQ_flash_rel_diff:.6e}")
    print(f"  dKV_flash[full]    max_diff={dKV_flash_max_diff:.6e}, rel_diff={dKV_flash_rel_diff:.6e}")
    return dQ_max_diff, dQ_rel_diff, dKV_max_diff, dKV_rel_diff


def bench_cuda_kernel_tflops(
    q: torch.Tensor,
    kv: torch.Tensor,
    dO: torch.Tensor,
    lse: torch.Tensor,
    O: torch.Tensor,
    indices: torch.Tensor,
):
    # Reference: tilelang/examples/deepseek_v32/sparse_mla_bwd.py per_token_flop
    S, H, D_Q = q.shape
    topk = indices.shape[1]
    D_V = dO.shape[2]
    per_token_flop = 2 * sum(
        [
            H * D_V * topk,
            H * D_Q * topk,
            H * D_Q * topk,
            H * D_Q * topk,
            H * D_V * topk,
        ]
    )

    from tilelang.profiler import do_bench

    def fn():
        mla_bwd_cuda.mla_bwd(q, kv, dO, lse, O, indices)

    ms = do_bench(fn, rep=50, warmup=10)
    tflops = per_token_flop * S / (ms * 1e-3) / 1e12
    print(f"cuda_kernel average time: {ms:.3f} ms")
    print(f"cuda_kernel bwd tflops: {tflops:.3f}")
    return ms, tflops


def test_mla_bwd():
    print("=" * 60)
    print("Testing mla_bwd kernel (dQ/dKV, sequence length > 1)")
    print("=" * 60)

    S = 4096
    B_H = 128
    D_Q = 576
    S_KV = 8192
    B_TOPK = 2048
    D_K = 576
    D_V = 512

    print(f"q shape: [{S}, {B_H}, {D_Q}]")
    print(f"kv shape: [{S_KV}, {D_K}]")
    print(f"dO shape: [{S}, {B_H}, {D_V}]")
    print(f"lse shape: [{S}, {B_H}]")
    print(f"O shape: [{S}, {B_H}, {D_V}]")
    print(f"indices shape: [{S}, {B_TOPK}]")
    print(f"dQ shape: [{S}, {B_H}, {D_Q}]")
    print(f"dKV shape: [{S_KV}, {D_K}]")
    print()

    torch.manual_seed(42)
    q = torch.randn(S, B_H, D_Q, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(S_KV, D_K, dtype=torch.bfloat16, device="cuda")
    dO = torch.randn(S, B_H, D_V, dtype=torch.bfloat16, device="cuda")
    lse = torch.randn(S, B_H, dtype=torch.float32, device="cuda") * 2.0 + 5.0
    O = torch.randn(S, B_H, D_V, dtype=torch.bfloat16, device="cuda")

    # Match tilelang sparse_mla_bwd.py style: valid causal prefix + SKV sentinel invalid tail.
    indices = torch.full((S, B_TOPK), S_KV, dtype=torch.int32, device="cuda")
    for t in range(S):
        n_valid = min(t + 1, B_TOPK)
        if n_valid > 0:
            idx = torch.randperm(t + 1, device="cuda")[:n_valid].to(torch.int32)
            indices[t, :n_valid] = idx

    dQ_max_diff, dQ_rel_diff, dKV_max_diff, dKV_rel_diff = run_one_case(
        "Random normal", q, kv, dO, lse, O, indices
    )

    rel_diff_threshold = 0.01
    ok_dQ = dQ_rel_diff < rel_diff_threshold
    ok_dKV = dKV_rel_diff < rel_diff_threshold

    print("=" * 60)
    print("Precision Validation (dQ/dKV)")
    print("=" * 60)
    print(f"{'PASS' if ok_dQ else 'FAIL'} dQ  rel_diff={dQ_rel_diff:.6e}, threshold={rel_diff_threshold:.2e}")
    print(f"{'PASS' if ok_dKV else 'FAIL'} dKV rel_diff={dKV_rel_diff:.6e}, threshold={rel_diff_threshold:.2e}")
    print(f"dQ  max_diff={dQ_max_diff:.6e}")
    print(f"dKV max_diff={dKV_max_diff:.6e}")

    print("=" * 60)
    print("CUDA Kernel Benchmark")
    print("=" * 60)
    bench_cuda_kernel_tflops(q, kv, dO, lse, O, indices)
    return ok_dQ and ok_dKV


if __name__ == "__main__":
    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())
    print()

    passed = test_mla_bwd()

    print("\n" + "=" * 60)
    print("All tests PASSED!" if passed else "Some tests FAILED!")
    print("=" * 60)
    raise SystemExit(0 if passed else 1)
