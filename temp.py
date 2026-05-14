#!/usr/bin/env python3
"""
Compressor 两种实现性能对比测试
==================================================
对比方案：
  1. torch  : Megatron/LoongForge 训练实现 (2 个独立 BF16 GEMM + 8+ 个 PyTorch kernel)
  2. vllm   : 1 个合并 BF16 GEMM + 1 个 Triton kernel（compress+norm+rope 全融合）

精度基线：torch 实现（使用 float32 中间计算）

测试配置：
  - seq_len  = 128K（131072）
  - compress_ratio = 4（CSA，overlap 模式）
  - head_dim = 512, rope_head_dim = 64, nope_head_dim = 448
  - hidden   = 7168
  - dtype    = bfloat16
  - 硬件     : SM100（Blackwell）

测试格式参考：
  DeepTraining/cuda_source/sparse_mla_bwd/tests/test_flash_mla_sparse_bwd.py
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ─── 全局配置 ─────────────────────────────────────────────────────────────────
SEQ_LEN  = 128 * 1024      # 131072
RATIO    = 4               # CSA compress ratio
COFF     = 2               # ratio==4 => overlap => coff=2
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM   # 448
HIDDEN   = 7168
PROJ_DIM = COFF * HEAD_DIM       # 1024  (coff * head_dim)
N_GROUPS = SEQ_LEN // RATIO      # 32768
DTYPE    = torch.bfloat16
EPS      = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────────────────────────────────────

def calc_diff(a: torch.Tensor, b: torch.Tensor):
    """计算最大绝对误差和平均相对误差。"""
    a, b = a.float(), b.float()
    abs_diff = (a - b).abs()
    max_diff = abs_diff.max().item()
    rel_diff = (abs_diff / (1e-4 + a.abs())).mean().item()
    return max_diff, rel_diff


def build_cos_sin(n_groups: int, rope_dim: int, ratio: int, device: torch.device):
    """
    为压缩后的位置构造 cos/sin。
    压缩后第 g 个 token 对应原始位置 g*ratio + (ratio-1)。
    返回 cos, sin 各 [n_groups, rope_dim//2] float32。
    """
    positions = torch.arange(n_groups, device=device).float() * ratio + (ratio - 1)
    half = rope_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(half, device=device).float() / half))
    theta = positions.unsqueeze(1) * inv_freq.unsqueeze(0)   # [n_groups, half]
    return theta.cos().float().contiguous(), theta.sin().float().contiguous()


# ═══════════════════════════════════════════════════════════════════════════════
# 实现 1：Torch baseline（Megatron / LoongForge 风格）
# ═══════════════════════════════════════════════════════════════════════════════

def _overlap_transform_torch(tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
    """
    Megatron Compressor._overlap_transform
    输入:  [n_groups, ratio, PROJ_DIM]
    输出:  [n_groups, 2*ratio, HEAD_DIM]
    """
    n, r, _ = tensor.shape
    d = HEAD_DIM
    out = tensor.new_full((n, 2 * r, d), fill_value)
    out[:, r:]   = tensor[:, :, d:]          # 当前组第二 coff -> slot [4:8]
    out[1:, :r]  = tensor[:-1, :, :d]        # 上一组第一 coff -> slot [0:4]
    return out


def _rmsnorm_torch(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps) * w).to(x.dtype)


def _rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    GPT-J style RoPE（interleaved pairs）作用于最后 ROPE_DIM 个维度。
    x:   [n_groups, HEAD_DIM]  bf16
    cos/sin: [n_groups, ROPE_DIM//2]  float32
    """
    x_nope = x[:, :NOPE_DIM]
    x_r    = x[:, NOPE_DIM:].float()              # [n_groups, 64]
    # interleaved pairs: (2i, 2i+1)
    x_even = x_r[:, 0::2]                         # [n_groups, 32]
    x_odd  = x_r[:, 1::2]
    new_even = x_even * cos - x_odd * sin
    new_odd  = x_even * sin + x_odd * cos
    # interleave back: [n_groups, 32, 2] -> [n_groups, 64]
    x_r_new = torch.stack([new_even, new_odd], dim=-1).flatten(-2)
    return torch.cat([x_nope, x_r_new.to(x.dtype)], dim=-1)


def torch_compressor(
    x:      torch.Tensor,   # [SEQ_LEN, HIDDEN]  bf16
    wkv:    torch.Tensor,   # [PROJ_DIM, HIDDEN]  bf16
    wgate:  torch.Tensor,   # [PROJ_DIM, HIDDEN]  bf16
    ape:    torch.Tensor,   # [RATIO, PROJ_DIM]   float32
    norm_w: torch.Tensor,   # [HEAD_DIM]          float32
    cos:    torch.Tensor,   # [N_GROUPS, ROPE_DIM//2]  float32
    sin:    torch.Tensor,   # [N_GROUPS, ROPE_DIM//2]  float32
) -> torch.Tensor:          # [N_GROUPS, HEAD_DIM]  bf16
    """Megatron Compressor.forward 的等价 PyTorch 实现（去掉 Megatron 框架依赖）"""
    sq = x.size(0)

    # 两个独立 BF16 GEMM
    kv   = F.linear(x, wkv)              # [sq, PROJ_DIM]  bf16
    gate = F.linear(x, wgate)            # [sq, PROJ_DIM]  bf16

    # 截断 & reshape
    cutoff = (sq // RATIO) * RATIO
    kv   = kv[:cutoff].float().view(N_GROUPS, RATIO, PROJ_DIM)
    gate = gate[:cutoff].float().view(N_GROUPS, RATIO, PROJ_DIM)

    # APE 加法
    gate = gate + ape.unsqueeze(0)        # [N_GROUPS, RATIO, PROJ_DIM]

    # Overlap transform
    kv   = _overlap_transform_torch(kv,   fill_value=0.0)
    gate = _overlap_transform_torch(gate, fill_value=float('-inf'))

    # Softmax + 加权求和
    kv = (kv * torch.softmax(gate, dim=1)).sum(dim=1)  # [N_GROUPS, HEAD_DIM]

    # RMSNorm
    kv = _rmsnorm_torch(kv, norm_w, EPS).to(DTYPE)

    # RoPE
    kv = _rope_torch(kv, cos, sin)

    return kv   # [N_GROUPS, HEAD_DIM]  bf16


# ═══════════════════════════════════════════════════════════════════════════════
# 实现 2：vllm 风格——1 个合并 GEMM + 1 个全融合 Triton kernel
#
# 核心逻辑来自：
#   vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py
#   函数 _fused_kv_compress_norm_rope_insert_sparse_attn（行 31-214）
#
# 训练适配：去掉 FP8 量化和 KV cache insert（这两部分为推理专用），
# 保留 compress → RMSNorm → GPT-J RoPE 全部在寄存器内完成（无中间全局内存写入）。
# GEMM：将 wkv 和 wgate 合并为 wkv_gate，使用 1 个大 GEMM。
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def vllm_fused_compress_norm_rope_kernel(
    kv_score_ptr,     # [SEQ_LEN, 2*PROJ_DIM]  float32
    ape_ptr,          # [RATIO, PROJ_DIM]       float32
    norm_w_ptr,       # [HEAD_DIM]              float32
    cos_ptr,          # [N_GROUPS, ROPE_DIM//2] float32
    sin_ptr,          # [N_GROUPS, ROPE_DIM//2] float32
    out_ptr,          # [N_GROUPS, HEAD_DIM]    bfloat16
    stride_tok,       # = 2*PROJ_DIM
    HEAD_DIM:  tl.constexpr,    # 512
    PROJ_DIM:  tl.constexpr,    # 1024
    NOPE_DIM:  tl.constexpr,    # 448
    ROPE_DIM:  tl.constexpr,    # 64
    RATIO:     tl.constexpr,    # 4
    eps:       tl.constexpr,    # 1e-6
):
    """
    vllm _fused_kv_compress_norm_rope_insert_sparse_attn 训练适配版本。
    每个 CTA 在寄存器中完成：
      overlap+softmax+pool（与 sglang kernel-1 相同逻辑）
      → RMSNorm（无全局内存写入）
      → GPT-J RoPE（无全局内存写入，参考 vllm 行 190-208）
      → 写出 bf16 结果

    相比 sglang 的 2-kernel 方案，节省一次 [N_GROUPS, HEAD_DIM] 的全局内存往返。
    """
    g = tl.program_id(0)
    d = tl.arange(0, HEAD_DIM)

    g_i64    = g.to(tl.int64)
    is_first = (g == 0)
    prev_g   = tl.maximum(g_i64 - 1, 0)

    # ── Step 1: online softmax + weighted sum ─────────────────────────────
    # 同 sglang kernel，用 -1e30 避免 exp(-inf - (-inf)) = NaN
    max_s = tl.full([HEAD_DIM], -1e30, dtype=tl.float32)
    sum_e = tl.zeros([HEAD_DIM], dtype=tl.float32)
    acc   = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for r in tl.static_range(RATIO):
        base_kv    = prev_g * (RATIO * stride_tok) + r * stride_tok
        base_score = base_kv + PROJ_DIM
        base_ape   = r * PROJ_DIM

        kv_v   = tl.load(kv_score_ptr + base_kv    + d, mask=~is_first, other=0.0)
        sc_raw = tl.load(kv_score_ptr + base_score  + d, mask=~is_first, other=float('-inf'))
        ape_v  = tl.load(ape_ptr + base_ape + d)
        sc_v   = tl.where(is_first, -1e30, sc_raw + ape_v)

        new_max = tl.maximum(max_s, sc_v)
        scale   = tl.exp(max_s - new_max)
        exp_v   = tl.exp(sc_v  - new_max)
        sum_e = sum_e * scale + exp_v
        acc   = acc   * scale + exp_v * kv_v
        max_s = new_max

    for r in tl.static_range(RATIO):
        base_kv    = g_i64 * (RATIO * stride_tok) + r * stride_tok + HEAD_DIM
        base_score = g_i64 * (RATIO * stride_tok) + r * stride_tok + PROJ_DIM + HEAD_DIM
        base_ape   = r * PROJ_DIM + HEAD_DIM

        kv_v  = tl.load(kv_score_ptr + base_kv    + d)
        sc_v  = tl.load(kv_score_ptr + base_score + d)
        ape_v = tl.load(ape_ptr + base_ape + d)
        sc_v  = sc_v + ape_v

        new_max = tl.maximum(max_s, sc_v)
        scale   = tl.exp(max_s - new_max)
        exp_v   = tl.exp(sc_v  - new_max)
        sum_e = sum_e * scale + exp_v
        acc   = acc   * scale + exp_v * kv_v
        max_s = new_max

    compressed = acc / sum_e     # [HEAD_DIM]  float32，在寄存器中

    # ── Step 2: RMSNorm（全程在寄存器中）────────────────────────────────
    w = tl.load(norm_w_ptr + d)
    variance = tl.sum(compressed * compressed, axis=0) / HEAD_DIM
    normed   = compressed * tl.rsqrt(variance + eps) * w   # [HEAD_DIM]  float32

    # 模拟基线在 RMSNorm 后的 BF16 cast（torch_compressor 中 .to(DTYPE) 再进 RoPE）
    normed = normed.to(tl.bfloat16).to(tl.float32)

    # ── Step 3: GPT-J RoPE（在寄存器中，参考 vllm 行 189-208）──────────
    NUM_PAIRS:  tl.constexpr = HEAD_DIM  // 2   # 256
    NOPE_PAIRS: tl.constexpr = NOPE_DIM  // 2   # 224
    HALF_ROPE:  tl.constexpr = ROPE_DIM  // 2   # 32

    pair_idx  = tl.arange(0, NUM_PAIRS)
    pair_2d   = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)

    rope_local = pair_idx - NOPE_PAIRS
    is_rope    = rope_local >= 0
    cs_idx     = tl.maximum(rope_local, 0)

    g_base = g_i64 * HALF_ROPE
    cos_v  = tl.load(cos_ptr + g_base + cs_idx, mask=is_rope, other=1.0)
    sin_v  = tl.load(sin_ptr + g_base + cs_idx, mask=is_rope, other=0.0)

    new_even = tl.where(is_rope, even * cos_v - odd * sin_v, even)
    new_odd  = tl.where(is_rope, even * sin_v + odd * cos_v, odd)

    result = tl.interleave(new_even, new_odd)    # [HEAD_DIM]

    tl.store(out_ptr + g_i64 * HEAD_DIM + d, result.to(tl.bfloat16))


def vllm_fused_fwd(
    kv_score: torch.Tensor,
    ape:      torch.Tensor,
    norm_w:   torch.Tensor,
    cos:      torch.Tensor,
    sin:      torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(N_GROUPS, HEAD_DIM, dtype=DTYPE, device=kv_score.device)
    vllm_fused_compress_norm_rope_kernel[(N_GROUPS,)](
        kv_score, ape, norm_w, cos, sin, out,
        stride_tok=2 * PROJ_DIM,
        HEAD_DIM=HEAD_DIM, PROJ_DIM=PROJ_DIM,
        NOPE_DIM=NOPE_DIM, ROPE_DIM=ROPE_DIM,
        RATIO=RATIO, eps=EPS,
    )
    return out


def vllm_compressor(
    x:        torch.Tensor,   # [SEQ_LEN, HIDDEN]  bf16
    wkv_gate: torch.Tensor,   # [2*PROJ_DIM, HIDDEN]  bf16  (wkv 和 wgate 合并)
    ape:      torch.Tensor,
    norm_w:   torch.Tensor,
    cos:      torch.Tensor,
    sin:      torch.Tensor,
) -> torch.Tensor:
    """vllm 风格：1 个合并 BF16 GEMM + cast FP32 + 1 个全融合 Triton kernel"""
    # 1 个合并 BF16 GEMM，输出 cast 到 FP32
    kv_score = F.linear(x, wkv_gate).float()   # BF16 GEMM → FP32
    kv_score = kv_score[:N_GROUPS * RATIO].contiguous()
    return vllm_fused_fwd(kv_score, ape, norm_w, cos, sin)


# ═══════════════════════════════════════════════════════════════════════════════
# 主测试函数
# ═══════════════════════════════════════════════════════════════════════════════

def test_compressor_perf(
    warmup:   int = 10,
    num_iter: int = 100,
    check_accuracy: bool = True,
):
    device = torch.device("cuda")
    torch.manual_seed(42)

    print(f"\n{'='*70}")
    print(f"  Compressor 性能对比测试")
    print(f"  seq_len={SEQ_LEN}, ratio={RATIO}, head_dim={HEAD_DIM}, "
          f"rope_dim={ROPE_DIM}, hidden={HIDDEN}")
    print(f"{'='*70}\n")

    # ── 构造权重和输入 ────────────────────────────────────────────────────
    x        = torch.randn(SEQ_LEN, HIDDEN, dtype=DTYPE, device=device)
    wkv      = torch.randn(PROJ_DIM, HIDDEN, dtype=DTYPE, device=device) * 0.02
    wgate    = torch.randn(PROJ_DIM, HIDDEN, dtype=DTYPE, device=device) * 0.02
    wkv_gate = torch.cat([wkv, wgate], dim=0)   # [2*PROJ_DIM, HIDDEN]  bf16
    ape      = torch.randn(RATIO, PROJ_DIM, dtype=torch.float32, device=device) * 0.1
    norm_w   = torch.ones(HEAD_DIM, dtype=torch.float32, device=device)
    cos, sin = build_cos_sin(N_GROUPS, ROPE_DIM, RATIO, device)

    # 预计算 float32 kv_score for post-GEMM benchmark（使用 FP32 GEMM，精度更高）
    with torch.no_grad():
        kv_f32   = F.linear(x.float(), wkv.float())
        gate_f32 = F.linear(x.float(), wgate.float())
        kv_score_f32 = torch.cat([kv_f32, gate_f32], dim=-1)[:N_GROUPS * RATIO].contiguous()

    # 预计算 BF16 GEMM → FP32 kv_score（与 torch_compressor 内部 GEMM 精度一致，用于精度验证）
    with torch.no_grad():
        kv_score_bf16gemm = torch.cat([
            F.linear(x, wkv).float(),
            F.linear(x, wgate).float(),
        ], dim=-1)[:N_GROUPS * RATIO].contiguous()

    print(f"  内存占用估算：")
    print(f"    x          : {x.numel()*2/1e9:.2f} GB  (bf16)")
    print(f"    kv_score   : {kv_score_f32.numel()*4/1e9:.2f} GB  (f32, post-GEMM)")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # 精度验证（以 torch 为基线）
    # ─────────────────────────────────────────────────────────────────────
    if check_accuracy:
        print("── 精度验证（基线 = torch 实现）─────────────────────────────────────")

        with torch.no_grad():
            ref = torch_compressor(x, wkv, wgate, ape, norm_w, cos, sin)

            # vllm：使用与 torch_compressor 相同的 BF16 GEMM 输出，隔离 GEMM 精度影响
            out_vl = vllm_fused_fwd(kv_score_bf16gemm, ape, norm_w, cos, sin)
            vl_max, vl_rel = calc_diff(ref, out_vl)
            print(f"  [torch vs vllm  ] max_diff={vl_max:.6f}, rel_diff={vl_rel:.6f}")

        print()

    # ─────────────────────────────────────────────────────────────────────
    # Benchmark helper
    # ─────────────────────────────────────────────────────────────────────
    def bench(fn, label):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(num_iter):
            fn()
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / num_iter
        print(f"  {label:40s}: {ms:.4f} ms")
        return ms

    # ─────────────────────────────────────────────────────────────────────
    # Benchmark A：仅 post-GEMM（compress + norm + rope）
    # ─────────────────────────────────────────────────────────────────────
    print("── Benchmark A: Post-GEMM（compress + norm + rope）─────────────────")

    with torch.no_grad():
        # torch post-GEMM（跳过 GEMM，使用预计算的 kv_f32/gate_f32）
        def torch_post_gemm():
            kv_v  = kv_f32.view(N_GROUPS, RATIO, PROJ_DIM)
            gt_v  = gate_f32.view(N_GROUPS, RATIO, PROJ_DIM)
            gt_v  = gt_v + ape.unsqueeze(0)
            kv_t  = _overlap_transform_torch(kv_v,  fill_value=0.0)
            gt_t  = _overlap_transform_torch(gt_v,  fill_value=float('-inf'))
            kv_o  = (kv_t * torch.softmax(gt_t, dim=1)).sum(dim=1)
            kv_o  = _rmsnorm_torch(kv_o, norm_w, EPS).to(DTYPE)
            kv_o  = _rope_torch(kv_o, cos, sin)
            return kv_o

        ms_torch_pg = bench(torch_post_gemm,
                            "torch (8+ kernels)")
        ms_vllm     = bench(
            lambda: vllm_fused_fwd(kv_score_f32, ape, norm_w, cos, sin),
            "vllm   (1 Triton kernel)")

    print(f"\n  vllm 相对 torch post-GEMM 加速: {ms_torch_pg / ms_vllm:.2f}x")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Benchmark B：完整 Compressor forward（GEMM + compress + norm + rope）
    # ─────────────────────────────────────────────────────────────────────
    print("── Benchmark B: 完整 forward（GEMM + compress + norm + rope）────────")

    with torch.no_grad():
        ms_torch_full = bench(
            lambda: torch_compressor(x, wkv, wgate, ape, norm_w, cos, sin),
            "torch  (2 GEMM + 8+ kernels)")
        ms_vllm_full  = bench(
            lambda: vllm_compressor(x, wkv_gate, ape, norm_w, cos, sin),
            "vllm   (1 GEMM + 1 Triton kernel)")

    print(f"\n  vllm 相对 torch 完整 forward 加速: {ms_torch_full / ms_vllm_full:.2f}x")
    print(f"  GEMM 时间（torch 2 GEMM）: {ms_torch_full - ms_torch_pg:.4f} ms")
    print(f"  GEMM 时间（vllm  1 GEMM）: {ms_vllm_full  - ms_vllm:.4f} ms")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # I/O bandwidth 分析
    # ─────────────────────────────────────────────────────────────────────
    print("── I/O 带宽分析（理论值）────────────────────────────────────────────")
    # Post-GEMM 的主要 IO：读 kv_score (f32) + 读 ape + 写 out (bf16)
    read_kv    = N_GROUPS * RATIO * 2 * PROJ_DIM * 4        # kv_score f32
    read_ape   = RATIO * PROJ_DIM * 4                        # ape f32
    write_out  = N_GROUPS * HEAD_DIM * 2                     # out bf16
    total_io   = read_kv + read_ape + write_out
    print(f"  Post-GEMM 理论 IO: {total_io / 1e9:.2f} GB")
    print(f"  torch post-GEMM:  {total_io / (ms_torch_pg * 1e-3) / 1e12:.2f} TB/s")
    print(f"  vllm   1-kernel:  {total_io / (ms_vllm     * 1e-3) / 1e12:.2f} TB/s")
    print()
    print("─" * 70)
    print("  完成。")


if __name__ == "__main__":
    test_compressor_perf(
        warmup=10,
        num_iter=100,
        check_accuracy=True,
    )

根据@test_compressor_perf.py ，完善@test_compressor_backward_perf.py  ，基线的反向直接使用torch的backward，优化代码的反向需要你实现一个triton的融合版本
