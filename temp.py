# ruff: noqa
"""
测试 ref_sparse_mla_bwd_interface (PyTorch 参考实现) 的耗时
前向和反向都使用切 chunk 实现避免 OOM
"""
import torch


def ref_sparse_mla_fwd_chunk(q_chunk, kv, indices_chunk, chunk_start, sm_scale, sk):
    """单个 chunk 的前向计算（支持 autograd）"""
    b, chunk_len, h, dim_q = q_chunk.shape
    b, _, g, _ = kv.shape

    dim = 512
    k = kv
    v = kv[..., :dim]

    _, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g

    # causal mask: 对于位置 chunk_start + i，只能看到 [0, chunk_start + i] 的位置
    chunk_positions = torch.arange(chunk_start, chunk_start + chunk_len, dtype=torch.int32, device=q_chunk.device).view(-1, 1)
    kv_positions = torch.arange(0, sk, dtype=torch.int32, device=q_chunk.device).view(1, -1)
    compressed_casual_mask = chunk_positions >= kv_positions  # (chunk_len, sk)

    # sparse mask from indices
    mask = q_chunk.new_zeros(b, g_index, chunk_len, sk + 1, dtype=torch.bool).scatter(3, indices_chunk.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, chunk_len, sk)
    mask = mask.view(b, g_index, 1, chunk_len, sk)

    q_view = q_chunk.view(b, chunk_len, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q_view, k)
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, chunk_len, sk)
    p = p.view(b, g, -1, chunk_len, sk)
    o_chunk = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o_chunk = o_chunk.reshape(b, chunk_len, h, dim_v)

    return o_chunk


def ref_sparse_mla_bwd_only(q, kv, do, indices, sm_scale=None, chunk_size=256):
    """
    只执行反向计算（前向在外部完成）
    返回 (dq, dkv, outputs_for_backward) 其中 outputs_for_backward 用于后续 backward
    """
    b, sq, h, dim_q = q.shape
    b, sk, g, dim_kv = kv.shape

    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale

    # indices 需要 transpose: (b, sq, g, topk) -> (b, g, sq, topk)
    indices_transposed = indices.transpose(1, 2)

    # 准备用于 backward 的数据
    chunks_data = []
    for chunk_start in range(0, sq, chunk_size):
        chunk_end = min(chunk_start + chunk_size, sq)

        q_chunk = q[:, chunk_start:chunk_end].detach().clone().float()
        kv_chunk = kv.detach().clone().float()
        q_chunk.requires_grad = True
        kv_chunk.requires_grad = True

        indices_chunk = indices_transposed[:, :, chunk_start:chunk_end]
        do_chunk = do[:, chunk_start:chunk_end].float()

        # 前向计算
        o_chunk = ref_sparse_mla_fwd_chunk(q_chunk, kv_chunk, indices_chunk, chunk_start, sm_scale, sk)

        chunks_data.append((q_chunk, kv_chunk, o_chunk, do_chunk, chunk_start))

    return chunks_data


def do_backward_only(chunks_data, dq, dkv):
    """只执行 backward 部分"""
    for q_chunk, kv_chunk, o_chunk, do_chunk, chunk_start in chunks_data:
        o_chunk.backward(do_chunk)
        chunk_end = chunk_start + q_chunk.shape[1]
        dq[:, chunk_start:chunk_end] = q_chunk.grad
        dkv += kv_chunk.grad


def do_bench_torch(fn, warmup=25, rep=100):
    """使用 CUDA events 进行精确的 GPU 计时"""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(rep):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    return sum(times) / len(times)


def test_ref_sparse_mla_bwd(
    B=1,
    S=4096,
    SKV=4096,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    chunk_size=256,
    warmup=25,
    rep=100,
):
    print("=" * 60)
    print("测试 ref_sparse_mla_bwd (PyTorch 参考实现) - 纯反向时间")
    print("=" * 60)
    print(f"参数配置:")
    print(f"  Batch Size (B): {B}")
    print(f"  Sequence Length (S): {S}")
    print(f"  KV Sequence Length (SKV): {SKV}")
    print(f"  Heads (H): {H}")
    print(f"  KV Heads (HKV): {HKV}")
    print(f"  QK Dim (DQK): {DQK}")
    print(f"  V Dim (DV): {DV}")
    print(f"  TopK: {topk}")
    print(f"  Dtype: {dtype}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Warmup: {warmup}, Rep: {rep}")
    print("-" * 60)

    torch.random.manual_seed(0)

    # 创建输入数据
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda")
    do = torch.randn((B, S, H, DV), dtype=dtype, device="cuda")

    # 创建 indices（向量化方式，避免慢速 for 循环）
    t_vals = torch.arange(S, device="cuda", dtype=torch.float32).view(1, S, 1, 1)
    max_vals = torch.clamp(t_vals, min=1).expand(B, S, HKV, topk)
    random_floats = torch.rand(B, S, HKV, topk, device="cuda")
    indices = (random_floats * max_vals).to(torch.int32)

    print(f"输入张量形状:")
    print(f"  q: {q.shape}")
    print(f"  kv: {kv.shape}")
    print(f"  do: {do.shape}")
    print(f"  indices: {indices.shape}")
    print("-" * 60)

    sm_scale = DQK**-0.5

    # 定义测试函数：每次迭代先前向（不计时），再只测 backward
    def fn():
        # 输出梯度张量
        dq = torch.zeros(B, S, H, DQK, dtype=torch.float32, device=q.device)
        dkv = torch.zeros(B, SKV, HKV, DQK, dtype=torch.float32, device=kv.device)

        # 前向（不计时）- 在 benchmark 循环外部执行
        chunks_data = ref_sparse_mla_bwd_only(q, kv, do, indices, sm_scale, chunk_size)
        torch.cuda.synchronize()

        # 只测 backward
        do_backward_only(chunks_data, dq, dkv)
        return dq, dkv

    # 由于需要分离前向和反向时间，使用自定义 benchmark
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark - 只测 backward 时间
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(rep):
        # 输出梯度张量
        dq = torch.zeros(B, S, H, DQK, dtype=torch.float32, device=q.device)
        dkv = torch.zeros(B, SKV, HKV, DQK, dtype=torch.float32, device=kv.device)

        # 前向（不计时）
        chunks_data = ref_sparse_mla_bwd_only(q, kv, do, indices, sm_scale, chunk_size)
        torch.cuda.synchronize()

        # 只测 backward 时间
        start_event.record()
        do_backward_only(chunks_data, dq, dkv)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    ms = sum(times) / len(times)

    # 计算性能指标（只计算反向的 FLOPS）
    per_token_flop = 2 * sum([
        H * DV * topk,      # dO @ V^T -> dP
        H * DQK * topk,     # dP @ K -> dQ
        H * DQK * topk,     # dP^T @ Q -> dK
        H * DV * topk,      # P^T @ dO -> dV
    ])
    io_bandwidth = (B * S * max(DQK * 2, DQK + DV) * topk * 2) / (ms * 1e-3) / 1e12  # TB/s
    tflops = (per_token_flop * S * B) / (ms * 1e-3) / 1e12  # TFLOPS

    print(f"结果 (纯反向时间，不含前向):")
    print(f"  平均耗时: {ms:.3f} ms")
    print(f"  IO 带宽: {io_bandwidth:.3f} TB/s")
    print(f"  计算吞吐: {tflops:.3f} TFLOPS")
    print("=" * 60)

    return ms


if __name__ == "__main__":
    test_ref_sparse_mla_bwd(
        B=1,
        S=4096,
        SKV=4096,
        H=128,
        HKV=1,
        DQK=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        chunk_size=256,  # 调小可减少显存使用，调大可能提升性能
        warmup=25,
        rep=100,
    )
