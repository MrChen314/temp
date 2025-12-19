# ruff: noqa
"""
测试 ref_sparse_mla_fwd_interface (PyTorch 参考实现) 的耗时
"""
import torch


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, chunk_size=256):
    """PyTorch 参考实现 - 分 chunk 计算避免 OOM"""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)  # (b, g, sq, topk)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale

    # 输出张量
    output = torch.zeros(b, sq, h, dim_v, dtype=q.dtype, device=q.device)

    # 分 chunk 处理，避免大序列 OOM
    for chunk_start in range(0, sq, chunk_size):
        chunk_end = min(chunk_start + chunk_size, sq)
        chunk_len = chunk_end - chunk_start

        # 当前 chunk 的 q 和 indices
        q_chunk = q[:, chunk_start:chunk_end]  # (b, chunk_len, h, dim_q)
        indices_chunk = indices[:, :, chunk_start:chunk_end]  # (b, g, chunk_len, topk)

        # causal mask: 对于位置 chunk_start + i，只能看到 [0, chunk_start + i] 的位置
        chunk_positions = torch.arange(chunk_start, chunk_end, dtype=torch.int32, device=q.device).view(-1, 1)
        kv_positions = torch.arange(0, sk, dtype=torch.int32, device=q.device).view(1, -1)
        compressed_casual_mask = chunk_positions >= kv_positions  # (chunk_len, sk)

        # sparse mask from indices
        mask = q_chunk.new_zeros(b, g_index, chunk_len, sk + 1, dtype=torch.bool).scatter(3, indices_chunk.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_casual_mask.view(1, 1, chunk_len, sk)
        mask = mask.view(b, g_index, 1, chunk_len, sk)

        q_chunk = q_chunk.view(b, chunk_len, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q_chunk, k)
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, chunk_len, sk)
        p = p.view(b, g, -1, chunk_len, sk)
        o_chunk = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o_chunk = o_chunk.reshape(b, chunk_len, h, dim_v)

        output[:, chunk_start:chunk_end] = o_chunk

    return output.to(torch.bfloat16)


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


def test_ref_sparse_mla_fwd(
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
    print("测试 ref_sparse_mla_fwd_interface (PyTorch 参考实现)")
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

    # 创建 indices（向量化方式，避免慢速 for 循环）
    # 对于位置 t，随机选择 [0, max(1, t)) 范围内的 topk 个索引
    t_vals = torch.arange(S, device="cuda", dtype=torch.float32).view(1, S, 1, 1)
    max_vals = torch.clamp(t_vals, min=1).expand(B, S, HKV, topk)
    random_floats = torch.rand(B, S, HKV, topk, device="cuda")
    indices = (random_floats * max_vals).to(torch.int32)

    print(f"输入张量形状:")
    print(f"  q: {q.shape}")
    print(f"  kv: {kv.shape}")
    print(f"  indices: {indices.shape}")
    print("-" * 60)

    # 定义测试函数
    def fn():
        return ref_sparse_mla_fwd_interface(q, kv, indices, chunk_size=chunk_size)

    # 测量耗时
    ms = do_bench_torch(fn, warmup=warmup, rep=rep)

    # 计算性能指标
    io_bandwidth = (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12  # TB/s
    tflops = (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12  # TFLOPS

    print(f"结果:")
    print(f"  平均耗时: {ms:.3f} ms")
    print(f"  IO 带宽: {io_bandwidth:.3f} TB/s")
    print(f"  计算吞吐: {tflops:.3f} TFLOPS")
    print("=" * 60)

    return ms


if __name__ == "__main__":
    test_ref_sparse_mla_fwd(
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
