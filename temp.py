# ruff: noqa
"""
测试 ref_sparse_mla_bwd_interface (PyTorch 参考实现) 的耗时
"""
import torch


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    """PyTorch 前向参考实现"""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device=q.device).view(-1, 1) >= torch.arange(
        1 - 1, sk * 1, 1, dtype=torch.int32, device=q.device
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : 1 - 1, 0] = True
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


def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, sm_scale=None, is_casual=True):
    """PyTorch 反向参考实现 - 使用 autograd"""
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale, is_casual)
    o.backward(do)
    return q.grad, kv.grad


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
    warmup=25,
    rep=100,
):
    print("=" * 60)
    print("测试 ref_sparse_mla_bwd_interface (PyTorch 参考实现)")
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

    # 定义测试函数
    def fn():
        return ref_sparse_mla_bwd_interface(q, kv, None, do, indices, None)

    # 测量耗时
    ms = do_bench_torch(fn, warmup=warmup, rep=rep)

    # 计算性能指标
    # 反向计算量约为前向的 2-3 倍
    per_token_flop = 2 * sum([
        H * DV * topk,      # dO @ V^T
        H * DQK * topk,     # dP @ K
        H * DQK * topk,     # dP^T @ Q
        H * DQK * topk,     # P^T @ dO
        H * DV * topk,      # Q @ K^T (前向)
    ])
    io_bandwidth = (B * S * max(DQK * 2, DQK + DV) * topk * 2) / (ms * 1e-3) / 1e12  # TB/s
    tflops = (per_token_flop * S * B) / (ms * 1e-3) / 1e12  # TFLOPS

    print(f"结果:")
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
        warmup=25,
        rep=100,
    )
