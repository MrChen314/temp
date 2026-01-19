# ruff: noqa
"""
测试 flash_mla_sparse_backward 与 ref_sparse_mla_bwd_interface 的精度对比
以及 flash_mla_sparse_backward 的性能测试
"""
import torch
import sys
sys.path.insert(0, "/Users/chenql/Desktop/workspace/operator/FlashMLA")
from flash_mla.flash_mla_interface import flash_mla_sparse_fwd, flash_mla_sparse_backward


# ==================== 参考实现：ref_sparse_mla_fwd_interface ====================
# 从 tilelang/examples/deepseek_v32/sparse_mla_fwd.py 复制
def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    """
    参考实现的稀疏 MLA 前向传播
    
    Args:
        q: [B, S, H, D], float32/bfloat16
        kv: [B, SKV, HKV, D], float32/bfloat16
        indices: [B, S, HKV, topk], int32
        sm_scale: softmax 缩放因子
        is_casual: 是否使用 causal mask
    
    Returns:
        o: [B, S, H, DV], bfloat16
    """
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)  # [B, HKV, S, topk]
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
        1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda"
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


# ==================== 参考实现：ref_sparse_mla_bwd_interface ====================
# 从 tilelang/examples/deepseek_v32/sparse_mla_bwd.py 复制并修改
def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, sm_scale=None, is_casual=True):
    """
    参考实现的稀疏 MLA 反向传播（使用 autograd）
    
    Args:
        q: [B, S, H, D], bfloat16
        kv: [B, SKV, HKV, D], bfloat16
        o: 未使用（通过前向重新计算）
        do: [B, S, H, DV], bfloat16
        indices: [B, S, HKV, topk], int32
        lse: 未使用
        sm_scale: softmax 缩放因子
        is_casual: 是否使用 causal mask
    
    Returns:
        (dq, dkv)
        - dq: [B, S, H, D], grad of q
        - dkv: [B, SKV, HKV, D], grad of kv
    """
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale, is_casual)
    o.backward(do)
    return q.grad, kv.grad


# ==================== 工具函数 ====================
def assert_tensors_similar(a, b, eps=1e-2, name="tensor"):
    """比较两个张量的相似度"""
    a = a.float()
    b = b.float()
    
    # 计算相对误差
    abs_diff = (a - b).abs()
    max_abs = torch.maximum(a.abs(), b.abs()).clamp(min=1e-8)
    rel_diff = abs_diff / max_abs
    
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"  {name}: max_rel_diff={max_rel_diff:.6f}, mean_rel_diff={mean_rel_diff:.6f}")
    
    if max_rel_diff > eps:
        print(f"  WARNING: {name} 相对误差超过阈值 {eps}")
        return False
    return True


# ==================== 精度测试 ====================
def test_accuracy(
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQKV=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
):
    """
    精度对比测试：比较 flash_mla_sparse_backward 和 ref_sparse_mla_bwd_interface
    """
    print(f"\n{'='*60}")
    print(f"精度测试: S={S}, SKV={SKV}, H={H}, HKV={HKV}, DQKV={DQKV}, DV={DV}, topk={topk}")
    print(f"{'='*60}")
    
    torch.random.manual_seed(42)
    
    # 生成输入数据（4D 格式用于 ref 函数）
    B = 1
    q_4d = torch.randn((B, S, H, DQKV), dtype=dtype, device="cuda")
    kv_4d = torch.randn((B, SKV, HKV, DQKV), dtype=dtype, device="cuda")
    do_4d = torch.randn((B, S, H, DV), dtype=dtype, device="cuda")
    
    # 生成稀疏索引
    indices_4d = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices_4d[b, t, h, : len(i_i)] = i_i
    
    # 转换为 3D 格式用于 flash_mla 函数
    q_3d = q_4d.squeeze(0).contiguous()  # [S, H, DQKV]
    kv_3d = kv_4d.squeeze(0).contiguous()  # [SKV, HKV, DQKV]
    do_3d = do_4d.squeeze(0).contiguous()  # [S, H, DV]
    indices_3d = indices_4d.squeeze(0).contiguous()  # [S, HKV, topk]
    
    sm_scale = DQKV ** (-0.5)
    
    # ========== 使用 flash_mla_sparse_fwd 获取前向输出 ==========
    print("\n1. 运行 flash_mla_sparse_fwd 获取前向输出...")
    out_flash, max_logits, lse = flash_mla_sparse_fwd(
        q_3d, kv_3d, indices_3d, sm_scale, DV
    )
    print(f"   out_flash shape: {out_flash.shape}, lse shape: {lse.shape}")
    
    # ========== 计算 flash_mla_sparse_backward ==========
    print("\n2. 运行 flash_mla_sparse_backward...")
    flash_dq, flash_dkv = flash_mla_sparse_backward(
        do_3d, q_3d, kv_3d, out_flash, indices_3d, lse, sm_scale, DV
    )
    print(f"   flash_dq shape: {flash_dq.shape}, flash_dkv shape: {flash_dkv.shape}")
    
    # ========== 计算 ref_sparse_mla_bwd_interface ==========
    print("\n3. 运行 ref_sparse_mla_bwd_interface...")
    ref_dq, ref_dkv = ref_sparse_mla_bwd_interface(
        q_4d, kv_4d, None, do_4d, indices_4d, None, sm_scale, is_casual=True
    )
    print(f"   ref_dq shape: {ref_dq.shape}, ref_dkv shape: {ref_dkv.shape}")
    
    # ========== 精度对比 ==========
    print("\n4. 精度对比:")
    
    # 转换 ref 结果为 3D 格式进行比较
    ref_dq_3d = ref_dq.squeeze(0)  # [S, H, DQKV]
    ref_dkv_3d = ref_dkv.squeeze(0)  # [SKV, HKV, DQKV]
    
    # 对于 flash_dkv 是 float32，ref_dkv 是 bfloat16/float32
    dq_passed = assert_tensors_similar(flash_dq, ref_dq_3d, eps=1e-2, name="dq")
    dkv_passed = assert_tensors_similar(flash_dkv, ref_dkv_3d, eps=1e-2, name="dkv")
    
    if dq_passed and dkv_passed:
        print("\n精度测试通过!")
    else:
        print("\n精度测试失败!")
    
    return dq_passed and dkv_passed


# ==================== 性能测试 ====================
def test_performance(
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQKV=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    warmup=250,
    rep=100,
):
    """
    性能测试：测量 flash_mla_sparse_backward 的 tflops
    """
    print(f"\n{'='*60}")
    print(f"性能测试: S={S}, SKV={SKV}, H={H}, HKV={HKV}, DQKV={DQKV}, DV={DV}, topk={topk}")
    print(f"{'='*60}")
    
    torch.random.manual_seed(42)
    
    # 生成输入数据（3D 格式）
    q = torch.randn((S, H, DQKV), dtype=dtype, device="cuda").contiguous()
    kv = torch.randn((SKV, HKV, DQKV), dtype=dtype, device="cuda").contiguous()
    do = torch.randn((S, H, DV), dtype=dtype, device="cuda").contiguous()
    
    # 生成稀疏索引
    indices = torch.full((S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for t in range(S):
        for h in range(HKV):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i
    indices = indices.contiguous()
    
    sm_scale = DQKV ** (-0.5)
    
    # 获取前向输出
    print("\n1. 运行 flash_mla_sparse_fwd 获取前向输出...")
    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, indices, sm_scale, DV)
    
    # 定义 benchmark 函数
    def fn():
        return flash_mla_sparse_backward(do, q, kv, out, indices, lse, sm_scale, DV)
    
    # 使用 triton 的 do_bench 或手动测量
    print("\n2. 运行性能测试...")
    try:
        from triton.testing import do_bench
        ms = do_bench(fn, warmup=warmup, rep=rep)
    except ImportError:
        # 手动测量
        print("   未找到 triton，使用手动测量...")
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        
        # Benchmark
        import time
        start = time.perf_counter()
        for _ in range(rep):
            fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        ms = (end - start) / rep * 1000
    
    # 计算 tflops
    # 参考 tilelang/examples/deepseek_v32/sparse_mla_bwd.py:316-333
    per_token_flop = 2 * sum([
        H * DV * topk,      # dO @ V^T (计算 dP 的一部分)
        H * DQKV * topk,    # dP @ K (计算 dQ)
        H * DQKV * topk,    # dP^T @ Q (计算 dK)
        H * DQKV * topk,    # P^T @ dO (计算 dV 的一部分)
        H * DV * topk,      # Q @ K^T (重计算 attention scores)
    ])
    
    bwd_tflops = per_token_flop * S / (ms * 1e-3) / 1e12
    
    print(f"\n3. 结果:")
    print(f"   Average time: {ms:.3f} ms")
    print(f"   bwd tflops = {bwd_tflops:.2f}")
    
    return ms, bwd_tflops


# ==================== Main ====================
if __name__ == "__main__":
    # 精度测试
    test_accuracy(
        S=4096,
        SKV=4096,
        H=128,
        HKV=1,
        DQKV=576,
        DV=512,
        topk=2048,
    )
    
    # 性能测试
    test_performance(
        S=4096,
        SKV=4096,
        H=128,
        HKV=1,
        DQKV=576,
        DV=512,
        topk=2048,
    )
