# ruff: noqa
"""
Sparse MLA Forward 调试和性能分析脚本

使用 TileLang 的调试工具进行深入分析:
1. 查看生成的 CUDA 代码
2. 性能 profiling
3. 尝试更多配置组合
4. 分析瓶颈

当前最佳结果 (from sparse_mla_fwd_optimized_result.txt):
- 配置: block_I=64, num_stages=2, threads=128
- 性能: 1.089 ms, 8.87 TFLOPS, 2.47 TB/s
"""
import torch
import tilelang
from tilelang import language as T
from tilelang.profiler import do_bench


# ============================================================================
# 1. Kernel 定义 (与 sparse_mla_fwd_optimized.py 相同)
# ============================================================================

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
    block_I=64,
    num_stages=2,
    threads=128,
):
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    REPLICATE_H = head_kv // 64 if head_kv > 64 else 1
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
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            
            mask = T.alloc_fragment([BI], "bool")
            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
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
            max_kv_i = s_i

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
# 2. 调试函数
# ============================================================================

def analyze_cuda_code(block_I=64, num_stages=2, threads=128):
    """分析生成的 CUDA 代码"""
    print("=" * 70)
    print(f"CUDA 代码分析 (block_I={block_I}, stages={num_stages}, threads={threads})")
    print("=" * 70)
    
    # 输入参数
    heads, dim, tail_dim, topk = 2, 256, 64, 2048
    
    kernel = sparse_mla_fwd_debug(
        heads, dim, tail_dim, topk,
        block_I=block_I, num_stages=num_stages, threads=threads
    )
    
    # 获取 CUDA 源码
    cuda_source = kernel.get_kernel_source()
    
    # 保存到文件
    filename = f"cuda_kernel_bi{block_I}_s{num_stages}_t{threads}.cu"
    with open(filename, "w") as f:
        f.write(cuda_source)
    print(f"CUDA 源码已保存到: {filename}")
    
    # 分析关键特征
    print("\n📊 代码特征分析:")
    features = {
        "cp.async": cuda_source.count("cp.async"),
        "ldmatrix": cuda_source.count("ldmatrix"),
        "mma.sync": cuda_source.count("mma.sync"),
        "__shared__": cuda_source.count("__shared__"),
        "barrier": cuda_source.count("barrier"),
        "syncthreads": cuda_source.count("__syncthreads"),
    }
    for key, count in features.items():
        if count > 0:
            print(f"  - {key}: {count} 次")
    
    # 打印部分代码
    print("\n📄 CUDA 代码片段 (前 100 行):")
    print("-" * 70)
    lines = cuda_source.split('\n')[:100]
    for i, line in enumerate(lines):
        print(f"{i+1:4d} | {line}")
    print("-" * 70)
    
    return cuda_source


def benchmark_config(block_I, num_stages, threads, warmup=50, rep=100):
    """测试单个配置的性能"""
    # 输入参数
    B, S, SKV, H, HKV = 1, 2048, 2048, 2, 1
    DQK, DV = 320, 256
    topk = 2048
    dtype = torch.bfloat16
    
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda")
    
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    
    try:
        kernel = sparse_mla_fwd_debug(
            H, DV, DQK - DV, topk,
            block_I=block_I, num_stages=num_stages, threads=threads
        )
        
        def fn():
            return kernel(q, kv, indices)
        
        # 预热
        for _ in range(10):
            fn()
        
        ms = do_bench(fn, warmup=warmup, rep=rep)
        
        io_bytes = B * S * DQK * topk * 2
        flops = B * S * (DQK + DV) * topk * 2 * H
        
        return {
            'ms': ms,
            'tflops': flops / (ms * 1e-3) / 1e12,
            'bandwidth': io_bytes / (ms * 1e-3) / 1e12,
            'success': True,
        }
    except Exception as e:
        return {
            'ms': float('inf'),
            'tflops': 0,
            'bandwidth': 0,
            'success': False,
            'error': str(e),
        }


def extended_config_search():
    """扩展配置搜索，尝试更多组合"""
    print("=" * 70)
    print("扩展配置搜索")
    print("=" * 70)
    
    # 当前最佳: block_I=64, num_stages=2, threads=128 → 1.089 ms
    # 尝试更多配置
    configs = [
        # 原最佳配置
        (64, 2, 128),
        
        # 尝试更少线程
        (64, 2, 64),
        (64, 2, 96),
        
        # 尝试更多 pipeline stages
        (64, 3, 128),
        (64, 4, 128),
        (64, 4, 64),
        
        # 尝试更小的 block_I
        (32, 2, 128),
        (32, 2, 64),
        (32, 4, 64),
        
        # 尝试 block_I=128 配合少线程
        (128, 1, 64),
        (128, 1, 128),
        (128, 2, 64),
        
        # 尝试 num_stages=1
        (64, 1, 64),
        (64, 1, 128),
        (32, 1, 64),
    ]
    
    results = []
    for block_I, num_stages, threads in configs:
        print(f"\n测试: block_I={block_I}, num_stages={num_stages}, threads={threads}")
        result = benchmark_config(block_I, num_stages, threads)
        
        if result['success']:
            print(f"  ✓ {result['ms']:.3f} ms, {result['tflops']:.2f} TFLOPS, {result['bandwidth']:.2f} TB/s")
        else:
            print(f"  ✗ 失败: {result.get('error', 'unknown')[:50]}")
        
        results.append((block_I, num_stages, threads, result))
    
    # 排序并显示结果
    print("\n" + "=" * 70)
    print("性能排名 (从快到慢)")
    print("=" * 70)
    print(f"{'block_I':>8} {'stages':>7} {'threads':>8} {'Time(ms)':>10} {'TFLOPS':>8} {'BW(TB/s)':>10}")
    print("-" * 60)
    
    results.sort(key=lambda x: x[3]['ms'])
    for block_I, num_stages, threads, r in results:
        if r['success']:
            print(f"{block_I:>8} {num_stages:>7} {threads:>8} {r['ms']:>10.3f} {r['tflops']:>8.2f} {r['bandwidth']:>10.2f}")
        else:
            print(f"{block_I:>8} {num_stages:>7} {threads:>8} {'FAILED':>10}")
    
    return results


def profiler_analysis(block_I=64, num_stages=2, threads=128):
    """使用 profiler 进行详细分析"""
    print("=" * 70)
    print(f"Profiler 详细分析 (block_I={block_I}, stages={num_stages}, threads={threads})")
    print("=" * 70)
    
    # 输入参数
    B, S, SKV, H, HKV = 1, 2048, 2048, 2, 1
    DQK, DV = 320, 256
    topk = 2048
    dtype = torch.bfloat16
    
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda")
    
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    
    kernel = sparse_mla_fwd_debug(
        H, DV, DQK - DV, topk,
        block_I=block_I, num_stages=num_stages, threads=threads
    )
    
    # 获取 profiler
    profiler = kernel.get_profiler()
    
    # 多种测量模式
    print("\n📊 性能测量:")
    
    # 使用不同的 return_mode
    latency_mean = profiler.do_bench(warmup=50, rep=200, return_mode="mean")
    latency_median = profiler.do_bench(warmup=50, rep=200, return_mode="median")
    latency_min = profiler.do_bench(warmup=50, rep=200, return_mode="min")
    latency_max = profiler.do_bench(warmup=50, rep=200, return_mode="max")
    
    print(f"  Mean:   {latency_mean:.3f} ms")
    print(f"  Median: {latency_median:.3f} ms")
    print(f"  Min:    {latency_min:.3f} ms")
    print(f"  Max:    {latency_max:.3f} ms")
    print(f"  Jitter: {latency_max - latency_min:.3f} ms ({(latency_max - latency_min) / latency_median * 100:.1f}%)")
    
    # 分位数
    print("\n📈 分位数分析:")
    quantiles = profiler.do_bench(warmup=50, rep=200, quantiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    percentiles = [25, 50, 75, 90, 95, 99]
    for p, q in zip(percentiles, quantiles):
        print(f"  P{p}: {q:.3f} ms")
    
    # 计算性能指标
    io_bytes = B * S * DQK * topk * 2
    flops = B * S * (DQK + DV) * topk * 2 * H
    
    print("\n📊 性能指标 (使用 median):")
    print(f"  延迟:     {latency_median:.3f} ms")
    print(f"  TFLOPS:   {flops / (latency_median * 1e-3) / 1e12:.2f}")
    print(f"  带宽:     {io_bytes / (latency_median * 1e-3) / 1e12:.2f} TB/s")
    
    # H20 理论峰值
    h20_memory_bw = 4.0  # TB/s
    bandwidth = io_bytes / (latency_median * 1e-3) / 1e12
    print(f"  带宽效率: {bandwidth / h20_memory_bw * 100:.1f}% (vs H20 peak 4 TB/s)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="分析 CUDA 代码")
    parser.add_argument("--search", action="store_true", help="扩展配置搜索")
    parser.add_argument("--profile", action="store_true", help="详细 profiling")
    parser.add_argument("--all", action="store_true", help="运行所有分析")
    parser.add_argument("--block_I", type=int, default=64)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    args = parser.parse_args()
    
    if args.all or (not args.cuda and not args.search and not args.profile):
        args.cuda = True
        args.search = True
        args.profile = True
    
    print("=" * 70)
    print("Sparse MLA Forward 调试和性能分析")
    print("=" * 70)
    print("当前最佳结果: block_I=64, num_stages=2, threads=128 → 1.089 ms")
    print("=" * 70)
    
    if args.cuda:
        analyze_cuda_code(args.block_I, args.num_stages, args.threads)
    
    if args.profile:
        profiler_analysis(args.block_I, args.num_stages, args.threads)
    
    if args.search:
        extended_config_search()


if __name__ == "__main__":
    main()
