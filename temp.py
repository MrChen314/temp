# ruff: noqa
"""
Sparse MLA Forward - 优化版本
针对 NVIDIA H20 GPU 优化 (78 SMs, 227KB shared memory per block with opt-in)

输入配置:
- heads=2, dim=256, tail_dim=64, topk=2048, kv_group=1
- q.shape:[1, 2048, 2, 320], kv.shape:[1, 2048, 1, 320]
"""
import torch
import tilelang
from tilelang import language as T
from utils import assert_tensors_similar


def compute_shared_memory_usage(H_per_block, D, D_tail, BI, num_stages, dtype_size=2, accum_dtype_size=4):
    """计算 shared memory 使用量 (单位: bytes)"""
    # 静态分配 (不参与 pipeline)
    Q_shared = H_per_block * D * dtype_size
    Q_tail_shared = H_per_block * D_tail * dtype_size
    O_shared = H_per_block * D * dtype_size
    Lse_shared = H_per_block * accum_dtype_size
    S_shared = H_per_block * BI * dtype_size
    
    # Pipeline 分配 (需要 double/triple buffer)
    KV_shared = BI * D * dtype_size * num_stages
    K_tail_shared = BI * D_tail * dtype_size * num_stages
    
    total = Q_shared + Q_tail_shared + O_shared + Lse_shared + S_shared + KV_shared + K_tail_shared
    return {
        'Q_shared': Q_shared,
        'Q_tail_shared': Q_tail_shared,
        'KV_shared': KV_shared,
        'K_tail_shared': K_tail_shared,
        'O_shared': O_shared,
        'Lse_shared': Lse_shared,
        'S_shared': S_shared,
        'total': total,
        'total_kb': total / 1024,
    }


def compute_register_usage(H_per_block, D, BI, threads, accum_dtype_size=4):
    """计算寄存器使用量 (单位: bytes per thread)"""
    # Fragment 分配
    acc_o = H_per_block * D * accum_dtype_size
    acc_s = H_per_block * BI * accum_dtype_size
    sumexp = H_per_block * accum_dtype_size
    sumexp_i = H_per_block * accum_dtype_size
    alpha = H_per_block * accum_dtype_size
    m_i = H_per_block * accum_dtype_size
    m_i_prev = H_per_block * accum_dtype_size
    mask = BI  # bool, ~1 byte each
    
    total = acc_o + acc_s + sumexp + sumexp_i + alpha + m_i + m_i_prev + mask
    per_thread = total / threads
    registers_per_thread = per_thread / 4  # 32-bit registers
    
    return {
        'acc_o': acc_o,
        'acc_s': acc_s,
        'scalar_fragments': sumexp + sumexp_i + alpha + m_i + m_i_prev,
        'mask': mask,
        'total': total,
        'per_thread_bytes': per_thread,
        'registers_per_thread': registers_per_thread,
    }


def find_optimal_config(heads, dim, tail_dim, topk, kv_group=1, 
                        shared_mem_limit_kb=227, max_registers_per_thread=256):
    """
    自动搜索最优配置
    
    H20 限制:
    - Shared memory: 227 KB (opt-in)
    - Registers per SM: 65536
    - Max threads per block: 1024
    """
    head_kv = heads // kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    H_per_block = padded_H if head_kv <= 64 else 64
    
    # 候选配置
    block_I_candidates = [32, 64, 128, 256]
    num_stages_candidates = [1, 2, 3, 4]
    threads_candidates = [128, 256, 512]
    
    valid_configs = []
    
    for block_I in block_I_candidates:
        if topk % block_I != 0:
            continue
            
        for num_stages in num_stages_candidates:
            for threads in threads_candidates:
                smem = compute_shared_memory_usage(H_per_block, dim, tail_dim, block_I, num_stages)
                regs = compute_register_usage(H_per_block, dim, block_I, threads)
                
                if smem['total_kb'] <= shared_mem_limit_kb and regs['registers_per_thread'] <= max_registers_per_thread:
                    # 计算循环次数
                    num_iterations = tilelang.cdiv(topk, block_I)
                    
                    # 启发式评分: 优先减少循环次数，同时平衡 shared memory 使用
                    # 更高的 block_I 和 num_stages 通常更好
                    score = (
                        block_I * 10  # 更大的 block_I 减少循环次数
                        + num_stages * 5  # 更多 stages 提高流水线效率
                        - smem['total_kb'] * 0.1  # 轻微惩罚过高的 shared memory 使用
                        + (256 - regs['registers_per_thread']) * 0.5  # 奖励较低的寄存器使用
                    )
                    
                    valid_configs.append({
                        'block_I': block_I,
                        'num_stages': num_stages,
                        'threads': threads,
                        'smem_kb': smem['total_kb'],
                        'regs_per_thread': regs['registers_per_thread'],
                        'num_iterations': num_iterations,
                        'score': score,
                    })
    
    # 按评分排序
    valid_configs.sort(key=lambda x: x['score'], reverse=True)
    return valid_configs


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_optimized(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=128,      # 优化: 从 64 增加到 128，减少循环次数
    num_stages=2,      # 保持 2 stages，避免 shared memory 超限
    threads=256,
):
    """
    优化后的 Sparse MLA Forward kernel
    
    主要优化:
    1. block_I 从 64 增加到 128，循环次数从 32 减少到 16
    2. 更好的内存访问合并
    3. 减少冗余的 shared memory 分配
    """
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, f"topk({topk}) should be divisible by block_I({block_I})"
    
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

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

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, otherwise you should handle Q copy and Output copy with your mask"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    # 打印资源使用情况
    smem_usage = compute_shared_memory_usage(H_per_block, D, D_tail, BI, num_stages)
    reg_usage = compute_register_usage(H_per_block, D, BI, threads)
    print(f"[资源分析] Shared Memory: {smem_usage['total_kb']:.2f} KB / 227 KB")
    print(f"[资源分析] Registers per thread: {reg_usage['registers_per_thread']:.1f} / 256")
    print(f"[资源分析] 循环次数: {NI}, block_I: {BI}, num_stages: {num_stages}")

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            # Shared memory 分配
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            
            # 寄存器分配
            mask = T.alloc_fragment([BI], "bool")
            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            # 初始化
            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # 避免 -inf - inf 产生 nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            # 加载 Q 到 shared memory (只在循环外加载一次)
            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            # 主循环 - 使用 software pipeline
            for i_i in T.Pipelined(NI, num_stages=num_stages):
                # 计算 mask (causal attention)
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                # 加载 KV 到 shared memory (gather 操作)
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                # 应用 mask 到 attention scores
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                
                # Q @ K^T (分两部分计算)
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                
                # Online softmax 更新
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                
                # 计算缩放因子 alpha
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                
                # 计算 softmax(scores)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                
                # 更新 sumexp
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                
                # 缩放之前的 output
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                # P @ V
                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # 最终归一化
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            # 写回结果 (直接写回 global memory，减少 shared memory 使用)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mla_fwd_interface_optimized(
    q, kv, indices, 
    sm_scale=None, 
    return_p_sum: bool = False, 
    d_v=256,  # 根据输入调整
    block_I=128,  # 优化后的默认值
    num_stages=2, 
    threads=256
):
    """优化后的接口函数"""
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    dim = d_v
    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = sparse_mla_fwd_optimized(
        heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual, 
        block_I=block_I, num_stages=num_stages, threads=threads
    )
    out, lse = kernel(q, kv, indices)
    return out, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, d_v=256):
    """参考实现"""
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    dim = d_v
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


def test_sparse_mla_fwd_optimized(
    B=1,
    S=2048,
    SKV=2048,
    H=2,
    HKV=1,
    DQK=320,  # dim + tail_dim = 256 + 64
    DV=256,
    topk=2048,
    dtype=torch.bfloat16,
    check_correctness=True,
    block_I=128,  # 优化后的值
    num_stages=2,
    threads=256,
):
    """测试优化后的 kernel"""
    print("=" * 60)
    print(f"测试配置: B={B}, S={S}, SKV={SKV}, H={H}, HKV={HKV}")
    print(f"         DQK={DQK}, DV={DV}, topk={topk}")
    print(f"         block_I={block_I}, num_stages={num_stages}, threads={threads}")
    print("=" * 60)
    
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    tl_out, tl_lse = sparse_mla_fwd_interface_optimized(
        q, kv, indices, 
        d_v=DV,
        block_I=block_I, 
        num_stages=num_stages, 
        threads=threads
    )

    if check_correctness:
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)
        assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")
        print("✓ 正确性验证通过")

    def fn():
        return sparse_mla_fwd_interface_optimized(
            q, kv, indices, 
            d_v=DV,
            block_I=block_I, 
            num_stages=num_stages, 
            threads=threads
        )

    from tilelang.profiler import do_bench

    ms = do_bench(fn, rep=100, warmup=250)
    print(f"平均时间: {ms:.3f} ms")
    print(f"FWD IO bandwidth = {(B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12:.2f} TB/s")
    print(f"FWD TFLOPS = {(B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12:.2f}")
    
    return ms


def benchmark_different_configs():
    """测试不同配置的性能"""
    print("\n" + "=" * 70)
    print("不同配置性能对比")
    print("=" * 70)
    
    # 根据输入文件的参数
    configs = [
        # (block_I, num_stages, threads)
        (64, 2, 256),   # 原始配置
        (128, 2, 256),  # 增大 block_I
        (64, 3, 256),   # 增加 pipeline stages
        (128, 2, 128),  # 减少线程数
        (128, 2, 512),  # 增加线程数
        (64, 2, 128),   # 原始 block_I + 少线程
    ]
    
    results = []
    for block_I, num_stages, threads in configs:
        print(f"\n--- 配置: block_I={block_I}, num_stages={num_stages}, threads={threads} ---")
        try:
            ms = test_sparse_mla_fwd_optimized(
                B=1, S=2048, SKV=2048, H=2, HKV=1, DQK=320, DV=256, topk=2048,
                block_I=block_I, num_stages=num_stages, threads=threads,
                check_correctness=False  # 只测性能
            )
            results.append((block_I, num_stages, threads, ms))
        except Exception as e:
            print(f"配置失败: {e}")
            results.append((block_I, num_stages, threads, float('inf')))
    
    print("\n" + "=" * 70)
    print("性能排名 (从快到慢):")
    print("=" * 70)
    results.sort(key=lambda x: x[3])
    for i, (block_I, num_stages, threads, ms) in enumerate(results):
        if ms == float('inf'):
            print(f"{i+1}. block_I={block_I}, num_stages={num_stages}, threads={threads}: 失败")
        else:
            print(f"{i+1}. block_I={block_I}, num_stages={num_stages}, threads={threads}: {ms:.3f} ms")


def print_config_analysis():
    """打印配置分析"""
    print("\n" + "=" * 70)
    print("配置空间分析 (NVIDIA H20: 227KB shared memory)")
    print("=" * 70)
    
    # 根据输入文件的参数
    heads, dim, tail_dim, topk, kv_group = 2, 256, 64, 2048, 1
    
    configs = find_optimal_config(heads, dim, tail_dim, topk, kv_group)
    
    print(f"\n输入参数: heads={heads}, dim={dim}, tail_dim={tail_dim}, topk={topk}")
    print(f"\n找到 {len(configs)} 个有效配置，前 10 个最优配置:")
    print("-" * 70)
    print(f"{'block_I':>8} {'stages':>7} {'threads':>8} {'SMEM(KB)':>10} {'Regs/Thr':>10} {'Iters':>6} {'Score':>8}")
    print("-" * 70)
    
    for cfg in configs[:10]:
        print(f"{cfg['block_I']:>8} {cfg['num_stages']:>7} {cfg['threads']:>8} "
              f"{cfg['smem_kb']:>10.1f} {cfg['regs_per_thread']:>10.1f} "
              f"{cfg['num_iterations']:>6} {cfg['score']:>8.1f}")


if __name__ == "__main__":
    # 1. 打印配置分析
    print_config_analysis()
    
    # 2. 使用输入文件中的具体参数测试
    print("\n" + "=" * 70)
    print("使用输入文件参数测试")
    print("=" * 70)
    
    test_sparse_mla_fwd_optimized(
        B=1,
        S=2048,
        SKV=2048,
        H=2,
        HKV=1,
        DQK=320,  # 256 + 64
        DV=256,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        block_I=128,  # 推荐的优化配置
        num_stages=2,
        threads=256,
    )
    
    # 3. 对比不同配置
    benchmark_different_configs()
