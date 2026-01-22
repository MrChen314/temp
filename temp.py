# ruff: noqa
"""
Triton 实现的 Sparse MLA Backward Pass
参考 tilelang 版本实现，与 ref_sparse_mla_bwd_interface 对比精度
"""

import torch
import triton
import triton.language as tl
# from utils import assert_tensors_similar


# ============================================================================
# Preprocess Kernel: 计算 Delta = rowsum(O * dO)
# ============================================================================
@triton.jit
def preprocess_kernel(
    O_ptr,          # [B, S, H, D]
    dO_ptr,         # [B, S, H, D]
    Delta_ptr,      # [B, S, H]
    stride_o_b, stride_o_s, stride_o_h, stride_o_d,
    stride_do_b, stride_do_s, stride_do_h, stride_do_d,
    stride_delta_b, stride_delta_s, stride_delta_h,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    计算 Delta[b, s, h] = sum_d(O[b, s, h, d] * dO[b, s, h, d])
    Grid: (H, S, B)
    """
    # 获取当前 block 的索引
    h_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    # 计算 O 和 dO 的基址
    o_base = b_idx * stride_o_b + s_idx * stride_o_s + h_idx * stride_o_h
    do_base = b_idx * stride_do_b + s_idx * stride_do_s + h_idx * stride_do_h
    
    # 累加器
    acc = 0.0
    
    # 遍历 D 维度
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        # 加载 O 和 dO
        o_vals = tl.load(O_ptr + o_base + d_offs * stride_o_d, mask=d_mask, other=0.0)
        do_vals = tl.load(dO_ptr + do_base + d_offs * stride_do_d, mask=d_mask, other=0.0)
        
        # 累加 O * dO
        acc += tl.sum(o_vals.to(tl.float32) * do_vals.to(tl.float32))
    
    # 存储结果
    delta_offset = b_idx * stride_delta_b + s_idx * stride_delta_s + h_idx * stride_delta_h
    tl.store(Delta_ptr + delta_offset, acc)


def launch_preprocess(O, dO):
    """
    启动 preprocess kernel
    O: [B, S, H, D]
    dO: [B, S, H, D]
    返回: Delta [B, S, H]
    """
    B, S, H, D = O.shape
    Delta = torch.empty((B, S, H), dtype=torch.float32, device=O.device)
    
    BLOCK_D = 64
    grid = (H, S, B)
    
    preprocess_kernel[grid](
        O, dO, Delta,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        Delta.stride(0), Delta.stride(1), Delta.stride(2),
        S, H, D,
        BLOCK_D,
    )
    
    return Delta


# ============================================================================
# Postprocess Kernel: dKV float32 -> bfloat16
# ============================================================================
@triton.jit
def postprocess_kernel(
    dKV_fp32_ptr,   # [B, S_kv, kv_group, D+D_tail] float32
    dKV_bf16_ptr,   # [B, S_kv, kv_group, D+D_tail] bfloat16
    stride_b, stride_s, stride_g, stride_d,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    将 dKV 从 float32 转换为 bfloat16
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    vals = tl.load(dKV_fp32_ptr + offs, mask=mask, other=0.0)
    tl.store(dKV_bf16_ptr + offs, vals.to(tl.bfloat16), mask=mask)


def launch_postprocess(dKV_fp32):
    """
    启动 postprocess kernel
    dKV_fp32: [B, S_kv, kv_group, D+D_tail] float32
    返回: dKV_bf16 [B, S_kv, kv_group, D+D_tail] bfloat16
    """
    dKV_bf16 = torch.empty_like(dKV_fp32, dtype=torch.bfloat16)
    total_elements = dKV_fp32.numel()
    
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    postprocess_kernel[grid](
        dKV_fp32, dKV_bf16,
        dKV_fp32.stride(0), dKV_fp32.stride(1), dKV_fp32.stride(2), dKV_fp32.stride(3),
        total_elements,
        BLOCK_SIZE,
    )
    
    return dKV_bf16


@triton.jit
def sparse_mla_bwd_kernel(
    # 输入指针
    Q_ptr,          # [B, S, H, D+D_tail]
    KV_ptr,         # [B, S_kv, kv_group, D+D_tail]
    dO_ptr,         # [B, S, H, D]
    Indices_ptr,    # [B, S, kv_group, topk]
    Lse_ptr,        # [B, S, H]
    Delta_ptr,      # [B, S, H]
    # 输出指针
    dQ_ptr,         # [B, S, H, D+D_tail]
    dKV_ptr,        # [B, S_kv, kv_group, D+D_tail] (float32)
    # Q strides
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    # KV strides  
    stride_kv_b, stride_kv_s, stride_kv_g, stride_kv_d,
    # dO strides
    stride_do_b, stride_do_s, stride_do_h, stride_do_d,
    # Indices strides
    stride_idx_b, stride_idx_s, stride_idx_g, stride_idx_k,
    # Lse strides
    stride_lse_b, stride_lse_s, stride_lse_h,
    # Delta strides
    stride_delta_b, stride_delta_s, stride_delta_h,
    # dQ strides
    stride_dq_b, stride_dq_s, stride_dq_h, stride_dq_d,
    # dKV strides
    stride_dkv_b, stride_dkv_s, stride_dkv_g, stride_dkv_d,
    # 常量参数
    B: tl.constexpr,
    S: tl.constexpr,
    S_kv: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_tail: tl.constexpr,
    topk: tl.constexpr,
    kv_group: tl.constexpr,
    sm_scale,
    # Block 大小
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    优化版本的 Sparse MLA BWD kernel
    使用向量化 tl.dot 操作
    """
    sm_scale_log2 = sm_scale * 1.44269504
    
    # 获取索引
    s_idx = tl.program_id(0)
    b_idx = tl.program_id(1)
    hz_idx = tl.program_id(2)
    
    # 计算 head 相关索引
    H_kv = H // kv_group
    num_h_blocks = tl.cdiv(H_kv, BLOCK_H)
    g_idx = hz_idx // num_h_blocks
    h_block_idx = hz_idx % num_h_blocks
    h_start = g_idx * H_kv + h_block_idx * BLOCK_H
    
    # Head offsets
    h_offs = tl.arange(0, BLOCK_H)
    h_indices = h_start + h_offs
    h_mask = h_indices < ((g_idx + 1) * H_kv)
    
    # D offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_tail_offs = tl.arange(0, D_tail) if D_tail <= 64 else tl.arange(0, 64)
    
    # Causal mask
    max_kv_idx = s_idx
    NS = topk // BLOCK_N
    
    # 基址计算
    q_base = b_idx * stride_q_b + s_idx * stride_q_s
    do_base = b_idx * stride_do_b + s_idx * stride_do_s
    idx_base = b_idx * stride_idx_b + s_idx * stride_idx_s + g_idx * stride_idx_g
    kv_base = b_idx * stride_kv_b + g_idx * stride_kv_g
    
    # 加载 Q: [BLOCK_H, D] - 分块加载
    q_ptrs = Q_ptr + q_base + h_indices[:, None] * stride_q_h + d_offs[None, :] * stride_q_d
    q_local = tl.load(q_ptrs, mask=h_mask[:, None] & (d_offs[None, :] < D), other=0.0).to(tl.float32)
    
    # 加载 Q_tail: [BLOCK_H, D_tail]
    q_tail_ptrs = Q_ptr + q_base + h_indices[:, None] * stride_q_h + (D + d_tail_offs[None, :]) * stride_q_d
    q_tail_local = tl.load(q_tail_ptrs, mask=h_mask[:, None] & (d_tail_offs[None, :] < D_tail), other=0.0).to(tl.float32)
    
    # 加载 dO: [BLOCK_H, D]
    do_ptrs = dO_ptr + do_base + h_indices[:, None] * stride_do_h + d_offs[None, :] * stride_do_d
    do_local = tl.load(do_ptrs, mask=h_mask[:, None] & (d_offs[None, :] < D), other=0.0).to(tl.float32)
    
    # 加载 Lse 和 Delta: [BLOCK_H]
    lse_ptrs = Lse_ptr + b_idx * stride_lse_b + s_idx * stride_lse_s + h_indices * stride_lse_h
    lse_local = tl.load(lse_ptrs, mask=h_mask, other=0.0)
    
    delta_ptrs = Delta_ptr + b_idx * stride_delta_b + s_idx * stride_delta_s + h_indices * stride_delta_h
    delta_local = tl.load(delta_ptrs, mask=h_mask, other=0.0)
    
    # 初始化 dQ 累加器
    acc_dq = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    acc_dq_tail = tl.zeros([BLOCK_H, D_tail if D_tail <= 64 else 64], dtype=tl.float32)
    
    # 主循环
    for i_block in range(NS):
        # 加载 indices
        n_offs = tl.arange(0, BLOCK_N)
        idx_ptrs = Indices_ptr + idx_base + (i_block * BLOCK_N + n_offs) * stride_idx_k
        indices = tl.load(idx_ptrs)
        
        # Causal mask
        causal_mask = indices <= max_kv_idx
        
        # 加载 KV: [BLOCK_N, D] 通过 gather
        kv_local = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        kv_tail_local = tl.zeros([BLOCK_N, D_tail if D_tail <= 64 else 64], dtype=tl.float32)
        
        for n_i in tl.static_range(BLOCK_N):
            kv_idx = tl.load(Indices_ptr + idx_base + (i_block * BLOCK_N + n_i) * stride_idx_k)
            kv_row_ptr = KV_ptr + kv_base + kv_idx * stride_kv_s
            
            # 加载 KV 的 D 部分
            kv_d_ptrs = kv_row_ptr + d_offs * stride_kv_d
            kv_row = tl.load(kv_d_ptrs, mask=d_offs < D, other=0.0).to(tl.float32)
            kv_local = tl.where((tl.arange(0, BLOCK_N) == n_i)[:, None], kv_row[None, :], kv_local)
            
            # 加载 KV_tail
            kv_tail_ptrs = kv_row_ptr + (D + d_tail_offs) * stride_kv_d
            kv_tail_row = tl.load(kv_tail_ptrs, mask=d_tail_offs < D_tail, other=0.0).to(tl.float32)
            kv_tail_local = tl.where((tl.arange(0, BLOCK_N) == n_i)[:, None], kv_tail_row[None, :], kv_tail_local)
        
        # 计算 attention scores: Q @ KV^T
        # [BLOCK_H, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_H, BLOCK_N]
        acc_p = tl.dot(q_local, tl.trans(kv_local))
        acc_p += tl.dot(q_tail_local, tl.trans(kv_tail_local))
        
        # 应用 causal mask
        acc_p = tl.where(causal_mask[None, :], acc_p, -1e10)
        
        # P = exp2(acc_p * sm_scale_log2 - Lse)
        P = tl.exp2(acc_p * sm_scale_log2 - lse_local[:, None])
        
        # 计算 dP = dO @ V^T
        acc_dp = tl.dot(do_local, tl.trans(kv_local))
        
        # dP = P * (acc_dp - Delta) * sm_scale
        dP = P * (acc_dp - delta_local[:, None]) * sm_scale
        
        # 累加 dQ: dP @ KV
        acc_dq += tl.dot(dP.to(q_local.dtype), kv_local)
        acc_dq_tail += tl.dot(dP.to(q_tail_local.dtype), kv_tail_local)
        
        # 计算 dKV 并原子累加
        # dKV = dP^T @ Q + P^T @ dO: [BLOCK_N, D]
        dkv_local = tl.dot(tl.trans(dP.to(q_local.dtype)), q_local) + tl.dot(tl.trans(P.to(do_local.dtype)), do_local)
        dkv_tail_local = tl.dot(tl.trans(dP.to(q_tail_local.dtype)), q_tail_local)
        
        # 原子累加到 dKV
        for n_i in tl.static_range(BLOCK_N):
            kv_idx = tl.load(Indices_ptr + idx_base + (i_block * BLOCK_N + n_i) * stride_idx_k)
            dkv_row_ptr = dKV_ptr + b_idx * stride_dkv_b + kv_idx * stride_dkv_s + g_idx * stride_dkv_g
            
            # 提取第 n_i 行 (使用 mask + sum 代替直接索引)
            n_mask = tl.arange(0, BLOCK_N) == n_i
            dkv_row = tl.sum(tl.where(n_mask[:, None], dkv_local, 0.0), axis=0)
            dkv_tail_row = tl.sum(tl.where(n_mask[:, None], dkv_tail_local, 0.0), axis=0)
            
            # 向量化原子累加 D 部分
            d_offs_local = tl.arange(0, BLOCK_D)
            tl.atomic_add(dkv_row_ptr + d_offs_local * stride_dkv_d, dkv_row, mask=d_offs_local < D)
            
            # 向量化原子累加 D_tail 部分
            d_tail_offs_local = tl.arange(0, D_tail) if D_tail <= 64 else tl.arange(0, 64)
            tl.atomic_add(dkv_row_ptr + (D + d_tail_offs_local) * stride_dkv_d, dkv_tail_row, mask=d_tail_offs_local < D_tail)
    
    # 存储 dQ
    dq_base = b_idx * stride_dq_b + s_idx * stride_dq_s
    dq_ptrs = dQ_ptr + dq_base + h_indices[:, None] * stride_dq_h + d_offs[None, :] * stride_dq_d
    tl.store(dq_ptrs, acc_dq.to(tl.bfloat16), mask=h_mask[:, None] & (d_offs[None, :] < D))
    
    dq_tail_ptrs = dQ_ptr + dq_base + h_indices[:, None] * stride_dq_h + (D + d_tail_offs[None, :]) * stride_dq_d
    tl.store(dq_tail_ptrs, acc_dq_tail.to(tl.bfloat16), mask=h_mask[:, None] & (d_tail_offs[None, :] < D_tail))


# ============================================================================
# 接口函数
# ============================================================================
def sparse_mla_bwd_triton(q, kv, o, do, indices, lse, sm_scale=None, is_causal=True):
    """
    Triton 实现的 Sparse MLA 反向传播
    
    Args:
        q: [B, S, H, DQKV] Query tensor
        kv: [B, S_kv, kv_group, DQKV] KV tensor
        o: [B, S, H, D] Forward output (用于计算 Delta)
        do: [B, S, H, D] Output gradient
        indices: [B, S, kv_group, topk] Sparse indices
        lse: [B, S, H] Log-sum-exp from forward pass
        sm_scale: Softmax scale factor
        is_causal: Whether to use causal attention
        
    Returns:
        dq: [B, S, H, DQKV] Query gradient
        dkv: [B, S_kv, kv_group, DQKV] KV gradient
    """
    assert q.is_contiguous()
    assert kv.is_contiguous()
    assert o.is_contiguous()
    assert do.is_contiguous()
    assert indices.is_contiguous()
    assert lse.is_contiguous()
    
    B, S, H, DQKV = q.shape
    _, S_kv, kv_group, _ = kv.shape
    D = 512  # V 的维度
    D_tail = DQKV - D
    topk = indices.shape[-1]
    
    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    
    # Step 1: Preprocess - 计算 Delta
    delta = launch_preprocess(o, do)
    
    # Step 2: 分配输出 tensors
    dq = torch.zeros_like(q)
    dkv = torch.zeros(B, S_kv, kv_group, DQKV, dtype=torch.float32, device=q.device)
    
    # Step 3: 启动主 BWD kernel
    H_kv = H // kv_group
    BLOCK_H = min(64, max(16, H_kv))  # 根据 H_kv 调整
    BLOCK_N = 32  # Index block size
    BLOCK_D = 512  # D 的处理块大小
    
    num_h_blocks = (H_kv + BLOCK_H - 1) // BLOCK_H
    grid = (S, B, kv_group * num_h_blocks)
    
    sparse_mla_bwd_kernel[grid](
        q, kv, do, indices, lse, delta, dq, dkv,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # KV strides
        kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
        # dO strides
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        # Indices strides
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        # Lse strides
        lse.stride(0), lse.stride(1), lse.stride(2),
        # Delta strides
        delta.stride(0), delta.stride(1), delta.stride(2),
        # dQ strides
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        # dKV strides
        dkv.stride(0), dkv.stride(1), dkv.stride(2), dkv.stride(3),
        # 常量
        B, S, S_kv, H, D, D_tail, topk, kv_group,
        sm_scale,
        BLOCK_H, BLOCK_N, BLOCK_D,
    )
    
    # Step 4: Postprocess - 转换 dkv 为 bfloat16
    dkv = launch_postprocess(dkv)
    
    return dq, dkv


# ============================================================================
# 参考实现 (从 sparse_mla_bwd.py 导入)
# ============================================================================
def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, sm_scale=None, is_causal=True):
    """
    参考实现：使用 PyTorch autograd 计算梯度
    """
    from sparse_mla_bwd.sparse_mla_fwd import ref_sparse_mla_fwd_interface
    
    q = q.detach().clone()
    kv = kv.detach().clone()
    q.requires_grad = True
    kv.requires_grad = True
    o = ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale, is_causal)
    o.backward(do)
    return q.grad, kv.grad


# ============================================================================
# 测试函数
# ============================================================================
def test_sparse_mla_bwd_triton(
    B=1, S=4096, SKV=8192, H=64, HKV=1, DQKV=576, DV=512, topk=2048,
    dtype=torch.bfloat16, check_correctness=True
):
    """
    测试 Triton 实现的 sparse_mla_bwd
    与 ref_sparse_mla_bwd_interface 对比精度
    """
    print(f"测试配置: B={B}, S={S}, SKV={SKV}, H={H}, HKV={HKV}, DQKV={DQKV}, DV={DV}, topk={topk}")
    
    # 准备数据
    torch.manual_seed(42)
    q = torch.randn((B, S, H, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQKV), dtype=dtype, device="cuda").requires_grad_(True)
    do = torch.randn((B, S, H, DV), dtype=dtype, device="cuda")
    
    # 创建稀疏 indices
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i
    
    # Forward pass (获取 o 和 lse)
    from sparse_mla_fwd import sparse_mla_fwd_interface
    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices)
    
    # Triton BWD
    triton_dq, triton_dkv = sparse_mla_bwd_triton(q, kv, tl_out, do, indices, tl_lse)
    
    # Reference BWD
    ref_dq, ref_dkv = ref_sparse_mla_bwd_interface(q, kv, None, do, indices, None)
    
    if check_correctness:
        # 精度检查
        def calc_diff(a, b):
            abs_diff = torch.abs(a - b)
            max_diff = abs_diff.max().item()
            rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
            return max_diff, rel_diff

        max_diff, rel_diff = calc_diff(ref_dq, triton_dq)
        max_diff, rel_diff = calc_diff(ref_dkv, triton_dkv)
        print(f"{max_diff=}\n{rel_diff=}\n")
    
    # 性能测试
    per_token_flop = 2 * sum([
        H * DV * topk,      # dO @ V^T
        H * DQKV * topk,    # Q @ K^T
        H * DQKV * topk,    # dP @ K (dQ)
        H * DQKV * topk,    # dP^T @ Q (dKV)
        H * DV * topk,      # P^T @ dO (dKV)
    ])
    
    def fn():
        return sparse_mla_bwd_triton(q, kv, tl_out, do, indices, tl_lse)
    
    ms = triton.testing.do_bench(fn, warmup=250, rep=100)
    print(f"Triton BWD 平均时间: {ms:.3f} ms")
    print(f"BWD IO 带宽 = {(B * S * max(DQKV * 2, DQKV + DV) * topk * 2) / (ms * 1e-3) / 1e12:.2f} TB/s")
    print(f"BWD TFLOPS = {per_token_flop * S / (ms * 1e-3) / 1e12:.2f}")


if __name__ == "__main__":
    test_sparse_mla_bwd_triton(
        B=1, S=4096, SKV=8192, H=64, HKV=1, DQKV=576, DV=512, topk=2048,
        dtype=torch.bfloat16, check_correctness=True
    )
