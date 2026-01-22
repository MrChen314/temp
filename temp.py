# ruff: noqa
"""
Triton 实现的 Sparse MLA Backward Pass
参考 tilelang 版本实现，与 ref_sparse_mla_bwd_interface 对比精度

关键优化：
1. 分块处理 D 维度，避免共享内存/寄存器溢出
2. 使用较小的 block size: BLOCK_H=64, BLOCK_N=32
3. 分阶段计算和写回
"""

import torch
import triton
import triton.language as tl


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
    h_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    o_base = b_idx * stride_o_b + s_idx * stride_o_s + h_idx * stride_o_h
    do_base = b_idx * stride_do_b + s_idx * stride_do_s + h_idx * stride_do_h
    
    acc = 0.0
    
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        o_vals = tl.load(O_ptr + o_base + d_offs * stride_o_d, mask=d_mask, other=0.0)
        do_vals = tl.load(dO_ptr + do_base + d_offs * stride_do_d, mask=d_mask, other=0.0)
        
        acc += tl.sum(o_vals.to(tl.float32) * do_vals.to(tl.float32))
    
    delta_offset = b_idx * stride_delta_b + s_idx * stride_delta_s + h_idx * stride_delta_h
    tl.store(Delta_ptr + delta_offset, acc)


def launch_preprocess(O, dO):
    """启动 preprocess kernel"""
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
    dKV_fp32_ptr,
    dKV_bf16_ptr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """将 dKV 从 float32 转换为 bfloat16"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    vals = tl.load(dKV_fp32_ptr + offs, mask=mask, other=0.0)
    tl.store(dKV_bf16_ptr + offs, vals.to(tl.bfloat16), mask=mask)


def launch_postprocess(dKV_fp32):
    """启动 postprocess kernel"""
    dKV_bf16 = torch.empty_like(dKV_fp32, dtype=torch.bfloat16)
    total_elements = dKV_fp32.numel()
    
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    postprocess_kernel[grid](
        dKV_fp32, dKV_bf16,
        total_elements,
        BLOCK_SIZE,
    )
    
    return dKV_bf16


# ============================================================================
# Main BWD Kernel - 分块处理 D 维度
# ============================================================================
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
    S: tl.constexpr,
    S_kv: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    D_tail: tl.constexpr,
    topk: tl.constexpr,
    kv_group: tl.constexpr,
    sm_scale,
    # Block 大小 - 使用较小的 block 避免内存溢出
    BLOCK_H: tl.constexpr,   # head block = 64
    BLOCK_N: tl.constexpr,   # KV block = 32
    BLOCK_D: tl.constexpr,   # D 分块大小 = 64
):
    """
    Sparse MLA BWD kernel
    
    关键优化: 分块处理 D 维度
    - 将 D=512 分成 D/BLOCK_D = 8 个小块
    - 每次只加载和计算一个 D 块
    - 累加 attention scores 跨所有 D 块
    
    Grid: (S, B, kv_group * num_h_blocks)
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
    
    # Causal mask
    max_kv_idx = s_idx
    NS = topk // BLOCK_N
    
    # 基址计算
    q_base = b_idx * stride_q_b + s_idx * stride_q_s
    do_base = b_idx * stride_do_b + s_idx * stride_do_s
    idx_base = b_idx * stride_idx_b + s_idx * stride_idx_s + g_idx * stride_idx_g
    kv_base = b_idx * stride_kv_b + g_idx * stride_kv_g
    
    # 加载 Lse 和 Delta: [BLOCK_H] - 这些是标量，很小
    lse_ptrs = Lse_ptr + b_idx * stride_lse_b + s_idx * stride_lse_s + h_indices * stride_lse_h
    lse_local = tl.load(lse_ptrs, mask=h_mask, other=0.0)
    
    delta_ptrs = Delta_ptr + b_idx * stride_delta_b + s_idx * stride_delta_s + h_indices * stride_delta_h
    delta_local = tl.load(delta_ptrs, mask=h_mask, other=0.0)
    
    # ========== 主循环: 遍历 index blocks ==========
    for i_block in range(NS):
        # 加载 indices: [BLOCK_N]
        n_offs = tl.arange(0, BLOCK_N)
        idx_ptrs = Indices_ptr + idx_base + (i_block * BLOCK_N + n_offs) * stride_idx_k
        indices = tl.load(idx_ptrs)
        
        # Causal mask
        causal_mask = indices <= max_kv_idx
        
        # ========== 阶段1: 计算 attention scores (需要遍历所有 D 块) ==========
        # acc_p = Q @ KV^T, 需要累加所有 D 块的贡献
        acc_p = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        
        # 遍历 D 维度的块 (D 部分)
        d_offs = tl.arange(0, BLOCK_D)
        for d_block in range(0, D, BLOCK_D):
            d_idx = d_block + d_offs
            d_mask = d_idx < D
            
            # 加载 Q 块: [BLOCK_H, BLOCK_D]
            q_ptrs = Q_ptr + q_base + h_indices[:, None] * stride_q_h + d_idx[None, :] * stride_q_d
            q_block = tl.load(q_ptrs, mask=h_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # 加载 KV 块: [BLOCK_N, BLOCK_D] 通过 gather
            kv_block = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
            for n_i in tl.static_range(BLOCK_N):
                kv_idx = tl.load(Indices_ptr + idx_base + (i_block * BLOCK_N + n_i) * stride_idx_k)
                kv_row_ptr = KV_ptr + kv_base + kv_idx * stride_kv_s
                kv_d_ptrs = kv_row_ptr + d_idx * stride_kv_d
                kv_row = tl.load(kv_d_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                kv_block = tl.where((tl.arange(0, BLOCK_N) == n_i)[:, None], kv_row[None, :], kv_block)
            
            # 累加: Q @ KV^T
            acc_p += tl.dot(q_block, tl.trans(kv_block))
        
        # D_tail 部分
        d_tail_offs = tl.arange(0, BLOCK_D)
        d_tail_mask = d_tail_offs < D_tail
        
        # 加载 Q_tail: [BLOCK_H, D_tail]
        q_tail_ptrs = Q_ptr + q_base + h_indices[:, None] * stride_q_h + (D + d_tail_offs[None, :]) * stride_q_d
        q_tail = tl.load(q_tail_ptrs, mask=h_mask[:, None] & d_tail_mask[None, :], other=0.0).to(tl.float32)
        
        # 加载 KV_tail: [BLOCK_N, D_tail]
        kv_tail = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        for n_i in tl.static_range(BLOCK_N):
            kv_idx = tl.load(Indices_ptr + idx_base + (i_block * BLOCK_N + n_i) * stride_idx_k)
            kv_row_ptr = KV_ptr + kv_base + kv_idx * stride_kv_s
            kv_tail_ptrs = kv_row_ptr + (D + d_tail_offs) * stride_kv_d
            kv_tail_row = tl.load(kv_tail_ptrs, mask=d_tail_mask, other=0.0).to(tl.float32)
            kv_tail = tl.where((tl.arange(0, BLOCK_N) == n_i)[:, None], kv_tail_row[None, :], kv_tail)
        
        # 累加 Q_tail @ KV_tail^T
        acc_p += tl.dot(q_tail, tl.trans(kv_tail))
        
        # 应用 causal mask
        acc_p = tl.where(causal_mask[None, :], acc_p, -1e10)
        
        # P = exp2(acc_p * sm_scale_log2 - Lse)
        P = tl.exp2(acc_p * sm_scale_log2 - lse_local[:, None])
        
        # ========== 阶段2: 计算 dP ==========
        # acc_dp = dO @ V^T (V = KV[:, :D])
        acc_dp = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        
        for d_block in range(0, D, BLOCK_D):
            d_idx = d_block + d_offs
            d_mask = d_idx < D
            
            # 加载 dO 块: [BLOCK_H, BLOCK_D]
            do_ptrs = dO_ptr + do_base + h_indices[:, None] * stride_do_h + d_idx[None, :] * stride_do_d
            do_block = tl.load(do_ptrs, mask=h_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # 加载 V 块 (= KV[:, :D]): [BLOCK_N, BLOCK_D]
            v_block = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
            for n_i in tl.static_range(BLOCK_N):
                kv_idx = tl.load(Indices_ptr + idx_base + (i_block * BLOCK_N + n_i) * stride_idx_k)
                kv_row_ptr = KV_ptr + kv_base + kv_idx * stride_kv_s
                kv_d_ptrs = kv_row_ptr + d_idx * stride_kv_d
                kv_row = tl.load(kv_d_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                v_block = tl.where((tl.arange(0, BLOCK_N) == n_i)[:, None], kv_row[None, :], v_block)
            
            # 累加: dO @ V^T
            acc_dp += tl.dot(do_block, tl.trans(v_block))
        
        # dP = P * (acc_dp - Delta) * sm_scale
        dP = P * (acc_dp - delta_local[:, None]) * sm_scale
        
        # ========== 阶段3: 累加 dQ (分块处理) ==========
        for d_block in range(0, D, BLOCK_D):
            d_idx = d_block + d_offs
            d_mask = d_idx < D
            
            # 加载 KV 块用于计算 dQ
            kv_block = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
            for n_i in tl.static_range(BLOCK_N):
                kv_idx = tl.load(Indices_ptr + idx_base + (i_block * BLOCK_N + n_i) * stride_idx_k)
                kv_row_ptr = KV_ptr + kv_base + kv_idx * stride_kv_s
                kv_d_ptrs = kv_row_ptr + d_idx * stride_kv_d
                kv_row = tl.load(kv_d_ptrs, mask=d_mask, other=0.0).to(tl.float32)
                kv_block = tl.where((tl.arange(0, BLOCK_N) == n_i)[:, None], kv_row[None, :], kv_block)
            
            # dQ 块 = dP @ KV: [BLOCK_H, BLOCK_D]
            dq_block = tl.dot(dP.to(tl.float32), kv_block)
            
            # 原子累加到 dQ (因为多个 index block 会累加到同一个 dQ 位置)
            dq_base = b_idx * stride_dq_b + s_idx * stride_dq_s
            dq_ptrs = dQ_ptr + dq_base + h_indices[:, None] * stride_dq_h + d_idx[None, :] * stride_dq_d
            
            # 使用原子加法累加 dQ
            for h_i in tl.static_range(BLOCK_H):
                for d_i in tl.static_range(BLOCK_D):
                    if (h_start + h_i) < H and (d_block + d_i) < D:
                        tl.atomic_add(dQ_ptr + dq_base + (h_start + h_i) * stride_dq_h + (d_block + d_i) * stride_dq_d, 
                                     dq_block[h_i, d_i])
        
        # dQ_tail = dP @ KV_tail
        dq_tail_block = tl.dot(dP.to(tl.float32), kv_tail)
        
        # 原子累加 dQ_tail
        for h_i in tl.static_range(BLOCK_H):
            for d_i in tl.static_range(BLOCK_D):
                if (h_start + h_i) < H and d_i < D_tail:
                    tl.atomic_add(dQ_ptr + b_idx * stride_dq_b + s_idx * stride_dq_s + 
                                 (h_start + h_i) * stride_dq_h + (D + d_i) * stride_dq_d,
                                 dq_tail_block[h_i, d_i])
        
        # ========== 阶段4: 计算并原子累加 dKV (分块处理) ==========
        for d_block in range(0, D, BLOCK_D):
            d_idx = d_block + d_offs
            d_mask = d_idx < D
            
            # 加载 Q 块
            q_ptrs = Q_ptr + q_base + h_indices[:, None] * stride_q_h + d_idx[None, :] * stride_q_d
            q_block = tl.load(q_ptrs, mask=h_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # 加载 dO 块
            do_ptrs = dO_ptr + do_base + h_indices[:, None] * stride_do_h + d_idx[None, :] * stride_do_d
            do_block = tl.load(do_ptrs, mask=h_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # dKV = dP^T @ Q + P^T @ dO: [BLOCK_N, BLOCK_D]
            dkv_block = tl.dot(tl.trans(dP.to(tl.float32)), q_block) + tl.dot(tl.trans(P.to(tl.float32)), do_block)
            
            # 原子累加到 dKV
            for n_i in tl.static_range(BLOCK_N):
                kv_idx = tl.load(Indices_ptr + idx_base + (i_block * BLOCK_N + n_i) * stride_idx_k)
                dkv_row_ptr = dKV_ptr + b_idx * stride_dkv_b + kv_idx * stride_dkv_s + g_idx * stride_dkv_g
                
                # 提取第 n_i 行
                n_mask = tl.arange(0, BLOCK_N) == n_i
                dkv_row = tl.sum(tl.where(n_mask[:, None], dkv_block, 0.0), axis=0)
                
                # 向量化原子累加
                tl.atomic_add(dkv_row_ptr + d_idx * stride_dkv_d, dkv_row, mask=d_mask)
        
        # dKV_tail = dP^T @ Q_tail: [BLOCK_N, D_tail]
        dkv_tail_block = tl.dot(tl.trans(dP.to(tl.float32)), q_tail)
        
        # 原子累加 dKV_tail
        for n_i in tl.static_range(BLOCK_N):
            kv_idx = tl.load(Indices_ptr + idx_base + (i_block * BLOCK_N + n_i) * stride_idx_k)
            dkv_row_ptr = dKV_ptr + b_idx * stride_dkv_b + kv_idx * stride_dkv_s + g_idx * stride_dkv_g
            
            n_mask = tl.arange(0, BLOCK_N) == n_i
            dkv_tail_row = tl.sum(tl.where(n_mask[:, None], dkv_tail_block, 0.0), axis=0)
            
            tl.atomic_add(dkv_row_ptr + (D + d_tail_offs) * stride_dkv_d, dkv_tail_row, mask=d_tail_mask)


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
    
    # Step 2: 分配输出 tensors (dQ 使用 float32 因为要原子累加)
    dq = torch.zeros(B, S, H, DQKV, dtype=torch.float32, device=q.device)
    dkv = torch.zeros(B, S_kv, kv_group, DQKV, dtype=torch.float32, device=q.device)
    
    # Step 3: 启动主 BWD kernel
    H_kv = H // kv_group
    # 使用较小的 block size 以避免内存溢出
    BLOCK_H = min(64, max(16, H_kv))
    BLOCK_N = 32   # 与 tilelang 一致
    BLOCK_D = 64   # 分块处理 D 维度
    
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
        S, S_kv, H, D, D_tail, topk, kv_group,
        sm_scale,
        BLOCK_H, BLOCK_N, BLOCK_D,
    )
    
    # Step 4: 转换为 bfloat16
    dq = dq.to(torch.bfloat16)
    dkv = launch_postprocess(dkv)
    
    return dq, dkv


# ============================================================================
# 参考实现
# ============================================================================
def ref_sparse_mla_bwd_interface(q, kv, o, do, indices, lse, sm_scale=None, is_causal=True):
    """参考实现：使用 PyTorch autograd 计算梯度"""
    from sparse_mla_fwd import ref_sparse_mla_fwd_interface
    
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
        def calc_diff(name, a, b):
            abs_diff = torch.abs(a.float() - b.float())
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()
            rel_diff = (abs_diff / (1e-6 + torch.abs(b.float()))).mean().item()
            print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_diff={rel_diff:.6f}")
            return max_diff, rel_diff

        calc_diff("dQ", triton_dq, ref_dq)
        calc_diff("dKV", triton_dkv, ref_dkv)
    
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
