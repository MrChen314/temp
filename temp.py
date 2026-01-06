# ruff: noqa
"""
TileLang 实现的 Indexer Loss Kernel - 2 线程组流水线优化版本

优化点：
1. Producer 线程组预取 K 数据，使用双缓冲隐藏 global memory 延迟
2. 计算线程组执行 QK 计算 + Online Softmax + 归一化累加

架构：
- tx < 128: 计算线程组（QK 计算 + Online Softmax + 归一化累加）
- tx >= 128: Producer 线程组（按 indices 预取 K 数据到 shared memory）
"""

import torch
import tilelang
from tilelang import language as T
from dataclasses import dataclass
from typing import List, Optional
import torch.nn.functional as F


@tilelang.jit(
    out_idx=[-1],  # attn_sum 作为输出
    compile_flags=[
        "-O3",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10",
        "-DNDEBUG",
    ],
)
def indexer_loss_fwd_kernel(
    batch_size,
    num_heads,
    chunk_size,
    head_dim,
    topk,
    chunk_offset,
    scaling,
    block_tk=64,
    block_h=16,
    threads=256,
):
    """
    TileLang 实现的 Indexer Loss Forward Kernel - 2 线程组版本
    
    每个 program 处理一个 (batch, query_row)。
    对于每个 head_block：
    - Pass1: 遍历所有 topk_blocks，计算 online softmax 得到全局 m 和 l
    - Pass2: 重新遍历，使用全局 m 和 l 计算归一化概率并累加
    
    流水线优化：Producer 线程组使用双缓冲预取 K 数据
    """
    assert topk % block_tk == 0, "topk should be divisible by block_tk"
    
    # 确保 block_h 满足 tensor core 要求
    block_h = max(block_h, 16)
    
    # 计算块数量
    NTK = tilelang.cdiv(topk, block_tk)
    NH = tilelang.cdiv(num_heads, block_h)
    D = head_dim
    
    # 数据类型
    dtype = T.bfloat16
    accum_dtype = T.float32
    indices_dtype = T.int32
    
    # Tensor shapes
    q_shape = [batch_size, num_heads, chunk_size, head_dim]
    k_shape = [batch_size, chunk_size + chunk_offset, head_dim]
    indices_shape = [batch_size, chunk_size, topk]
    attn_sum_shape = [batch_size, chunk_size, topk]
    
    BI = block_tk
    BH = block_h
    
    NEG_INF = -1e9
    LOG2E = 1.44269504  # log2(e)
    
    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        AttnSum: T.Tensor(attn_sum_shape, accum_dtype),
    ):
        with T.Kernel(chunk_size, batch_size, threads=threads) as (bx, by):
            # ===============================================================
            # Shared Memory 分配
            # ===============================================================
            # Q 缓存: [BH, D] - 当前 head_block 的 Q
            Q_shared = T.alloc_shared([BH, D], dtype)
            
            # K 双缓冲: [BI, D] x 2
            K_shared_0 = T.alloc_shared([BI, D], dtype)
            K_shared_1 = T.alloc_shared([BI, D], dtype)
            
            # 有效性标记双缓冲
            is_valid_0 = T.alloc_shared([BI], "bool", scope="shared")
            is_valid_1 = T.alloc_shared([BI], "bool", scope="shared")
            
            # Softmax 参数的 shared memory 版本（用于避免 layout 冲突）
            m_block_shared = T.alloc_shared([BH], accum_dtype)
            m_new_shared = T.alloc_shared([BH], accum_dtype)
            m_global_shared = T.alloc_shared([BH], accum_dtype)
            l_global_shared = T.alloc_shared([BH], accum_dtype)
            
            # exp_sum 的 shared memory 缓存
            exp_sum_shared = T.alloc_shared([BI], accum_dtype)
            
            # Fragment 分配
            acc_qk = T.alloc_fragment([BI, BH], accum_dtype)
            m_global = T.alloc_fragment([BH], accum_dtype)
            l_global = T.alloc_fragment([BH], accum_dtype)
            m_block = T.alloc_fragment([BH], accum_dtype)
            m_new = T.alloc_fragment([BH], accum_dtype)
            alpha = T.alloc_fragment([BH], accum_dtype)
            
            # 用于 reduce 操作的中间变量
            block_sum = T.alloc_fragment([BH], accum_dtype)
            
            indices_local = T.alloc_local([1], indices_dtype)
            
            # ===============================================================
            # Barrier 分配（2 线程组）
            # ===============================================================
            bar_q = T.alloc_barrier(arrive_count=256)
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)
            bar_k_0_free = T.alloc_barrier(arrive_count=128)
            bar_k_1_free = T.alloc_barrier(arrive_count=128)
            
            # ===============================================================
            # 计算索引
            # ===============================================================
            b_i = by
            s_i = bx
            global_query_pos = chunk_offset + s_i
            
            tx = T.get_thread_binding()
            
            # ===============================================================
            # 按 head_block 分组处理
            # ===============================================================
            for h_block in T.serial(NH):
                h_start = h_block * BH
                
                # 加载 Q 到 shared memory
                T.copy(Q[b_i, h_start:h_start + BH, s_i, 0:D], Q_shared)
                T.barrier_arrive(bar_q)
                
                if tx < 128:
                    # =======================================================
                    # 计算线程组
                    # =======================================================
                    T.set_max_nreg(240, 1)
                    
                    # 初始化 Online Softmax 状态
                    T.fill(m_global, NEG_INF)
                    T.fill(l_global, 0.0)
                    
                    T.barrier_wait(bar_q, 0)
                    
                    # Pass 1: 遍历所有 topk_blocks，计算 online softmax
                    for i_i in T.serial(T.ceildiv(NTK, 2)):
                        # ----- Buffer 0 -----
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))
                        
                        # 初始化 QK（根据有效性设置初值）
                        for tk_i, h_i in T.Parallel(BI, BH):
                            acc_qk[tk_i, h_i] = T.if_then_else(is_valid_0[tk_i], 0.0, NEG_INF)
                        
                        # QK = K @ Q^T
                        T.gemm(K_shared_0, Q_shared, acc_qk, transpose_B=True, wg_wait=-1)
                        T.wait_wgmma(0)
                        
                        # 缩放
                        for tk_i, h_i in T.Parallel(BI, BH):
                            acc_qk[tk_i, h_i] = acc_qk[tk_i, h_i] * scaling
                        
                        # Online Softmax 更新 - 步骤1：计算块内 max
                        T.reduce_max(acc_qk, m_block, dim=0, clear=True)
                        # 将 m_block 存入 shared memory 以避免 layout 冲突
                        T.copy(m_block, m_block_shared)
                        
                        # 步骤2：更新全局 max 和 alpha（从 shared memory 读取 m_block）
                        for h_i in T.Parallel(BH):
                            m_new[h_i] = T.max(m_global[h_i], m_block_shared[h_i])
                            alpha[h_i] = T.exp2((m_global[h_i] - m_new[h_i]) * LOG2E)
                        
                        # 将 m_new 存入 shared memory 以避免 2D/1D layout 冲突
                        T.copy(m_new, m_new_shared)
                        
                        # 步骤3：计算 exp 值（2D 操作，从 shared memory 读取 m_new）
                        for tk_i, h_i in T.Parallel(BI, BH):
                            raw_exp = T.exp2((acc_qk[tk_i, h_i] - m_new_shared[h_i]) * LOG2E)
                            acc_qk[tk_i, h_i] = T.if_then_else(is_valid_0[tk_i], raw_exp, 0.0)
                        
                        # 步骤4：更新 l_global（使用预分配的 block_sum 避免作用域问题）
                        T.fill(block_sum, 0.0)
                        for tk_i in T.serial(BI):
                            for h_i in T.Parallel(BH):
                                block_sum[h_i] = block_sum[h_i] + acc_qk[tk_i, h_i]
                        
                        for h_i in T.Parallel(BH):
                            l_global[h_i] = l_global[h_i] * alpha[h_i] + block_sum[h_i]
                            m_global[h_i] = m_new[h_i]
                        
                        T.barrier_arrive(bar_k_0_free[0])
                        
                        # ----- Buffer 1 -----
                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))
                        
                        for tk_i, h_i in T.Parallel(BI, BH):
                            acc_qk[tk_i, h_i] = T.if_then_else(is_valid_1[tk_i], 0.0, NEG_INF)
                        
                        T.gemm(K_shared_1, Q_shared, acc_qk, transpose_B=True, wg_wait=-1)
                        T.wait_wgmma(0)
                        
                        for tk_i, h_i in T.Parallel(BI, BH):
                            acc_qk[tk_i, h_i] = acc_qk[tk_i, h_i] * scaling
                        
                        T.reduce_max(acc_qk, m_block, dim=0, clear=True)
                        T.copy(m_block, m_block_shared)
                        
                        for h_i in T.Parallel(BH):
                            m_new[h_i] = T.max(m_global[h_i], m_block_shared[h_i])
                            alpha[h_i] = T.exp2((m_global[h_i] - m_new[h_i]) * LOG2E)
                        
                        T.copy(m_new, m_new_shared)
                        
                        for tk_i, h_i in T.Parallel(BI, BH):
                            raw_exp = T.exp2((acc_qk[tk_i, h_i] - m_new_shared[h_i]) * LOG2E)
                            acc_qk[tk_i, h_i] = T.if_then_else(is_valid_1[tk_i], raw_exp, 0.0)
                        
                        T.fill(block_sum, 0.0)
                        for tk_i in T.serial(BI):
                            for h_i in T.Parallel(BH):
                                block_sum[h_i] = block_sum[h_i] + acc_qk[tk_i, h_i]
                        
                        for h_i in T.Parallel(BH):
                            l_global[h_i] = l_global[h_i] * alpha[h_i] + block_sum[h_i]
                            m_global[h_i] = m_new[h_i]
                        
                        T.barrier_arrive(bar_k_1_free[0])
                    
                    # 存储最终的 softmax 参数到 shared memory
                    # 处理边界情况
                    for h_i in T.Parallel(BH):
                        m_global[h_i] = T.if_then_else(m_global[h_i] == NEG_INF, 0.0, m_global[h_i])
                        l_global[h_i] = T.if_then_else(l_global[h_i] < 1e-9, 1.0, l_global[h_i])
                    
                    T.copy(m_global, m_global_shared)
                    T.copy(l_global, l_global_shared)
                    
                    # Pass 2: 重新遍历，使用全局 m 和 l 计算归一化概率并累加
                    for i_i in T.serial(T.ceildiv(NTK, 2)):
                        tk_start_0 = (i_i * 2) * BI
                        tk_start_1 = (i_i * 2 + 1) * BI
                        
                        # ----- Buffer 0 -----
                        # 等待 Producer 重新加载数据
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1) + (NTK + 1) // 2)
                        
                        for tk_i, h_i in T.Parallel(BI, BH):
                            acc_qk[tk_i, h_i] = T.if_then_else(is_valid_0[tk_i], 0.0, NEG_INF)
                        
                        T.gemm(K_shared_0, Q_shared, acc_qk, transpose_B=True, wg_wait=-1)
                        T.wait_wgmma(0)
                        
                        for tk_i, h_i in T.Parallel(BI, BH):
                            acc_qk[tk_i, h_i] = acc_qk[tk_i, h_i] * scaling
                        
                        # 使用全局 m 和 l 计算归一化概率
                        # p[tk_i, h_i] = exp(qk - m_global) / l_global
                        for tk_i, h_i in T.Parallel(BI, BH):
                            raw_exp = T.exp2((acc_qk[tk_i, h_i] - m_global_shared[h_i]) * LOG2E)
                            p_val = raw_exp / l_global_shared[h_i]
                            acc_qk[tk_i, h_i] = T.if_then_else(is_valid_0[tk_i], p_val, 0.0)
                        
                        # 对 head 求和（使用预分配的 exp_sum_shared 避免作用域问题）
                        for tk_i in T.Parallel(BI):
                            exp_sum_shared[tk_i] = 0.0
                        for h_i in T.serial(BH):
                            for tk_i in T.Parallel(BI):
                                exp_sum_shared[tk_i] = exp_sum_shared[tk_i] + acc_qk[tk_i, h_i]
                        
                        # 写入到输出
                        for tk_i in T.Parallel(BI):
                            curr_tk = tk_start_0 + tk_i
                            if curr_tk < topk:
                                if h_block == 0:
                                    AttnSum[b_i, s_i, curr_tk] = exp_sum_shared[tk_i]
                                else:
                                    old_val = AttnSum[b_i, s_i, curr_tk]
                                    AttnSum[b_i, s_i, curr_tk] = old_val + exp_sum_shared[tk_i]
                        
                        T.barrier_arrive(bar_k_0_free[0])
                        
                        # ----- Buffer 1 -----
                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1) + (NTK + 1) // 2)
                        
                        for tk_i, h_i in T.Parallel(BI, BH):
                            acc_qk[tk_i, h_i] = T.if_then_else(is_valid_1[tk_i], 0.0, NEG_INF)
                        
                        T.gemm(K_shared_1, Q_shared, acc_qk, transpose_B=True, wg_wait=-1)
                        T.wait_wgmma(0)
                        
                        for tk_i, h_i in T.Parallel(BI, BH):
                            acc_qk[tk_i, h_i] = acc_qk[tk_i, h_i] * scaling
                        
                        for tk_i, h_i in T.Parallel(BI, BH):
                            raw_exp = T.exp2((acc_qk[tk_i, h_i] - m_global_shared[h_i]) * LOG2E)
                            p_val = raw_exp / l_global_shared[h_i]
                            acc_qk[tk_i, h_i] = T.if_then_else(is_valid_1[tk_i], p_val, 0.0)
                        
                        for tk_i in T.Parallel(BI):
                            exp_sum_shared[tk_i] = 0.0
                        for h_i in T.serial(BH):
                            for tk_i in T.Parallel(BI):
                                exp_sum_shared[tk_i] = exp_sum_shared[tk_i] + acc_qk[tk_i, h_i]
                        
                        for tk_i in T.Parallel(BI):
                            curr_tk = tk_start_1 + tk_i
                            if curr_tk < topk:
                                if h_block == 0:
                                    AttnSum[b_i, s_i, curr_tk] = exp_sum_shared[tk_i]
                                else:
                                    old_val = AttnSum[b_i, s_i, curr_tk]
                                    AttnSum[b_i, s_i, curr_tk] = old_val + exp_sum_shared[tk_i]
                        
                        T.barrier_arrive(bar_k_1_free[0])
                    
                else:
                    # =======================================================
                    # Producer 线程组：预取 K 数据到 shared memory
                    # =======================================================
                    T.set_max_nreg(80, 0)
                    
                    # Pass 1 的数据加载
                    for i_i in T.serial(T.ceildiv(NTK, 2)):
                        # ----- Buffer 0 -----
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        
                        thread_id = tx - 128  # 0-127
                        rows_per_thread = T.ceildiv(BI, 128)
                        
                        for r in T.serial(rows_per_thread):
                            row_idx = thread_id * rows_per_thread + r
                            if row_idx < BI:
                                tk_global_idx = (i_i * 2) * BI + row_idx
                                if tk_global_idx < topk:
                                    indices_local[0] = Indices[b_i, s_i, tk_global_idx]
                                    is_valid_0[row_idx] = indices_local[0] <= global_query_pos
                                    
                                    if is_valid_0[row_idx]:
                                        with T.attr("default", "async_scope", 1):
                                            for d in T.serial(D):
                                                K_shared_0[row_idx, d] = K[b_i, indices_local[0], d]
                                else:
                                    is_valid_0[row_idx] = False
                        
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])
                        
                        # ----- Buffer 1 -----
                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        
                        for r in T.serial(rows_per_thread):
                            row_idx = thread_id * rows_per_thread + r
                            if row_idx < BI:
                                tk_global_idx = (i_i * 2 + 1) * BI + row_idx
                                if tk_global_idx < topk:
                                    indices_local[0] = Indices[b_i, s_i, tk_global_idx]
                                    is_valid_1[row_idx] = indices_local[0] <= global_query_pos
                                    
                                    if is_valid_1[row_idx]:
                                        with T.attr("default", "async_scope", 1):
                                            for d in T.serial(D):
                                                K_shared_1[row_idx, d] = K[b_i, indices_local[0], d]
                                else:
                                    is_valid_1[row_idx] = False
                        
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])
                    
                    # Pass 2 的数据加载（重复加载以支持第二轮计算）
                    for i_i in T.serial(T.ceildiv(NTK, 2)):
                        # ----- Buffer 0 -----
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1) + (NTK + 1) // 2)
                        
                        thread_id = tx - 128
                        rows_per_thread = T.ceildiv(BI, 128)
                        
                        for r in T.serial(rows_per_thread):
                            row_idx = thread_id * rows_per_thread + r
                            if row_idx < BI:
                                tk_global_idx = (i_i * 2) * BI + row_idx
                                if tk_global_idx < topk:
                                    indices_local[0] = Indices[b_i, s_i, tk_global_idx]
                                    is_valid_0[row_idx] = indices_local[0] <= global_query_pos
                                    
                                    if is_valid_0[row_idx]:
                                        with T.attr("default", "async_scope", 1):
                                            for d in T.serial(D):
                                                K_shared_0[row_idx, d] = K[b_i, indices_local[0], d]
                                else:
                                    is_valid_0[row_idx] = False
                        
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])
                        
                        # ----- Buffer 1 -----
                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1) + (NTK + 1) // 2)
                        
                        for r in T.serial(rows_per_thread):
                            row_idx = thread_id * rows_per_thread + r
                            if row_idx < BI:
                                tk_global_idx = (i_i * 2 + 1) * BI + row_idx
                                if tk_global_idx < topk:
                                    indices_local[0] = Indices[b_i, s_i, tk_global_idx]
                                    is_valid_1[row_idx] = indices_local[0] <= global_query_pos
                                    
                                    if is_valid_1[row_idx]:
                                        with T.attr("default", "async_scope", 1):
                                            for d in T.serial(D):
                                                K_shared_1[row_idx, d] = K[b_i, indices_local[0], d]
                                else:
                                    is_valid_1[row_idx] = False
                        
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])
    
    return main


def indexer_loss_fwd_tilelang(
    query, key, indices, scaling, chunk_offset=0, 
    block_tk=64, block_h=16, threads=256,
    return_kernel=False, print_kernel=False
):
    """
    TileLang 实现的 Indexer Loss 前向传播
    
    Args:
        query: [batch, num_heads, chunk_size, head_dim]
        key: [batch, kv_len, head_dim]
        indices: [batch, chunk_size, topk]
        scaling: attention scaling factor
        chunk_offset: 当前 chunk 在完整序列中的起始位置
        block_tk: topk 分块大小
        block_h: head 分块大小
        threads: 线程数
        return_kernel: 是否返回编译后的 kernel
        print_kernel: 是否打印 kernel 源码
    
    Returns:
        attn_sum: [batch, chunk_size, topk]
    """
    assert query.is_contiguous() and key.is_contiguous() and indices.is_contiguous()
    
    batch_size, num_heads, chunk_size, head_dim = query.shape
    _, kv_len, _ = key.shape
    _, _, topk = indices.shape
    
    # 调整 block 大小
    block_tk = min(block_tk, topk)
    # 确保 block_tk 能整除 topk
    while topk % block_tk != 0:
        block_tk -= 1
    block_tk = max(block_tk, 16)
    
    block_h = min(block_h, num_heads)
    block_h = max(block_h, 16)
    
    # 转换数据类型
    query_bf16 = query.to(torch.bfloat16) if query.dtype != torch.bfloat16 else query
    key_bf16 = key.to(torch.bfloat16) if key.dtype != torch.bfloat16 else key
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices
    
    # 创建输出 tensor
    attn_sum = torch.zeros(batch_size, chunk_size, topk, device=query.device, dtype=torch.float32)
    
    # 编译 kernel
    kernel = indexer_loss_fwd_kernel(
        batch_size, num_heads, chunk_size, head_dim, topk,
        chunk_offset, scaling, block_tk, block_h, threads
    )
    
    if print_kernel:
        print(kernel.get_kernel_source())
    
    # 执行 kernel
    attn_sum, = kernel(query_bf16, key_bf16, indices_i32, attn_sum)
    
    if return_kernel:
        return attn_sum, kernel
    
    return attn_sum


# ============================================================================
# Autograd Function
# ============================================================================

class IndexerLossFunctionTileLang(torch.autograd.Function):
    """
    TileLang 版本的 Indexer Loss Autograd Function
    """
    
    @staticmethod
    def forward(ctx, query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10, 
                block_tk=64, block_h=16):
        """前向传播"""
        batch_size, num_heads, chunk_size, head_dim = query.shape
        topk = indices.shape[-1]
        
        block_tk = min(block_tk, topk)
        while topk % block_tk != 0:
            block_tk -= 1
        block_tk = max(block_tk, 16)
        
        block_h = min(block_h, num_heads)
        block_h = max(block_h, 16)
        
        # 计算 attn_sum
        attn_sum = indexer_loss_fwd_tilelang(
            query, key, indices, scaling, chunk_offset,
            block_tk=block_tk, block_h=block_h
        )
        
        # 返回 dummy loss
        dummy_loss = torch.tensor(0.0, device=query.device, dtype=torch.float32, requires_grad=True)
        
        # 保存反向传播需要的张量
        ctx.save_for_backward(index_score, indices, attn_sum)
        ctx.chunk_offset = chunk_offset
        ctx.eps = eps
        ctx.batch_size = batch_size
        
        return dummy_loss
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播: grad = index_prob - attn_dist"""
        index_score, indices, attn_sum = ctx.saved_tensors
        chunk_offset = ctx.chunk_offset
        eps = ctx.eps
        batch_size = ctx.batch_size
        
        _, chunk_size, topk = index_score.shape
        device = index_score.device
        
        NEG_INF = -1e9
        
        # 计算 attn_total
        attn_total = attn_sum.sum(dim=-1, keepdim=True)
        attn_total = torch.clamp(attn_total, min=eps)
        
        # 计算 attn_dist
        attn_dist = attn_sum / attn_total
        
        # 创建 causal mask
        query_positions = chunk_offset + torch.arange(chunk_size, device=device).view(1, -1, 1)
        causal_mask = indices > query_positions
        
        # 计算 index_prob = softmax(index_score)
        index_score_masked = index_score.masked_fill(causal_mask, NEG_INF)
        index_prob = F.softmax(index_score_masked, dim=-1)
        
        # 梯度 = index_prob - attn_dist
        grad = index_prob - attn_dist
        grad = grad.masked_fill(causal_mask, 0.0)
        
        # 乘以上游梯度并除以 batch_size
        grad = grad * grad_output / batch_size
        
        return None, None, grad, None, None, None, None, None, None


def indexer_loss_tilelang(query, key, index_score, indices, scaling, chunk_offset=0, eps=1e-10,
                          block_tk=64, block_h=16):
    """TileLang 版本的 Indexer Loss 函数"""
    return IndexerLossFunctionTileLang.apply(
        query, key, index_score, indices, scaling, chunk_offset, eps, block_tk, block_h
    )


# ============================================================================
# PyTorch 参考实现
# ============================================================================

def pytorch_reference_attn_sum(query, key, index_mask, topk_indices, scaling):
    """PyTorch 参考实现: 计算 attn_sum"""
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(index_mask, -1e9)
    attn = torch.softmax(attn, dim=-1)
    
    attn_sum_full = attn.sum(dim=1)
    attn_sum_sparse = torch.gather(attn_sum_full, dim=-1, index=topk_indices)
    
    return attn_sum_sparse


def generate_index_mask_from_score(index_score, topk, device='cuda', chunk_offset=0):
    """从 index_score 生成 index_mask 和 topk_indices"""
    batch_size, chunk_size, seq_len = index_score.shape
    
    query_positions = chunk_offset + torch.arange(chunk_size, device=device).view(-1, 1)
    key_positions = torch.arange(seq_len, device=device).view(1, -1)
    causal_mask = key_positions > query_positions
    
    causal_index_score = index_score.masked_fill(causal_mask, float('-inf'))
    topk_indices = causal_index_score.topk(topk, dim=-1)[1]
    
    index_mask = torch.full(causal_index_score.shape, True, device=device)
    index_mask = index_mask.scatter_(-1, topk_indices, False)
    index_mask = torch.logical_or(index_mask, causal_mask)
    index_mask = index_mask.unsqueeze(1)
    
    return index_mask, topk_indices


# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class TestConfig:
    """测试配置"""
    name: str
    batch_size: int = 1
    num_heads: int = 8
    chunk_size: int = 256
    seq_len: int = 256
    head_dim: int = 64
    topk: int = 32
    seed: int = 42
    
    def __str__(self):
        return (f"batch={self.batch_size}, heads={self.num_heads}, "
                f"chunk={self.chunk_size}, seq={self.seq_len}, "
                f"dim={self.head_dim}, topk={self.topk}")


# ============================================================================
# 精度测试
# ============================================================================

def run_fwd_accuracy_test(config: TestConfig, device: str = 'cuda'):
    """运行单个前向精度测试"""
    torch.manual_seed(config.seed)
    scaling = 1.0 / (config.head_dim ** 0.5)
    
    query = torch.randn(config.batch_size, config.num_heads, config.chunk_size, 
                        config.head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(config.batch_size, config.seq_len, config.head_dim, 
                      device=device, dtype=torch.bfloat16)
    index_score_full = torch.randn(config.batch_size, config.chunk_size, config.seq_len, 
                                   device=device, dtype=torch.bfloat16)
    
    chunk_offset = config.seq_len - config.chunk_size
    index_mask, topk_indices = generate_index_mask_from_score(
        index_score_full, config.topk, device, chunk_offset=chunk_offset)
    
    # PyTorch 参考
    ref_attn_sum = pytorch_reference_attn_sum(query, key, index_mask, topk_indices, scaling)
    
    # TileLang 版本
    try:
        tl_attn_sum = indexer_loss_fwd_tilelang(
            query, key, topk_indices, scaling, chunk_offset=chunk_offset
        )
        
        abs_diff = (ref_attn_sum - tl_attn_sum).abs().max().item()
        rel_diff = abs_diff / (ref_attn_sum.abs().max().item() + 1e-10)
        passed = rel_diff < 1e-2
        
        return {
            'config': config,
            'ref_max': ref_attn_sum.abs().max().item(),
            'tl_max': tl_attn_sum.abs().max().item(),
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'passed': passed,
            'error': None
        }
    except Exception as e:
        import traceback
        return {
            'config': config,
            'ref_max': ref_attn_sum.abs().max().item(),
            'tl_max': 0.0,
            'abs_diff': float('inf'),
            'rel_diff': float('inf'),
            'passed': False,
            'error': str(e) + "\n" + traceback.format_exc()
        }


def test_fwd_accuracy(configs: List[TestConfig]):
    """批量运行前向精度测试"""
    print("\n" + "=" * 100)
    print("前向精度测试 (PyTorch attn_sum vs TileLang attn_sum)")
    print("=" * 100)
    
    results = []
    for config in configs:
        result = run_fwd_accuracy_test(config)
        results.append(result)
    
    print(f"\n{'Name':<12} {'Config':<55} {'RefMax':<12} {'TLMax':<12} {'RelDiff':<12} {'Pass':<6}")
    print("-" * 109)
    for r in results:
        if r['error']:
            print(f"{r['config'].name:<12} {str(r['config']):<55} Error")
            print(f"  Error: {r['error'][:200]}")
        else:
            print(f"{r['config'].name:<12} {str(r['config']):<55} "
                  f"{r['ref_max']:<12.6f} {r['tl_max']:<12.6f} {r['rel_diff']:<12.2e} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 109)
    print(f"前向测试 (attn_sum): {passed_count}/{len(results)} 通过")
    
    return results


# ============================================================================
# 性能测试
# ============================================================================

def test_performance_comparison(
    batch_size: int = 1,
    num_heads: int = 16,
    chunk_size: int = 4 * 1024,
    seq_len: int = 8 * 1024,
    head_dim: int = 128,
    topk: int = 512,
    seed: int = 42,
    num_warmup: int = 10,
    num_benchmark: int = 50,
):
    """性能对比测试: Triton vs TileLang"""
    import time
    from indexer_loss_kernel_opt import compute_attn_sum_opt
    
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("\n" + "=" * 80)
    print("性能对比测试 (Triton vs TileLang)")
    print("=" * 80)
    print(f"参数: batch={batch_size}, heads={num_heads}, chunk={chunk_size}, seq={seq_len}, dim={head_dim}, topk={topk}")
    print("=" * 80)
    
    query = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    chunk_offset = seq_len - chunk_size
    index_score_full = torch.randn(batch_size, chunk_size, seq_len, device=device, dtype=torch.bfloat16)
    index_mask, topk_indices = generate_index_mask_from_score(index_score_full, topk, device, chunk_offset=chunk_offset)
    
    results = {}
    memory_stats = {}
    
    torch.cuda.synchronize()
    base_memory = torch.cuda.memory_allocated() / (1024**3)
    
    # Triton 优化版测试
    torch.cuda.reset_peak_memory_stats()
    for _ in range(num_warmup):
        _ = compute_attn_sum_opt(query, key, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_attn_sum_opt(query, key, topk_indices, scaling, chunk_offset=chunk_offset)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    triton_peak = torch.cuda.max_memory_allocated() / (1024**3)
    results['triton'] = triton_time
    memory_stats['triton'] = triton_peak
    
    print(f"\n>>> Triton 性能: {triton_time:.3f} ms")
    
    # TileLang 版本测试
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(num_warmup):
            _ = indexer_loss_fwd_tilelang(query, key, topk_indices, scaling, chunk_offset=chunk_offset)
        torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(num_benchmark):
            _ = indexer_loss_fwd_tilelang(query, key, topk_indices, scaling, chunk_offset=chunk_offset)
        torch.cuda.synchronize()
        tl_time = (time.time() - start) / num_benchmark * 1000
        tl_peak = torch.cuda.max_memory_allocated() / (1024**3)
        results['tilelang'] = tl_time
        memory_stats['tilelang'] = tl_peak
        
        print(f">>> TileLang 性能: {tl_time:.3f} ms (加速: {triton_time/tl_time:.2f}x)")
        
        print(f"\n>>> 显存峰值")
        print(f"  基准显存:    {base_memory:.2f} GB")
        print(f"  Triton 峰值: {memory_stats['triton']:.2f} GB")
        print(f"  TileLang 峰值: {memory_stats['tilelang']:.2f} GB")
    except Exception as e:
        import traceback
        print(f"\n>>> TileLang 执行失败: {e}")
        traceback.print_exc()
        results['tilelang'] = float('inf')
        memory_stats['tilelang'] = 0.0
    
    return results, memory_stats


# ============================================================================
# 主测试入口
# ============================================================================

if __name__ == "__main__":
    # 精度测试配置（使用较小的规模以便于调试）
    accuracy_configs = [
        TestConfig(name="小规模", batch_size=1, num_heads=16, chunk_size=32, seq_len=64, head_dim=64, topk=32),
        TestConfig(name="中等规模", batch_size=1, num_heads=32, chunk_size=64, seq_len=128, head_dim=64, topk=64),
        TestConfig(name="大规模", batch_size=1, num_heads=64, chunk_size=128, seq_len=256, head_dim=128, topk=128),
    ]
    
    # ========== 前向精度测试 ==========
    test_fwd_accuracy(accuracy_configs)
    
    # ========== 性能对比测试 ==========
    test_performance_comparison(
        batch_size=1,
        num_heads=128,
        chunk_size=4 * 1024,
        seq_len=8 * 1024,
        head_dim=576,
        topk=2048,
        num_warmup=3,
        num_benchmark=10,
    )
