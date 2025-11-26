"""
IndexerBaselineFunction - 无量化版本（分块显存友好版）

使用 weights 进行加权计算，与 indexer_no_quant.py 保持一致。
反向传播使用分块计算以节省显存。
"""

import torch


class IndexerBaselineFunction(torch.autograd.Function):
    """
    无量化的 Indexer 基线实现（分块显存友好版）。
    
    使用 weights 对每个 head 进行加权计算。
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor):
        """
        无量化 Indexer 前向计算。
        
        参数:
            q: query tensor, shape = (b, m, h, d)
            k: key tensor, shape = (b, n, d)
            weights: 权重 tensor, 支持多种形状:
                     - (h, m): 标准形状
                     - (m, h): 已转置的形状
                     - (m, 1, h): 带中间维度的形状
        
        返回:
            index_score: shape = (b, m, n)
        """
        assert q.shape[-1] == 128, f"q last dim must be 128, got {q.shape[-1]}"
        assert k.shape[-1] == 128, f"k last dim must be 128, got {k.shape[-1]}"

        # 计算 softmax_scale
        softmax_scale = q.shape[-1] ** -0.5  # 1/sqrt(D)
        
        # 保存原始 weights 形状，用于反向传播时恢复
        weights_shape = weights.shape
        
        # 处理不同形状的 weights，统一转换为 (m, h)
        if weights.dim() == 2:
            # 可能是 (h, m) 或 (m, h)
            # 假设 weights 需要转置（原始是 (h, m)）
            weights_transposed = torch.transpose(weights, 0, 1).contiguous()  # (h, m) -> (m, h)
        elif weights.dim() == 3:
            # (m, 1, h) 形状，squeeze 中间维度
            weights_transposed = weights.squeeze(1)  # (m, 1, h) -> (m, h)
        else:
            raise ValueError(f"Unsupported weights shape: {weights.shape}")
        
        q_s_weights = weights_transposed * softmax_scale  # (m, h)

        # 调用核心计算函数
        index_score, cache = index_baseline_forward(q, k, q_s_weights)

        # 保存反向所需 tensor
        ctx.save_for_backward(q, k, q_s_weights)
        ctx.softmax_scale = softmax_scale
        ctx.cache = cache
        ctx.weights_shape = weights_shape  # 保存原始形状

        return index_score

    @staticmethod
    def backward(ctx, grad_output):
        """
        无量化 Indexer 反向计算。
        
        参数:
            grad_output: 输出梯度, shape = (b, m, n)
        
        返回:
            grad_q: q 的梯度, shape = (b, m, h, d)
            grad_k: k 的梯度, shape = (b, n, d)
            grad_weights: weights 的梯度, 形状与输入 weights 一致
        """
        q, k, q_s_weights = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        cache = ctx.cache
        weights_shape = ctx.weights_shape  # 原始 weights 形状

        grad_q, grad_k, grad_q_s = index_baseline_backward_tile(
            grad_output.contiguous(), q, k, q_s_weights, cache
        )
        
        # 计算 weights 的梯度
        # grad_q_s shape: (b, m, h)
        # 需要沿 batch 维度求和
        grad_q_s_sum = grad_q_s.sum(dim=0)  # (m, h)
        grad_weights_mh = grad_q_s_sum * softmax_scale  # (m, h)
        
        # 根据原始 weights 形状恢复梯度形状
        if len(weights_shape) == 2:
            # 原始是 (h, m)，需要转置
            grad_weights = torch.transpose(grad_weights_mh, 0, 1).contiguous()  # (h, m)
        elif len(weights_shape) == 3:
            # 原始是 (m, 1, h)，需要 unsqueeze
            grad_weights = grad_weights_mh.unsqueeze(1)  # (m, 1, h)
        else:
            grad_weights = grad_weights_mh
        
        return grad_q, grad_k, grad_weights


# ------------------------------------------------------------------
# 前向：使用 q_s 进行加权
# ------------------------------------------------------------------
def index_baseline_forward(
    q: torch.Tensor, 
    k: torch.Tensor, 
    q_s: torch.Tensor
) -> tuple:
    """
    无量化 Indexer 前向核心计算。
    
    参数:
        q: query tensor, shape = (b, m, h, d)
        k: key tensor, shape = (b, n, d)
        q_s: query 的缩放因子 (weights * softmax_scale), shape = (m, h) 或 (b, m, h)
    
    返回:
        index_score: shape = (b, m, n)
        cache: 缓存的中间结果，用于反向传播
    """
    # Step 1: 计算 q 和 k 的点积
    logits = torch.einsum('bmhd,bnd->bmhn', q, k)  # (B, M, H, N)
    
    # Step 2: ReLU 激活
    relu_logits = torch.relu(logits)
    
    # Step 3: 乘以 q_s（缩放因子）
    # q_s.unsqueeze(-1): (m, h) -> (m, h, 1) 或 (b, m, h) -> (b, m, h, 1)
    # 会广播到 (b, m, h, n)
    scaled_logits = relu_logits * q_s.unsqueeze(-1)
    
    # Step 4: 沿 head 维度求和
    logits_sum = scaled_logits.sum(dim=2)  # (B, M, N)
    
    # k_s 为全 1（无量化），所以 index_score = logits_sum
    index_score = logits_sum
    
    # 缓存中间结果
    cache = (logits, relu_logits)
    
    return index_score, cache


# ------------------------------------------------------------------
# 反向：分块计算以节省显存
# ------------------------------------------------------------------
def index_baseline_backward_tile(
    d_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    q_s: torch.Tensor,
    cache: tuple,
    tile_M: int = 128,
    tile_N: int = 128
) -> tuple:
    """
    无量化 Indexer 反向核心计算（分块显存友好版）。
    
    参数:
        d_out: 输出梯度, shape = (b, m, n)
        q: query tensor, shape = (b, m, h, d)
        k: key tensor, shape = (b, n, d)
        q_s: query 的缩放因子, shape = (m, h) 或 (b, m, h)
        cache: 缓存的中间结果 (logits, relu_logits)
        tile_M: M 方向的分块大小
        tile_N: N 方向的分块大小
    
    返回:
        d_q: q 的梯度, shape = (b, m, h, d)
        d_k: k 的梯度, shape = (b, n, d)
        d_q_s: q_s 的梯度, shape = (b, m, h)
    """
    logits, relu_logits = cache
    B, M, H, D = q.shape
    N = k.shape[1]
    
    # 判断 q_s 是 2D (m, h) 还是 3D (b, m, h)
    q_s_is_3d = (q_s.dim() == 3)

    # 输出梯度
    dq = torch.zeros_like(q)           # (B, M, H, D)
    dk = torch.zeros_like(k)           # (B, N, D)
    d_q_s = torch.zeros(B, M, H, device=q.device, dtype=q.dtype)  # (B, M, H)

    # 外循环：M 方向
    for i in range(0, M, tile_M):
        m1, m2 = i, min(i + tile_M, M)
        q_tile = q[:, m1:m2]                    # (B, tile_M, H, D)
        d_out_tile = d_out[:, m1:m2]            # (B, tile_M, N)
        
        # 根据 q_s 的维度选择切片方式
        if q_s_is_3d:
            q_s_tile = q_s[:, m1:m2, :]         # (B, tile_M, H)
        else:
            q_s_tile = q_s[m1:m2]               # (tile_M, H)

        # 内循环：N 方向
        for j in range(0, N, tile_N):
            n1, n2 = j, min(j + tile_N, N)

            # 1. 取出当前块需要的张量
            k_tile = k[:, n1:n2]                      # (B, tile_N, D)
            logits_tile = logits[:, m1:m2, :, n1:n2]  # (B, tile_M, H, tile_N)
            relu_logits_tile = relu_logits[:, m1:m2, :, n1:n2]  # (B, tile_M, H, tile_N)
            mask_tile = (logits_tile > 0).float()     # ReLU gate

            # 2. 反向传播链式法则
            # index_score = logits_sum (k_s=1)
            # d_logits_sum = d_out
            d_logits_sum_tile = d_out_tile[:, :, n1:n2]  # (B, tile_M, tile_N)
            
            # logits_sum = scaled_logits.sum(dim=2)
            # d_scaled_logits = d_logits_sum.unsqueeze(2).expand(...)
            d_scaled_logits_tile = d_logits_sum_tile.unsqueeze(2)  # (B, tile_M, 1, tile_N)
            
            # scaled_logits = relu_logits * q_s.unsqueeze(-1)
            # d_relu_logits = d_scaled_logits * q_s.unsqueeze(-1)
            # d_q_s += (d_scaled_logits * relu_logits).sum(dim=-1)
            q_s_tile_expanded = q_s_tile.unsqueeze(-1)  # (tile_M, H, 1) 或 (B, tile_M, H, 1)
            d_relu_logits_tile = d_scaled_logits_tile * q_s_tile_expanded  # (B, tile_M, H, tile_N)
            d_q_s[:, m1:m2] += (d_scaled_logits_tile * relu_logits_tile).sum(dim=-1)  # (B, tile_M, H)
            
            # relu_logits = relu(logits)
            # d_logits = d_relu_logits * (logits > 0)
            d_logits_tile = d_relu_logits_tile * mask_tile  # (B, tile_M, H, tile_N)

            # 3. 立即计算局部贡献并累加
            # logits = einsum('bmhd,bnd->bmhn', q, k)
            # d_q = einsum('bmhn,bnd->bmhd', d_logits, k)
            # d_k = einsum('bmhn,bmhd->bnd', d_logits, q)
            dq[:, m1:m2] += torch.einsum('bmhn,bnd->bmhd', d_logits_tile, k_tile)
            dk[:, n1:n2] += torch.einsum('bmhn,bmhd->bnd', d_logits_tile, q_tile)

            # 4. 释放当前块所有中间张量
            del d_logits_tile, d_relu_logits_tile, d_scaled_logits_tile
            del mask_tile, relu_logits_tile, logits_tile, k_tile
        
        del q_tile, d_out_tile, q_s_tile

    return dq, dk, d_q_s


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("=" * 60)
    print("IndexerBaselineFunction 分块版本测试")
    print("=" * 60)
    
    # 定义参数
    batch_size = 2
    seq_len_q = 4
    seq_len_k = 6
    num_heads = 8
    head_dim = 128  # 必须是 128
    
    print(f"\n参数设置:")
    print(f"  batch_size = {batch_size}")
    print(f"  seq_len_q = {seq_len_q}")
    print(f"  seq_len_k = {seq_len_k}")
    print(f"  num_heads = {num_heads}")
    print(f"  head_dim = {head_dim}")
    
    # 创建输入张量
    q = torch.randn(batch_size, seq_len_q, num_heads, head_dim, requires_grad=True)
    k = torch.randn(batch_size, seq_len_k, head_dim, requires_grad=True)
    weights = torch.randn(num_heads, seq_len_q, requires_grad=True)
    
    print(f"\n输入张量:")
    print(f"  q shape: {q.shape}")
    print(f"  k shape: {k.shape}")
    print(f"  weights shape: {weights.shape}")
    
    # 前向计算
    print(f"\n{'='*60}")
    print("前向计算")
    print("=" * 60)
    
    index_score = IndexerBaselineFunction.apply(q, k, weights)
    
    print(f"index_score shape: {index_score.shape}")
    print(f"index_score:\n{index_score}")
    
    # 反向计算
    print(f"\n{'='*60}")
    print("反向计算")
    print("=" * 60)
    
    grad_output = torch.ones_like(index_score)
    index_score.backward(grad_output)
    
    print(f"q.grad shape: {q.grad.shape}")
    print(f"k.grad shape: {k.grad.shape}")
    print(f"weights.grad shape: {weights.grad.shape}")
    
    print(f"\nq.grad (前5个元素): {q.grad.flatten()[:5]}")
    print(f"k.grad (前5个元素): {k.grad.flatten()[:5]}")
    print(f"weights.grad:\n{weights.grad}")
    
    # 数值梯度检查
    print(f"\n{'='*60}")
    print("梯度检查 (gradcheck)")
    print("=" * 60)
    
    q_small = torch.randn(1, 2, 2, 128, requires_grad=True, dtype=torch.float64)
    k_small = torch.randn(1, 3, 128, requires_grad=True, dtype=torch.float64)
    weights_small = torch.randn(2, 2, requires_grad=True, dtype=torch.float64)
    
    try:
        result = torch.autograd.gradcheck(
            IndexerBaselineFunction.apply,
            (q_small, k_small, weights_small),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3
        )
        print(f"梯度检查结果: {'✓ 通过' if result else '✗ 失败'}")
    except Exception as e:
        print(f"梯度检查异常: {e}")
    
    print(f"\n{'='*60}")
    print("测试完成")
    print("=" * 60)
