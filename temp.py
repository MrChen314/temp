"""
Triton实现的Index Loss计算

主要优化点:
1. 融合attention score计算、mask、softmax (FlashAttention-like)
2. 融合head维度的求和与KL散度计算
3. 不切chunk，直接处理完整序列
"""

import torch
import triton
import triton.language as tl


@triton.jit  
def _fused_index_loss_kernel(
    # Inputs
    Q_ptr,          # [batch, num_heads, seq_len, head_dim]
    K_ptr,          # [batch, seq_len, head_dim]
    IndexScore_ptr, # [batch, seq_len, seq_len]
    Mask_ptr,       # [batch, seq_len, seq_len]
    # Output
    Loss_ptr,       # [batch, seq_len]
    # Dimensions
    batch_size,
    num_heads: tl.constexpr,
    seq_len,
    head_dim: tl.constexpr,
    scaling,
    eps: tl.constexpr,
    # Strides for Q: [batch, num_heads, seq_len, head_dim]
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K: [batch, seq_len, head_dim]
    stride_kb, stride_ks, stride_kd,
    # Strides for IndexScore: [batch, seq_len, seq_len]
    stride_isb, stride_iss, stride_isk,
    # Strides for Mask: [batch, seq_len, seq_len]
    stride_mb, stride_ms, stride_mk,
    # Strides for Loss: [batch, seq_len]
    stride_lb, stride_ls,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    完全融合的index loss kernel
    每个program处理一个(batch, row)对
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    
    # ======= 步骤1: 计算每个head的softmax参数 =======
    # 存储每个head的max和sum
    head_max = tl.zeros([num_heads], dtype=tl.float32) - float("inf")
    head_sum = tl.zeros([num_heads], dtype=tl.float32)
    
    for h in tl.static_range(num_heads):
        # 加载query
        q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh + pid_row * stride_qs
        q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
        
        m_h = -float("inf")
        
        # 第一遍：找max
        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len
            
            # 加载key
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            # QK^T
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            # 加载并应用mask
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            m_h = tl.maximum(m_h, tl.max(qk))
        
        # 第二遍：计算sum
        s_h = 0.0
        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < seq_len
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            s_h += tl.sum(tl.exp(qk - m_h))
        
        # 更新head_max和head_sum
        head_max = tl.where(tl.arange(0, num_heads) == h, m_h, head_max)
        head_sum = tl.where(tl.arange(0, num_heads) == h, s_h, head_sum)
    
    # ======= 步骤2: 计算index_score的softmax参数 =======
    max_is = -float("inf")
    sum_is = 0.0
    
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        max_is = tl.maximum(max_is, tl.max(is_val))
    
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        sum_is += tl.sum(tl.exp(is_val - max_is))
    
    # ======= 步骤3: 计算attn分布的归一化分母 =======
    attn_norm = 0.0
    
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        attn_block = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for h in tl.static_range(num_heads):
            q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh + pid_row * stride_qs
            q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            # 获取该head的max和sum
            m_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_max, 0.0))
            s_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_sum, 0.0))
            
            attn_h = tl.exp(qk - m_h) / (s_h + 1e-10)
            attn_block += attn_h
        
        attn_norm += tl.sum(tl.where(n_mask, attn_block, 0.0))
    
    # ======= 步骤4: 计算KL散度 =======
    kl_sum = 0.0
    
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # 计算attention分布
        attn_block = tl.zeros([BLOCK_N], dtype=tl.float32)
        
        for h in tl.static_range(num_heads):
            q_ptr = Q_ptr + pid_batch * stride_qb + h * stride_qh + pid_row * stride_qs
            q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
            
            k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            
            qk = tl.sum(q[None, :] * k, axis=1) * scaling
            
            mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
            mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
            qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
            
            m_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_max, 0.0))
            s_h = tl.sum(tl.where(tl.arange(0, num_heads) == h, head_sum, 0.0))
            
            attn_h = tl.exp(qk - m_h) / (s_h + 1e-10)
            attn_block += attn_h
        
        # 归一化得到attn分布
        attn_dist = attn_block / (attn_norm + 1e-10) + eps
        
        # 计算index_prob
        is_ptrs = IndexScore_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_n * stride_isk
        is_val = tl.load(is_ptrs, mask=n_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        index_prob = tl.exp(is_val - max_is) / (sum_is + 1e-10) + eps
        
        # KL散度: attn * log(attn / index)
        kl = tl.where(n_mask, attn_dist * (tl.log(attn_dist) - tl.log(index_prob)), 0.0)
        kl_sum += tl.sum(kl)
    
    # 存储结果
    loss_ptr = Loss_ptr + pid_batch * stride_lb + pid_row * stride_ls
    tl.store(loss_ptr, kl_sum)


def compute_index_loss_triton(
    query,          # [batch, num_heads, seq_len, head_dim]
    key,            # [batch, seq_len, head_dim]
    index_score,    # [batch, seq_len, seq_len]
    index_mask,     # [batch, seq_len, seq_len]
    scaling,
):
    """
    高效的Triton实现index loss
    
    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, seq_len, head_dim]
        index_score: Index scores [batch, seq_len, seq_len]
        index_mask: Boolean mask [batch, seq_len, seq_len], True表示被mask
        scaling: Attention scaling factor
    
    Returns:
        Scalar KL divergence loss
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # 确保是contiguous
    query = query.contiguous()
    key = key.contiguous()
    index_score = index_score.contiguous()
    index_mask = index_mask.contiguous()
    
    # 输出
    loss = torch.zeros(batch_size, seq_len, device=query.device, dtype=torch.float32)
    
    # Block sizes
    BLOCK_N = min(1024, triton.next_power_of_2(seq_len))
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    # Grid: 每个(batch, row)一个program
    grid = (batch_size * seq_len,)
    
    eps = 1e-10
    
    _fused_index_loss_kernel[grid](
        query, key, index_score, index_mask, loss,
        batch_size, num_heads, seq_len, head_dim,
        scaling, eps,
        # strides
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        index_mask.stride(0), index_mask.stride(1), index_mask.stride(2),
        loss.stride(0), loss.stride(1),
        BLOCK_N, BLOCK_D,
    )
    
    return loss.mean()


def generate_topk_mask(batch_size, seq_len, topk, device='cuda'):
    """
    生成基于topk的index mask
    
    对于每个位置t，只保留前t个位置中的topk个（causal）
    
    Args:
        batch_size: batch大小
        seq_len: 序列长度
        topk: 每个位置保留的top k个
        device: 设备
    
    Returns:
        mask: [batch, seq_len, seq_len] bool tensor, True表示被mask掉
        indices: [batch, seq_len, topk] int tensor, topk的索引
    """
    # 创建 indices（向量化方式，避免慢速 for 循环）
    # 对于位置t，可选的范围是 [0, t]，所以max_val = t+1
    t_vals = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len, 1)
    max_vals = torch.clamp(t_vals + 1, min=1).expand(batch_size, seq_len, topk)
    random_floats = torch.rand(batch_size, seq_len, topk, device=device)
    indices = (random_floats * max_vals).to(torch.int64)
    
    # 创建mask: 默认全部为True（被mask），然后将topk位置设为False
    mask = torch.ones(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
    
    # 使用scatter将topk位置设为False
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, seq_len, topk)
    row_idx = torch.arange(seq_len, device=device).view(1, -1, 1).expand(batch_size, -1, topk)
    
    mask[batch_idx, row_idx, indices] = False
    
    # 同时应用causal mask: 位置 > 当前行的都要mask掉
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = mask | causal_mask.unsqueeze(0)
    
    return mask, indices


def pytorch_reference(query, key, index_score, index_mask, scaling):
    """
    PyTorch参考实现
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, seq_len, head_dim]
        index_score: [batch, seq_len, seq_len]
        index_mask: [batch, seq_len, seq_len] bool, True表示被mask
        scaling: attention缩放因子
    
    Returns:
        KL divergence loss
    """
    import torch.nn.functional as F
    
    eps = 1e-10
    
    # 计算attention: [batch, num_heads, seq_len, seq_len]
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(index_mask.unsqueeze(1), -1e9)
    attn = torch.softmax(attn, dim=-1)
    
    # 对heads求和并归一化: [batch, seq_len, seq_len]
    attn_sum = attn.sum(dim=1)
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    # Index score softmax
    index_score_masked = index_score.masked_fill(index_mask, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    # KL散度
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


def test_index_loss(
    batch_size: int = 2,
    num_heads: int = 4,
    seq_len: int = 256,
    head_dim: int = 64,
    topk: int = 32,
    seed: int = 42,
    num_warmup: int = 10,
    num_benchmark: int = 100,
):
    """
    测试Triton实现与PyTorch实现的一致性和性能
    
    Args:
        batch_size: batch大小
        num_heads: attention头数
        seq_len: 序列长度
        head_dim: 每个头的维度
        topk: 每个位置保留的top k个
        seed: 随机种子
        num_warmup: 预热迭代次数
        num_benchmark: 性能测试迭代次数
    """
    import time
    
    torch.manual_seed(seed)
    
    scaling = 1.0 / (head_dim ** 0.5)
    device = 'cuda'
    
    print(f"=" * 60)
    print(f"测试参数:")
    print(f"  batch_size = {batch_size}")
    print(f"  num_heads = {num_heads}")
    print(f"  seq_len = {seq_len}")
    print(f"  head_dim = {head_dim}")
    print(f"  topk = {topk}")
    print(f"  scaling = {scaling:.6f}")
    print(f"=" * 60)
    
    # 创建测试数据
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    
    # 生成基于topk的mask
    index_mask, topk_indices = generate_topk_mask(batch_size, seq_len, topk, device)
    
    print(f"\n生成的mask信息:")
    print(f"  index_mask shape: {index_mask.shape}")
    print(f"  topk_indices shape: {topk_indices.shape}")
    print(f"  每行未被mask的位置数: {(~index_mask[0, 0]).sum().item()}")
    
    # 测试正确性
    print(f"\n正确性测试:")
    ref_loss = pytorch_reference(query, key, index_score, index_mask, scaling)
    triton_loss = compute_index_loss_triton(query, key, index_score, index_mask, scaling)
    
    print(f"  PyTorch参考实现: {ref_loss.item():.6f}")
    print(f"  Triton实现: {triton_loss.item():.6f}")
    print(f"  差异: {abs(ref_loss.item() - triton_loss.item()):.6e}")
    
    # 性能测试
    print(f"\n性能测试 (warmup={num_warmup}, iterations={num_benchmark}):")
    
    # Warm up Triton
    for _ in range(num_warmup):
        _ = compute_index_loss_triton(query, key, index_score, index_mask, scaling)
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start = time.time()
    for _ in range(num_benchmark):
        _ = compute_index_loss_triton(query, key, index_score, index_mask, scaling)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_benchmark * 1000
    
    # Warm up PyTorch
    for _ in range(num_warmup):
        _ = pytorch_reference(query, key, index_score, index_mask, scaling)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_benchmark):
        _ = pytorch_reference(query, key, index_score, index_mask, scaling)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_benchmark * 1000
    
    print(f"  PyTorch时间: {pytorch_time:.3f} ms")
    print(f"  Triton时间: {triton_time:.3f} ms")
    
    if triton_time > 0:
        speedup = pytorch_time / triton_time
        print(f"  加速比: {speedup:.2f}x")
    
    # 内存使用估算
    print(f"\n内存使用估算:")
    attn_matrix_size = batch_size * num_heads * seq_len * seq_len * 4  # float32
    print(f"  Attention矩阵大小: {attn_matrix_size / 1024 / 1024:.2f} MB")
    print(f"  PyTorch实现需要存储完整attention矩阵")
    print(f"  Triton实现使用online算法，内存占用更小")
    
    return {
        'ref_loss': ref_loss.item(),
        'triton_loss': triton_loss.item(),
        'pytorch_time_ms': pytorch_time,
        'triton_time_ms': triton_time,
        'speedup': pytorch_time / triton_time if triton_time > 0 else float('inf'),
    }


def test_various_configs():
    """测试不同配置下的性能"""
    configs = [
        # (batch_size, num_heads, seq_len, head_dim, topk)
        (1, 8, 128, 64, 16),
        (1, 8, 256, 64, 32),
        (1, 8, 512, 64, 64),
        (2, 8, 256, 64, 32),
        (4, 4, 256, 64, 32),
        (1, 16, 256, 64, 32),
        (1, 8, 256, 128, 32),
    ]
    
    print("\n" + "=" * 80)
    print("不同配置下的性能测试")
    print("=" * 80)
    
    results = []
    for batch_size, num_heads, seq_len, head_dim, topk in configs:
        try:
            result = test_index_loss(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                topk=topk,
                num_warmup=5,
                num_benchmark=50,
            )
            results.append({
                'config': (batch_size, num_heads, seq_len, head_dim, topk),
                **result
            })
        except Exception as e:
            print(f"配置 {(batch_size, num_heads, seq_len, head_dim, topk)} 失败: {e}")
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("性能汇总")
    print("=" * 80)
    print(f"{'Config':<35} {'PyTorch(ms)':<15} {'Triton(ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        config_str = str(r['config'])
        print(f"{config_str:<35} {r['pytorch_time_ms']:<15.3f} {r['triton_time_ms']:<15.3f} {r['speedup']:<10.2f}x")


if __name__ == "__main__":
    # 默认测试
    test_index_loss(
        batch_size=2,
        num_heads=8,
        seq_len=256,
        head_dim=64,
        topk=32,
    )
    
    # 可选：运行多配置测试
    # test_various_configs()
