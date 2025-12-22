"""
Triton Unfused实现 - 用于调试精度问题

将计算拆分为4个独立步骤，每步可单独验证：
1. Attention Softmax: QK^T * scaling + mask + softmax
2. Head Sum + Normalize: sum over heads + normalize
3. Index Score Softmax: masked softmax
4. KL Divergence: kl_div计算
"""

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================================
# Step 1: Attention Softmax Kernel
# 输入: query [B, H, S, D], key [B, S, D], mask [B, S, S]
# 输出: attn [B, H, S, S]
# ============================================================================

@triton.jit
def _attention_softmax_kernel(
    Q_ptr,          # [batch, num_heads, seq_len, head_dim]
    K_ptr,          # [batch, seq_len, head_dim]
    Mask_ptr,       # [batch, seq_len, seq_len]
    Out_ptr,        # [batch, num_heads, seq_len, seq_len]
    # Dimensions
    batch_size,
    num_heads,
    seq_len,
    head_dim: tl.constexpr,
    scaling,
    # Strides for Q
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K
    stride_kb, stride_ks, stride_kd,
    # Strides for Mask
    stride_mb, stride_ms, stride_mk,
    # Strides for Out
    stride_ob, stride_oh, stride_os, stride_ok,
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    计算一行的attention softmax
    每个program处理一个 (batch, head, row)
    """
    pid = tl.program_id(0)
    num_rows_per_batch = num_heads * seq_len
    pid_batch = pid // num_rows_per_batch
    remainder = pid % num_rows_per_batch
    pid_head = remainder // seq_len
    pid_row = remainder % seq_len
    
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    
    # 加载query: [head_dim]
    q_ptr = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh + pid_row * stride_qs
    q = tl.load(q_ptr + offs_d * stride_qd, mask=d_mask, other=0.0)
    
    # 第一遍: 计算max
    m_max = -float("inf")
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # 加载key: [BLOCK_N, head_dim]
        k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        
        # QK^T * scaling
        qk = tl.sum(q[None, :] * k, axis=1) * scaling
        
        # 加载mask
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
        
        m_max = tl.maximum(m_max, tl.max(qk))
    
    # 第二遍: 计算sum(exp(x - max))
    l_sum = 0.0
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        
        qk = tl.sum(q[None, :] * k, axis=1) * scaling
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
        
        l_sum += tl.sum(tl.exp(qk - m_max))
    
    # 第三遍: 计算softmax并写出
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        k_ptrs = K_ptr + pid_batch * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        
        qk = tl.sum(q[None, :] * k, axis=1) * scaling
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_n * stride_mk
        mask_val = tl.load(mask_ptrs, mask=n_mask, other=True)
        qk = tl.where(mask_val | ~n_mask, -float("inf"), qk)
        
        # softmax
        p = tl.exp(qk - m_max) / l_sum
        
        # 写出
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_head * stride_oh + pid_row * stride_os + offs_n * stride_ok
        tl.store(out_ptrs, p, mask=n_mask)


def attention_softmax_triton(query, key, mask, scaling):
    """
    Step 1: 计算masked attention softmax
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, seq_len, head_dim]
        mask: [batch, seq_len, seq_len] bool, True表示mask掉
        scaling: float
    
    Returns:
        attn: [batch, num_heads, seq_len, seq_len]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    query = query.contiguous()
    key = key.contiguous()
    mask = mask.contiguous()
    
    # 输出
    attn = torch.zeros(batch_size, num_heads, seq_len, seq_len, 
                       device=query.device, dtype=query.dtype)
    
    BLOCK_N = min(1024, triton.next_power_of_2(seq_len))
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    grid = (batch_size * num_heads * seq_len,)
    
    _attention_softmax_kernel[grid](
        query, key, mask, attn,
        batch_size, num_heads, seq_len, head_dim, scaling,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        BLOCK_N, BLOCK_D,
    )
    
    return attn


def attention_softmax_ref(query, key, mask, scaling):
    """PyTorch参考实现 Step 1"""
    # query: [batch, num_heads, seq_len, head_dim]
    # key: [batch, seq_len, head_dim] -> [batch, 1, head_dim, seq_len]
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(mask.unsqueeze(1), -1e9)
    attn = torch.softmax(attn, dim=-1)
    return attn


# ============================================================================
# Step 2: Head Sum + Normalize Kernel
# 输入: attn [B, H, S, S]
# 输出: attn_dist [B, S, S]
# ============================================================================

@triton.jit
def _head_sum_normalize_kernel(
    Attn_ptr,       # [batch, num_heads, seq_len, seq_len]
    Out_ptr,        # [batch, seq_len, seq_len]
    # Dimensions
    batch_size,
    num_heads: tl.constexpr,
    seq_len,
    eps: tl.constexpr,
    # Strides for Attn
    stride_ab, stride_ah, stride_as, stride_ak,
    # Strides for Out
    stride_ob, stride_os, stride_ok,
    # Block size
    BLOCK_K: tl.constexpr,
):
    """
    对每个(batch, row)，先对heads求和，再归一化
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    # 第一遍: 计算sum over heads 和 total sum
    total_sum = 0.0
    
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        # 对所有heads求和
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h in tl.static_range(num_heads):
            attn_ptrs = Attn_ptr + pid_batch * stride_ab + h * stride_ah + pid_row * stride_as + offs_k * stride_ak
            attn_val = tl.load(attn_ptrs, mask=k_mask, other=0.0)
            acc += attn_val
        
        total_sum += tl.sum(tl.where(k_mask, acc, 0.0))
    
    # 第二遍: 归一化并写出
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for h in tl.static_range(num_heads):
            attn_ptrs = Attn_ptr + pid_batch * stride_ab + h * stride_ah + pid_row * stride_as + offs_k * stride_ak
            attn_val = tl.load(attn_ptrs, mask=k_mask, other=0.0)
            acc += attn_val
        
        # 归一化
        out_val = acc / (total_sum + eps)
        
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_row * stride_os + offs_k * stride_ok
        tl.store(out_ptrs, out_val, mask=k_mask)


def head_sum_normalize_triton(attn, eps=1e-10):
    """
    Step 2: 对attention在head维度求和并归一化
    
    Args:
        attn: [batch, num_heads, seq_len, seq_len]
        eps: 防止除零的小数
    
    Returns:
        attn_dist: [batch, seq_len, seq_len]
    """
    batch_size, num_heads, seq_len, _ = attn.shape
    
    attn = attn.contiguous()
    
    attn_dist = torch.zeros(batch_size, seq_len, seq_len, 
                            device=attn.device, dtype=attn.dtype)
    
    BLOCK_K = min(1024, triton.next_power_of_2(seq_len))
    
    grid = (batch_size * seq_len,)
    
    _head_sum_normalize_kernel[grid](
        attn, attn_dist,
        batch_size, num_heads, seq_len, eps,
        attn.stride(0), attn.stride(1), attn.stride(2), attn.stride(3),
        attn_dist.stride(0), attn_dist.stride(1), attn_dist.stride(2),
        BLOCK_K,
    )
    
    return attn_dist


def head_sum_normalize_ref(attn, eps=1e-10):
    """PyTorch参考实现 Step 2"""
    attn_sum = attn.sum(dim=1)  # [batch, seq_len, seq_len]
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    return attn_dist


# ============================================================================
# Step 3: Index Score Softmax Kernel
# 输入: index_score [B, S, S], mask [B, S, S]
# 输出: index_prob [B, S, S]
# ============================================================================

@triton.jit
def _index_score_softmax_kernel(
    IS_ptr,         # [batch, seq_len, seq_len]
    Mask_ptr,       # [batch, seq_len, seq_len]
    Out_ptr,        # [batch, seq_len, seq_len]
    # Dimensions
    batch_size,
    seq_len,
    eps: tl.constexpr,
    # Strides for IS
    stride_isb, stride_iss, stride_isk,
    # Strides for Mask
    stride_mb, stride_ms, stride_mk,
    # Strides for Out
    stride_ob, stride_os, stride_ok,
    # Block size
    BLOCK_K: tl.constexpr,
):
    """
    计算masked softmax for index_score
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    # 第一遍: 找max
    m_max = -float("inf")
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        is_ptrs = IS_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        mask_val = tl.load(mask_ptrs, mask=k_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        m_max = tl.maximum(m_max, tl.max(is_val))
    
    # 第二遍: 计算sum(exp)
    l_sum = 0.0
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        is_ptrs = IS_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        mask_val = tl.load(mask_ptrs, mask=k_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        l_sum += tl.sum(tl.exp(is_val - m_max))
    
    # 第三遍: 计算softmax并写出
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        is_ptrs = IS_ptr + pid_batch * stride_isb + pid_row * stride_iss + offs_k * stride_isk
        is_val = tl.load(is_ptrs, mask=k_mask, other=-float("inf"))
        
        mask_ptrs = Mask_ptr + pid_batch * stride_mb + pid_row * stride_ms + offs_k * stride_mk
        mask_val = tl.load(mask_ptrs, mask=k_mask, other=True)
        is_val = tl.where(mask_val, -float("inf"), is_val)
        
        # softmax + eps
        p = tl.exp(is_val - m_max) / l_sum + eps
        
        out_ptrs = Out_ptr + pid_batch * stride_ob + pid_row * stride_os + offs_k * stride_ok
        tl.store(out_ptrs, p, mask=k_mask)


def index_score_softmax_triton(index_score, mask, eps=1e-10):
    """
    Step 3: 计算index_score的masked softmax
    
    Args:
        index_score: [batch, seq_len, seq_len]
        mask: [batch, seq_len, seq_len] bool, True表示mask掉
        eps: 加到softmax结果上的小数
    
    Returns:
        index_prob: [batch, seq_len, seq_len]
    """
    batch_size, seq_len, _ = index_score.shape
    
    index_score = index_score.contiguous()
    mask = mask.contiguous()
    
    index_prob = torch.zeros_like(index_score)
    
    BLOCK_K = min(1024, triton.next_power_of_2(seq_len))
    
    grid = (batch_size * seq_len,)
    
    _index_score_softmax_kernel[grid](
        index_score, mask, index_prob,
        batch_size, seq_len, eps,
        index_score.stride(0), index_score.stride(1), index_score.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        index_prob.stride(0), index_prob.stride(1), index_prob.stride(2),
        BLOCK_K,
    )
    
    return index_prob


def index_score_softmax_ref(index_score, mask, eps=1e-10):
    """PyTorch参考实现 Step 3"""
    index_score_masked = index_score.masked_fill(mask, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    return index_prob


# ============================================================================
# Step 4: KL Divergence Kernel
# 输入: attn_dist [B, S, S], index_prob [B, S, S]
# 输出: loss scalar
# ============================================================================

@triton.jit
def _kl_div_kernel(
    AttnDist_ptr,   # [batch, seq_len, seq_len]
    IndexProb_ptr,  # [batch, seq_len, seq_len]
    Loss_ptr,       # [batch * seq_len] 中间结果
    # Dimensions
    batch_size,
    seq_len,
    eps: tl.constexpr,
    # Strides for AttnDist
    stride_ab, stride_as, stride_ak,
    # Strides for IndexProb
    stride_ib, stride_is, stride_ik,
    # Block size
    BLOCK_K: tl.constexpr,
):
    """
    计算KL散度: sum(attn_dist * (log(attn_dist) - log(index_prob)))
    注意: PyTorch的kl_div(input.log(), target)计算的是 target * (log(target) - input)
    """
    pid = tl.program_id(0)
    pid_batch = pid // seq_len
    pid_row = pid % seq_len
    
    kl_sum = 0.0
    
    for start_k in range(0, seq_len, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < seq_len
        
        # 加载attn_dist (+ eps for numerical stability)
        ad_ptrs = AttnDist_ptr + pid_batch * stride_ab + pid_row * stride_as + offs_k * stride_ak
        attn_dist = tl.load(ad_ptrs, mask=k_mask, other=0.0) + eps
        
        # 加载index_prob
        ip_ptrs = IndexProb_ptr + pid_batch * stride_ib + pid_row * stride_is + offs_k * stride_ik
        index_prob = tl.load(ip_ptrs, mask=k_mask, other=1.0)  # 避免log(0)
        
        # KL: attn_dist * (log(attn_dist) - log(index_prob))
        kl = attn_dist * (tl.log(attn_dist) - tl.log(index_prob))
        kl = tl.where(k_mask, kl, 0.0)
        
        kl_sum += tl.sum(kl)
    
    # 存储每行的KL和
    tl.store(Loss_ptr + pid, kl_sum)


def kl_div_triton(attn_dist, index_prob, eps=1e-10):
    """
    Step 4: 计算KL散度
    
    Args:
        attn_dist: [batch, seq_len, seq_len]
        index_prob: [batch, seq_len, seq_len]
        eps: 数值稳定性
    
    Returns:
        loss: scalar
    """
    batch_size, seq_len, _ = attn_dist.shape
    
    attn_dist = attn_dist.contiguous()
    index_prob = index_prob.contiguous()
    
    # 中间结果
    loss_per_row = torch.zeros(batch_size * seq_len, device=attn_dist.device, dtype=torch.float32)
    
    BLOCK_K = min(1024, triton.next_power_of_2(seq_len))
    
    grid = (batch_size * seq_len,)
    
    _kl_div_kernel[grid](
        attn_dist, index_prob, loss_per_row,
        batch_size, seq_len, eps,
        attn_dist.stride(0), attn_dist.stride(1), attn_dist.stride(2),
        index_prob.stride(0), index_prob.stride(1), index_prob.stride(2),
        BLOCK_K,
    )
    
    # batchmean: sum / batch_size
    return loss_per_row.sum() / batch_size


def kl_div_ref(attn_dist, index_prob, eps=1e-10):
    """PyTorch参考实现 Step 4"""
    # F.kl_div(input, target) computes: target * (log(target) - input)
    # 这里 input = index_prob.log(), target = attn_dist + eps
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    return kl_loss


# ============================================================================
# 组合函数
# ============================================================================

def compute_index_loss_unfused(query, key, index_score, index_mask, scaling):
    """
    Unfused版本: 串联4个步骤
    """
    # Step 1
    attn = attention_softmax_triton(query, key, index_mask, scaling)
    # Step 2
    attn_dist = head_sum_normalize_triton(attn)
    # Step 3
    index_prob = index_score_softmax_triton(index_score, index_mask)
    # Step 4
    loss = kl_div_triton(attn_dist, index_prob)
    
    return loss


def pytorch_reference_full(query, key, index_score, index_mask, scaling):
    """完整的PyTorch参考实现"""
    eps = 1e-10
    
    # Step 1
    attn = torch.matmul(query, key.unsqueeze(1).transpose(-1, -2)) * scaling
    attn = attn.masked_fill(index_mask.unsqueeze(1), -1e9)
    attn = torch.softmax(attn, dim=-1)
    
    # Step 2
    attn_sum = attn.sum(dim=1)
    attn_dist = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    
    # Step 3
    index_score_masked = index_score.masked_fill(index_mask, -1e9)
    index_prob = torch.softmax(index_score_masked, dim=-1) + eps
    
    # Step 4
    kl_loss = F.kl_div(index_prob.log(), attn_dist + eps, reduction='batchmean')
    
    return kl_loss


# ============================================================================
# 测试函数
# ============================================================================

def generate_topk_mask(batch_size, seq_len, topk, device='cuda'):
    """生成基于topk的index mask"""
    t_vals = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len, 1)
    max_vals = torch.clamp(t_vals + 1, min=1).expand(batch_size, seq_len, topk)
    random_floats = torch.rand(batch_size, seq_len, topk, device=device)
    indices = (random_floats * max_vals).to(torch.int64)
    
    mask = torch.ones(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, seq_len, topk)
    row_idx = torch.arange(seq_len, device=device).view(1, -1, 1).expand(batch_size, -1, topk)
    mask[batch_idx, row_idx, indices] = False
    
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = mask | causal_mask.unsqueeze(0)
    
    return mask, indices


def test_step1(batch_size=2, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试 Step 1: Attention Softmax"""
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    mask, _ = generate_topk_mask(batch_size, seq_len, topk, device)
    
    ref = attention_softmax_ref(query, key, mask, scaling)
    tri = attention_softmax_triton(query, key, mask, scaling)
    
    diff = (ref - tri).abs().max().item()
    print(f"Step 1 (Attention Softmax):")
    print(f"  Max diff: {diff:.6e}")
    print(f"  Ref sum: {ref.sum().item():.6f}, Triton sum: {tri.sum().item():.6f}")
    
    return diff < 1e-4


def test_step2(batch_size=2, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试 Step 2: Head Sum + Normalize"""
    torch.manual_seed(seed)
    device = 'cuda'
    
    attn = torch.rand(batch_size, num_heads, seq_len, seq_len, device=device, dtype=torch.float32)
    # 确保每行和为1（模拟softmax输出）
    attn = attn / attn.sum(dim=-1, keepdim=True)
    
    ref = head_sum_normalize_ref(attn)
    tri = head_sum_normalize_triton(attn)
    
    diff = (ref - tri).abs().max().item()
    print(f"Step 2 (Head Sum + Normalize):")
    print(f"  Max diff: {diff:.6e}")
    print(f"  Ref sum: {ref.sum().item():.6f}, Triton sum: {tri.sum().item():.6f}")
    
    return diff < 1e-5


def test_step3(batch_size=2, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试 Step 3: Index Score Softmax"""
    torch.manual_seed(seed)
    device = 'cuda'
    
    index_score = torch.randn(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    mask, _ = generate_topk_mask(batch_size, seq_len, topk, device)
    
    ref = index_score_softmax_ref(index_score, mask)
    tri = index_score_softmax_triton(index_score, mask)
    
    diff = (ref - tri).abs().max().item()
    print(f"Step 3 (Index Score Softmax):")
    print(f"  Max diff: {diff:.6e}")
    print(f"  Ref sum: {ref.sum().item():.6f}, Triton sum: {tri.sum().item():.6f}")
    
    return diff < 1e-5


def test_step4(batch_size=2, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试 Step 4: KL Divergence"""
    torch.manual_seed(seed)
    device = 'cuda'
    
    # 生成有效的概率分布
    attn_dist = torch.rand(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    attn_dist = attn_dist / attn_dist.sum(dim=-1, keepdim=True)
    
    index_prob = torch.rand(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    index_prob = index_prob / index_prob.sum(dim=-1, keepdim=True) + 1e-10
    
    ref = kl_div_ref(attn_dist, index_prob)
    tri = kl_div_triton(attn_dist, index_prob)
    
    diff = abs(ref.item() - tri.item())
    print(f"Step 4 (KL Divergence):")
    print(f"  Ref: {ref.item():.6f}, Triton: {tri.item():.6f}")
    print(f"  Diff: {diff:.6e}")
    
    return diff < 1e-4


def test_all_steps(batch_size=2, num_heads=8, seq_len=256, head_dim=64, topk=32, seed=42):
    """测试所有步骤串联"""
    torch.manual_seed(seed)
    device = 'cuda'
    scaling = 1.0 / (head_dim ** 0.5)
    
    print("=" * 60)
    print(f"测试参数:")
    print(f"  batch_size={batch_size}, num_heads={num_heads}")
    print(f"  seq_len={seq_len}, head_dim={head_dim}, topk={topk}")
    print("=" * 60)
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=torch.float32)
    index_score = torch.randn(batch_size, seq_len, seq_len, device=device, dtype=torch.float32)
    mask, _ = generate_topk_mask(batch_size, seq_len, topk, device)
    
    # 分步对比
    print("\n分步验证:")
    
    # Step 1
    attn_ref = attention_softmax_ref(query, key, mask, scaling)
    attn_tri = attention_softmax_triton(query, key, mask, scaling)
    diff1 = (attn_ref - attn_tri).abs().max().item()
    print(f"  Step 1 (Attention Softmax): max_diff = {diff1:.6e}")
    
    # Step 2
    attn_dist_ref = head_sum_normalize_ref(attn_ref)
    attn_dist_tri = head_sum_normalize_triton(attn_tri)
    diff2 = (attn_dist_ref - attn_dist_tri).abs().max().item()
    print(f"  Step 2 (Head Sum + Normalize): max_diff = {diff2:.6e}")
    
    # Step 3
    index_prob_ref = index_score_softmax_ref(index_score, mask)
    index_prob_tri = index_score_softmax_triton(index_score, mask)
    diff3 = (index_prob_ref - index_prob_tri).abs().max().item()
    print(f"  Step 3 (Index Score Softmax): max_diff = {diff3:.6e}")
    
    # Step 4
    kl_ref = kl_div_ref(attn_dist_ref, index_prob_ref)
    kl_tri = kl_div_triton(attn_dist_tri, index_prob_tri)
    diff4 = abs(kl_ref.item() - kl_tri.item())
    print(f"  Step 4 (KL Divergence): diff = {diff4:.6e}")
    
    # 完整对比
    print("\n完整流程对比:")
    full_ref = pytorch_reference_full(query, key, index_score, mask, scaling)
    full_tri = compute_index_loss_unfused(query, key, index_score, mask, scaling)
    
    print(f"  PyTorch完整实现: {full_ref.item():.6f}")
    print(f"  Triton Unfused: {full_tri.item():.6f}")
    print(f"  差异: {abs(full_ref.item() - full_tri.item()):.6e}")
    
    return {
        'step1_diff': diff1,
        'step2_diff': diff2,
        'step3_diff': diff3,
        'step4_diff': diff4,
        'full_ref': full_ref.item(),
        'full_triton': full_tri.item(),
    }


def test_unfused(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 256,
    head_dim: int = 64,
    topk: int = 32,
    seed: int = 42,
):
    """
    测试unfused实现的主入口
    """
    print("=" * 60)
    print("Triton Unfused 精度测试")
    print("=" * 60)
    
    print("\n>>> 测试 Step 1: Attention Softmax")
    test_step1(batch_size, num_heads, seq_len, head_dim, topk, seed)
    
    print("\n>>> 测试 Step 2: Head Sum + Normalize")
    test_step2(batch_size, num_heads, seq_len, head_dim, topk, seed)
    
    print("\n>>> 测试 Step 3: Index Score Softmax")
    test_step3(batch_size, num_heads, seq_len, head_dim, topk, seed)
    
    print("\n>>> 测试 Step 4: KL Divergence")
    test_step4(batch_size, num_heads, seq_len, head_dim, topk, seed)
    
    print("\n>>> 测试所有步骤串联")
    results = test_all_steps(batch_size, num_heads, seq_len, head_dim, topk, seed)
    
    return results


if __name__ == "__main__":
    test_unfused(
        batch_size=2,
        num_heads=8,
        seq_len=256,
        head_dim=64,
        topk=32,
    )

