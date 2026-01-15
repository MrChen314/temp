# ruff: noqa
"""
FlashMLA Sparse Forward 精度与性能测试

对比 flash_mla_sparse_fwd 与 ref_indexer_loss_fwd_interface 的前向精度和性能。
需要对 flash_mla 的 p_out 输出进行后处理计算 attn_sum。

后处理方案:
- flash_mla 输出: p_out = exp2(qk * scale - max_logits)
- lse = log2(sum of exp2) + max_logits
- attn_prob = p_out / exp2(lse - max_logits)
- attn_sum = attn_prob.sum(dim=1)
"""

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch

from flash_mla import flash_mla_sparse_fwd


# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class TestConfig:
    """测试配置"""
    name: str
    batch_size: int = 1  # flash_mla 无 batch 维度，固定为 1
    num_heads: int = 128
    chunk_size: int = 256
    seq_len: int = 256
    head_dim: int = 576  # 固定为 576
    d_v: int = 512       # 固定为 512
    topk: int = 128
    seed: int = 42
    
    def __str__(self):
        return (f"heads={self.num_heads}, chunk={self.chunk_size}, "
                f"seq={self.seq_len}, topk={self.topk}")


# ============================================================================
# 后处理函数：将 flash_mla 输出转换为 attn_sum
# ============================================================================

def flash_mla_to_attn_sum(
    p_out: torch.Tensor,      # [s_q, h_q, topk]
    max_logits: torch.Tensor, # [s_q, h_q]
    lse: torch.Tensor,        # [s_q, h_q]
    sm_scale
) -> torch.Tensor:
    """
    将 flash_mla_sparse_fwd 的输出转换为 attn_sum。
    
    计算公式:
    - p_out = qk矩阵乘积，掩码位置为-inf
    - lse = log2(sum of exp2) + max_logits
    - sum_exp = exp2(lse - max_logits) = sum of exp2
    - attn_prob = p_out / sum_exp
    - attn_sum = sum(attn_prob, dim=heads)
    
    Args:
        p_out: [s_q, h_q, topk] - attention 概率（未归一化）
        max_logits: [s_q, h_q] - 最大 logits（已乘以 scale*log2e）
        lse: [s_q, h_q] - log-sum-exp（log2 空间）
    
    Returns:
        attn_sum: [s_q, topk] - 所有 head 的 attention 概率求和
    """
    # p_out 是原始 qk 矩阵乘积结果 [s_q, h_q, topk]
    # max_logits 是每个位置的最大logits [s_q, h_q]
    # lse 是 log2(sum_exp) + max_logits [s_q, h_q]
    
    # 化简后的计算：
    # attn_prob = exp2(p_out * sm_scale - lse)
    # attn_prob = torch.exp2(p_out.to(torch.bfloat16) * sm_scale - lse.to(torch.bfloat16).unsqueeze(-1))  # [s_q, h_q, topk]
    
    # 沿 head 维度求和
    # attn_sum = attn_prob.sum(dim=1)  # [s_q, topk]


    ##===复杂计算
    # p_out 是原始 qk 矩阵乘积结果
    # 需要先计算 exp2(p_out * sm_scale - max_logits)
    # exp_scores = torch.exp2(p_out * sm_scale - max_logits.unsqueeze(-1))  # [s_q, h_q, topk]
    
    # # 计算 softmax 分母：sum_exp = exp2(lse - max_logits)
    # sum_exp = torch.exp2(lse - max_logits)  # [s_q, h_q]
    
    # # 计算归一化的 attention 概率
    # attn_prob = exp_scores / sum_exp.unsqueeze(-1)  # [s_q, h_q, topk]
    
    # # 沿 head 维度求和
    # attn_sum = attn_prob.sum(dim=1)  # [s_q, topk]




    # torch原生
    attn_score = p_out.to(torch.bfloat16) * sm_scale
    
    # Softmax
    attn_prob = torch.softmax(attn_score, dim=-1)  # [s_q, h_q, topk]
    
    # 沿 head 维度求和
    attn_sum = attn_prob.sum(dim=1)  # [s_q, topk]
    
    return attn_sum


# ============================================================================
# PyTorch 参考实现
# ============================================================================

def ref_attn_sum_pytorch(
    q: torch.Tensor,       # [s_q, h_q, d_qk]
    kv: torch.Tensor,      # [s_kv, h_kv, d_qk]
    indices: torch.Tensor, # [s_q, h_kv, topk]
    sm_scale: float,
) -> torch.Tensor:
    """
    PyTorch 参考实现: 计算 sparse attention 的 attn_sum。
    
    Args:
        q: [s_q, h_q, d_qk] - query
        kv: [s_kv, h_kv, d_qk] - key-value（h_kv 通常为 1）
        indices: [s_q, h_kv, topk] - 稀疏索引
        sm_scale: softmax scale factor
    
    Returns:
        attn_sum: [s_q, topk] - 所有 head 的 attention 概率求和
    """
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, _ = kv.shape
    _, _, topk = indices.shape
    
    # q = q.float()
    # kv = kv.float()
    
    # 获取索引（假设 h_kv=1）
    indices_2d = indices[:, 0, :]  # [s_q, topk]
    invalid_mask = (indices_2d < 0) | (indices_2d >= s_kv)
    
    # 收集 KV
    indices_clamped = indices_2d.masked_fill(invalid_mask, 0).flatten()
    k_selected = torch.index_select(kv[:, 0, :], 0, indices_clamped).view(s_q, topk, d_qk)  # [s_q, topk, d_qk]
    
    # 计算 attention scores: Q @ K^T
    attn_score = torch.einsum('shd,std->sht', q, k_selected)  # [s_q, h_q, topk]
    
    # 应用 mask 和 scale
    attn_score.masked_fill_(invalid_mask.unsqueeze(1), float('-inf'))
    attn_score *= sm_scale
    
    # Softmax
    attn_prob = torch.softmax(attn_score, dim=-1)  # [s_q, h_q, topk]
    
    # 沿 head 维度求和
    attn_sum = attn_prob.sum(dim=1)  # [s_q, topk]
    
    return attn_sum


# ============================================================================
# 精度测试
# ============================================================================

def run_accuracy_test(config: TestConfig, device: str = 'cuda') -> dict:
    """
    运行精度测试：对比 flash_mla_sparse_fwd 与 PyTorch 参考实现的 attn_sum。
    
    Args:
        config: 测试配置
        device: 设备
    
    Returns:
        测试结果字典
    """
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # 生成随机输入
    q = torch.randn(config.chunk_size, config.num_heads, config.head_dim, 
                    device=device, dtype=torch.bfloat16) / 10
    kv = torch.randn(config.seq_len, 1, config.head_dim, 
                     device=device, dtype=torch.bfloat16) / 10
    
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    
    # 生成 indices
    indices = torch.full((config.chunk_size, 1, config.topk), 
                         config.seq_len, dtype=torch.int32, device=device)
    for t in range(config.chunk_size):
        max_valid_idx = min(max(1, t + 1), config.seq_len)
        num_valid = min(config.topk, max_valid_idx)
        i_i = torch.randperm(max_valid_idx, device=device)[:num_valid]
        indices[t, 0, :num_valid] = i_i.to(torch.int32)
    
    # 计算 sm_scale
    sm_scale = 1.0 / math.sqrt(config.head_dim)
    
    # Flash MLA 前向
    out, max_logits, lse, p_out = flash_mla_sparse_fwd(q, kv, indices, sm_scale, config.d_v)
    
    # 后处理得到 attn_sum
    flash_attn_sum = flash_mla_to_attn_sum(p_out, max_logits, lse, sm_scale)
    
    # PyTorch 参考实现
    ref_attn_sum = ref_attn_sum_pytorch(q, kv, indices, sm_scale)
    
    # 计算误差
    flash_attn_sum_f32 = flash_attn_sum.float()
    ref_attn_sum_f32 = ref_attn_sum.float()
    
    def calc_diff(a, b):
        abs_diff = torch.abs(a - b)
        max_diff = abs_diff.max().item()
        rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item() * 100
        return max_diff, rel_diff
    
    max_diff, rel_diff = calc_diff(ref_attn_sum_f32, flash_attn_sum_f32.to(ref_attn_sum_f32.dtype))
    passed = rel_diff < 1e-3  # relative error < 0.001%
    
    return {
        'config': config,
        'max_diff': max_diff,
        'rel_diff': rel_diff,
        'flash_max': flash_attn_sum_f32.abs().max().item(),
        'ref_max': ref_attn_sum_f32.abs().max().item(),
        'passed': passed
    }


def test_accuracy(configs: List[TestConfig]) -> List[dict]:
    """批量运行精度测试"""
    print("\n" + "=" * 100)
    print("前向精度测试 (Flash MLA attn_sum vs PyTorch attn_sum)")
    print("=" * 100)
    
    results = []
    for config in configs:
        try:
            result = run_accuracy_test(config)
            results.append(result)
        except Exception as e:
            print(f"跳过测试 {config.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'Name':<15} {'Config':<45} {'MaxDiff':<12} {'RelDiff(%)':<12} {'Pass':<6}")
    print("-" * 90)
    for r in results:
        print(f"{r['config'].name:<15} {str(r['config']):<45} "
              f"{r['max_diff']:<12.2e} {r['rel_diff']:<12.4f} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 90)
    print(f"精度测试: {passed_count}/{len(results)} 通过")
    
    return results


# ============================================================================
# 性能测试
# ============================================================================

def run_performance_test(
    chunk_size: int = 4096,
    seq_len: int = 131072,
    num_heads: int = 128,
    head_dim: int = 576,
    d_v: int = 512,
    topk: int = 2048,
    num_warmup: int = 10,
    num_benchmark: int = 100,
    device: str = 'cuda',
) -> dict:
    """
    运行 flash_mla_sparse_fwd 的性能测试。
    
    TFLOPS 计算公式: (S * (DQK + DV) * topk * 2 * H) / time / 1e12
    
    Args:
        chunk_size: 序列长度
        seq_len: KV 序列长度
        num_heads: head 数量
        head_dim: head 维度（QK）
        d_v: value 维度
        topk: topk 数量
        num_warmup: 预热次数
        num_benchmark: 测试次数
        device: 设备
    
    Returns:
        性能测试结果
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # 生成随机输入
    q = torch.randn(chunk_size, num_heads, head_dim, device=device, dtype=torch.bfloat16) / 10
    kv = torch.randn(seq_len, 1, head_dim, device=device, dtype=torch.bfloat16) / 10
    
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    
    # 生成 indices
    indices = torch.full((chunk_size, 1, topk), seq_len, dtype=torch.int32, device=device)
    for t in range(chunk_size):
        max_valid_idx = min(max(1, t + 1), seq_len)
        num_valid = min(topk, max_valid_idx)
        i_i = torch.randperm(max_valid_idx, device=device)[:num_valid]
        indices[t, 0, :num_valid] = i_i.to(torch.int32)
    
    # 计算 sm_scale
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # 预热
    for _ in range(num_warmup):
        _ = flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v)
    torch.cuda.synchronize()
    
    # 性能测试
    start = time.time()
    for _ in range(num_benchmark):
        out, max_logits, lse, p_out = flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v)
        # _ = flash_mla_to_attn_sum(p_out, max_logits, lse, sm_scale)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / num_benchmark
    
    # 计算 TFLOPS
    # 公式: (S * (DQK + DV) * topk * 2 * H) / time / 1e12
    flops = chunk_size * (head_dim + d_v) * topk * 2 * num_heads
    tflops = flops / (elapsed_ms * 1e-3) / 1e12
    
    return {
        'chunk_size': chunk_size,
        'seq_len': seq_len,
        'num_heads': num_heads,
        'topk': topk,
        'elapsed_ms': elapsed_ms,
        'tflops': tflops,
    }


def test_performance():
    """运行性能测试"""
    print("\n" + "=" * 100)
    print("Flash MLA Sparse Forward 性能测试")
    print("=" * 100)
    
    # 测试配置
    test_cases = [
        # (chunk_size, seq_len, topk)
        (1024, 32768, 2048),
        (1024, 65536, 2048),
        (2048, 65536, 2048),
        (4096, 65536, 2048),
        (4096, 131072, 2048),
        (8192, 131072, 2048),
        (16 * 1024, 131072, 2048),
    ]
    
    print(f"\n{'ChunkSize':<12} {'SeqLen':<12} {'TopK':<10} {'Time(ms)':<12} {'TFLOPS':<12}")
    print("-" * 70)
    
    results = []
    for chunk_size, seq_len, topk in test_cases:
        try:
            result = run_performance_test(
                chunk_size=chunk_size,
                seq_len=seq_len,
                topk=topk,
                num_warmup=10,
                num_benchmark=100,
            )
            results.append(result)
            print(f"{result['chunk_size']:<12} {result['seq_len']:<12} {result['topk']:<10} "
                  f"{result['elapsed_ms']:<12.3f} {result['tflops']:<12.2f}")
        except Exception as e:
            print(f"跳过 chunk={chunk_size}, seq={seq_len}, topk={topk}: {e}")
    
    print("-" * 70)
    
    return results


# ============================================================================
# 与 TileLang 的精度对比测试
# ============================================================================

def run_accuracy_test_vs_tilelang(config: TestConfig, device: str = 'cuda') -> dict:
    """
    运行精度测试：对比 flash_mla_sparse_fwd 与 TileLang ref_indexer_loss_fwd_interface。
    
    Args:
        config: 测试配置
        device: 设备
    
    Returns:
        测试结果字典
    """
    # 尝试导入 tilelang 的参考实现
    try:
        from sparse_mla_tilelang import sparse_mla_fwd_interface, ref_sparse_mla_fwd_interface
    except ImportError as e:
        print(f"无法导入 ref_indexer_loss_fwd_interface: {e}")
        print("请确保 triton/chunk_loss 路径正确添加到 sys.path")
        return {
            'config': config,
            'max_diff': float('nan'),
            'rel_diff': float('nan'),
            'passed': False,
            'error': str(e)
        }
    
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # 生成随机输入（flash_mla 格式）
    q = torch.randn(config.chunk_size, config.num_heads, config.head_dim, 
                    device=device, dtype=torch.bfloat16) / 10
    kv = torch.randn(config.seq_len, 1, config.head_dim, 
                     device=device, dtype=torch.bfloat16) / 10
    
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    
    # 计算 chunk_offset（用于 tilelang）
    chunk_offset = config.seq_len - config.chunk_size
    kv_stride = 1
    
    # 生成 indices（flash_mla 格式: [s_q, h_kv, topk]）
    indices_flash = torch.full((config.chunk_size, 1, config.topk), 
                               config.seq_len, dtype=torch.int32, device=device)
    for t in range(config.chunk_size):
        max_valid_idx = min(max(1, ((t + chunk_offset) // kv_stride)), config.seq_len)
        num_valid = min(config.topk, max_valid_idx)
        i_i = torch.randperm(max_valid_idx, device=device)[:num_valid]
        indices_flash[t, 0, :num_valid] = i_i.to(torch.int32)
    
    # 转换为 tilelang 格式: [batch, seq_len, kv_group, topk]
    q_tl = q.unsqueeze(0)  # [1, s_q, h_q, d_qk]
    kv_tl = kv.unsqueeze(0)  # [1, s_kv, 1, d_qk]
    indices_tl = indices_flash.unsqueeze(0)  # [1, s_q, 1, topk]
    
    # 计算 sm_scale
    sm_scale = 1.0 / math.sqrt(config.head_dim)
    
    # Flash MLA 前向
    out, max_logits, lse, p_out = flash_mla_sparse_fwd(q, kv, indices_flash, sm_scale, config.d_v)
    
    # 后处理得到 attn_sum
    flash_attn_sum = flash_mla_to_attn_sum(p_out, max_logits, lse, sm_scale)
    
    # TileLang 参考实现
    tl_out, tl_lse = sparse_mla_fwd_interface(q_tl, kv_tl, indices_tl, q_start_index_s=0, kv_stride=1, sm_scale=sm_scale)
    ref_out = ref_sparse_mla_fwd_interface(q_tl, kv_tl, indices_tl, q_start_index_s=0, kv_stride=1, sm_scale=sm_scale)

    def print_diff(a, b, msg):
        abs_diff = torch.abs(a - b)
        print(f"  {msg}"
              f"    Max diff: {abs_diff.max().item(): .4f}"
              f"\tRel diff: {(abs_diff / (1e-4 + torch.abs(a))).mean().item() * 100: .4f}%")
        
    print_diff(tl_out.squeeze(0), out, "tl vs flash out")
    print_diff(ref_out.squeeze(0), out, "torch vs flash")
    print_diff(tl_lse.squeeze(0), lse, "tl vs flash lse")
    
    print(111)
    
    # max_diff, rel_diff = calc_diff(ref_attn_sum_f32, flash_attn_sum_f32.to(ref_attn_sum_f32.dtype))
    # passed = rel_diff < 1e-3  # relative error < 0.001%
    
    # tl_attn_sum = tl_attn_sum.squeeze(0)  # 移除 batch 维度
    
    # # 计算误差
    # flash_attn_sum_f32 = flash_attn_sum.float()
    # tl_attn_sum_f32 = tl_attn_sum.float()
    
    # abs_diff = torch.abs(flash_attn_sum_f32 - tl_attn_sum_f32)
    # max_diff = abs_diff.max().item()
    # rel_diff = (abs_diff / (1e-6 + torch.abs(tl_attn_sum_f32))).mean().item() * 100
    
    # # 判断是否通过
    # passed = max_diff < 1e-2 and rel_diff < 1.0
    
    # return {
    #     'config': config,
    #     'max_diff': max_diff,
    #     'rel_diff': rel_diff,
    #     'flash_max': flash_attn_sum_f32.abs().max().item(),
    #     'tl_max': tl_attn_sum_f32.abs().max().item(),
    #     'passed': passed
    # }


def test_accuracy_vs_tilelang(configs: List[TestConfig]) -> List[dict]:
    """批量运行与 TileLang 的精度对比测试"""
    print("\n" + "=" * 100)
    print("前向精度测试 (Flash MLA attn_sum vs TileLang attn_sum)")
    print("=" * 100)
    
    results = []
    for config in configs:
        try:
            result = run_accuracy_test_vs_tilelang(config)
            results.append(result)
        except Exception as e:
            print(f"跳过测试 {config.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'Name':<15} {'Config':<45} {'MaxDiff':<12} {'RelDiff(%)':<12} {'Pass':<6}")
    print("-" * 90)
    for r in results:
        print(f"{r['config'].name:<15} {str(r['config']):<45} "
              f"{r['max_diff']:<12.2e} {r['rel_diff']:<12.4f} {'✓' if r['passed'] else '✗':<6}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print("-" * 90)
    print(f"精度测试 (vs TileLang): {passed_count}/{len(results)} 通过")
    
    return results


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flash MLA Sparse Forward 精度与性能测试")
    parser.add_argument("--test_accuracy", action="store_true", help="运行精度测试 (vs PyTorch)")
    parser.add_argument("--test_accuracy_tilelang", action="store_true", help="运行精度测试 (vs TileLang)")
    parser.add_argument("--test_performance", action="store_true", help="运行性能测试")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    
    # 精度测试配置
    accuracy_configs = [
        TestConfig(name="小规模", num_heads=128, chunk_size=4096, seq_len=4096, topk=2048),
        TestConfig(name="中等规模", num_heads=128, chunk_size=8192, seq_len=4096, topk=2048),
        TestConfig(name="大规模", num_heads=128, chunk_size=8192, seq_len=8192, topk=2048),
        TestConfig(name="大topk", num_heads=128, chunk_size=8192, seq_len=16 * 1024, topk=2048),
        TestConfig(name="长序列", num_heads=128, chunk_size=16 * 1024, seq_len=128 * 1024, topk=2048),
    ]
    
    if args.all or args.test_accuracy:
        # 运行精度测试 (vs PyTorch)
        test_accuracy(accuracy_configs)
    
    if args.all or args.test_accuracy_tilelang:
        # 运行精度测试 (vs TileLang)
        test_accuracy_vs_tilelang(accuracy_configs)
    
    if args.all or args.test_performance:
        # 运行性能测试
        test_performance()
    
    if not any([args.test_accuracy, args.test_accuracy_tilelang, args.test_performance, args.all]):
        # 默认运行所有测试
        print("未指定测试类型，运行所有测试...")
        print("可用选项: --test_accuracy, --test_accuracy_tilelang, --test_performance, --all")
        test_accuracy_vs_tilelang(accuracy_configs)
        test_accuracy(accuracy_configs)
        test_performance()
