#!/usr/bin/env python3
"""
Test script for utcmma_ss (SM100 2x1SM SS MMA)
Tests the CUDA kernel against PyTorch reference implementation.
"""

import torch
import test_2sm_mma_cuda  # 编译后的模块


def test_utcmma_ss():
    """Test utcmma_ss precision against PyTorch matmul"""
    print("=" * 60)
    print("Testing utcmma_ss (SM100 2x1SM SS MMA)")
    print("=" * 60)
    
    # 矩阵规格
    M, N, K = 128, 128, 256
    print(f"Q shape: [{M}, {K}]")
    print(f"K shape: [{N}, {K}]")
    print(f"P shape: [{M}, {N}] = Q @ K.T")
    print()
    
    # 生成输入数据
    torch.manual_seed(42)
    Q = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    K_mat = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    
    print(f"Q dtype: {Q.dtype}, device: {Q.device}")
    print(f"K dtype: {K_mat.dtype}, device: {K_mat.device}")
    print()
    
    # CUDA kernel 计算
    print("Running CUDA kernel...")
    P_cuda = test_2sm_mma_cuda.utcmma_ss(Q, K_mat)
    torch.cuda.synchronize()
    print(f"P_cuda shape: {P_cuda.shape}, dtype: {P_cuda.dtype}")
    
    # PyTorch 参考计算
    print("Running PyTorch reference...")
    P_ref = torch.matmul(Q.float(), K_mat.float().T)
    print(f"P_ref shape: {P_ref.shape}, dtype: {P_ref.dtype}")
    print()
    
    # 精度对比
    diff = (P_cuda - P_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 相对误差
    rel_diff = diff / (P_ref.abs() + 1e-6)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print("=" * 60)
    print("Precision Results:")
    print("=" * 60)
    print(f"Max absolute diff:  {max_diff:.6e}")
    print(f"Mean absolute diff: {mean_diff:.6e}")
    print(f"Max relative diff:  {max_rel_diff:.6e}")
    print(f"Mean relative diff: {mean_rel_diff:.6e}")
    print()
    
    # 打印部分结果用于调试
    print("Sample values (first 5x5 block):")
    print("CUDA result:")
    print(P_cuda[:5, :5])
    print("Reference result:")
    print(P_ref[:5, :5])
    print()
    
    # 断言精度在合理范围内
    # bf16 输入、float 累加，允许的误差范围
    threshold = 1e-2
    if max_diff < threshold:
        print(f"✓ Test PASSED! (max_diff={max_diff:.6e} < {threshold})")
        return True
    else:
        print(f"✗ Test FAILED! (max_diff={max_diff:.6e} >= {threshold})")
        return False


def test_different_inputs():
    """Test with different input patterns"""
    print("\n" + "=" * 60)
    print("Testing with different input patterns")
    print("=" * 60)
    
    M, N, K = 128, 128, 256
    test_cases = [
        ("All ones", torch.ones, torch.ones),
        ("All zeros", torch.zeros, torch.zeros),
        ("Identity-like", lambda *args, **kwargs: torch.eye(M, K, **kwargs), 
                          lambda *args, **kwargs: torch.eye(N, K, **kwargs)),
        ("Random uniform", lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1,
                          lambda *args, **kwargs: torch.rand(*args, **kwargs) * 2 - 1),
    ]
    
    all_passed = True
    for name, q_gen, k_gen in test_cases:
        try:
            Q = q_gen(M, K, dtype=torch.bfloat16, device='cuda')
            K_mat = k_gen(N, K, dtype=torch.bfloat16, device='cuda')
            
            P_cuda = test_2sm_mma_cuda.utcmma_ss(Q, K_mat)
            P_ref = torch.matmul(Q.float(), K_mat.float().T)
            
            max_diff = (P_cuda - P_ref).abs().max().item()
            status = "✓" if max_diff < 1e-2 else "✗"
            print(f"{status} {name}: max_diff = {max_diff:.6e}")
            
            if max_diff >= 1e-2:
                all_passed = False
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("CUDA Device:", torch.cuda.get_device_name())
    print("CUDA Capability:", torch.cuda.get_device_capability())
    print()
    
    passed = test_utcmma_ss()
    
    if passed:
        passed = test_different_inputs()
    
    print("\n" + "=" * 60)
    if passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
    
    exit(0 if passed else 1)
