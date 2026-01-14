import torch
import time
from flash_mla.flash_mla_interface import flash_mla_sparse_fwd, calc_attn_dist

def test_performance():
    # 测试参数 (与sparse_mla_fwd.py保持一致)
    B = 1
    S = 8192
    SKV = 128 * 1024 
    H = 128
    DQK = 576
    DV = 512
    topk = 2048
    dtype = torch.bfloat16
    
    # 生成随机输入
    torch.manual_seed(0)
    q = torch.randn((S, H, DQK), dtype=dtype, device='cuda') / 10
    kv = torch.randn((SKV, 1, DQK), dtype=dtype, device='cuda') / 10
    
    # 生成indices (注意flash_mla需要[batch, seq_len, topk]格式)
    indices = torch.full((S, 1, topk), SKV, dtype=torch.int32, device='cuda')
    for t in range(S):
        i_i = torch.randperm(min(max(1, t), SKV), device='cuda')[:topk]
        indices[t, 0, :len(i_i)] = i_i
    
    # 计算sm_scale
    sm_scale = 1.0 / (DQK ** 0.5)
    
    # 预热
    for _ in range(10):
        _ = flash_mla_sparse_fwd(q, kv, indices, sm_scale, DV)
    torch.cuda.synchronize()
    
    # 性能测试
    start = time.time()
    for _ in range(100):
        out = flash_mla_sparse_fwd(q, kv, indices, sm_scale, DV)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000 / 100
    
    # 计算带宽和TFLOPS
    io_bandwidth = (B * S * DQK * topk * 2) / (elapsed_ms * 1e-3) / 1e12
    tflops = (B * S * (DQK + DV) * topk * 2 * H) / (elapsed_ms * 1e-3) / 1e12
    
    print(f"Average time: {elapsed_ms:.3f} ms")
    print(f"IO Bandwidth: {io_bandwidth:.2f} TB/s")
    print(f"TFLOPS: {tflops:.2f}")


if __name__ == "__main__":
    test_performance()
