import torch
import triton
import triton.language as tl
import math


@triton.jit
def triton_attn_dist_kernel(
    p_out_ptr,      # [s_q, h_q, topk]
    output_ptr,     # [s_q, topk]
    sm_scale,
    s_q, topk,
    stride_p_s, stride_p_h, stride_p_k,
    stride_o_s, stride_o_k,
    H_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    s_idx = tl.program_id(0)
    k_offs = tl.arange(0, BLOCK_K)
    # Create mask to ensure we only access valid k indices
    k_mask = k_offs < topk
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    
    for h_idx in range(H_Q):
        p_ptrs = p_out_ptr + s_idx * stride_p_s + h_idx * stride_p_h + k_offs * stride_p_k
        
        # Load with mask to prevent out-of-bounds access
        p = tl.load(p_ptrs, mask=k_mask, other=0.0)
        attn_score = p * sm_scale
        
        # Compute softmax only over valid elements
        # First, mask out invalid elements before computing max
        attn_score_masked = tl.where(k_mask, attn_score, float('-inf'))
        max_val = tl.max(attn_score_masked, axis=0)
        exp_val = tl.exp(attn_score_masked - max_val)
        # Mask out invalid elements in sum
        exp_val_masked = tl.where(k_mask, exp_val, 0.0)
        sum_exp = tl.sum(exp_val_masked, axis=0)
        attn_prob = tl.where(k_mask, exp_val / sum_exp, 0.0)  # [BLOCK_K]
        
        acc += attn_prob
    
    # Normalize: divide by sum to match reference implementation
    # Reference: attn_sum / attn_sum.sum(-1, keepdim=True)
    acc_masked = tl.where(k_mask, acc, 0.0)
    sum_acc = tl.sum(acc_masked, axis=0)
    output_val = tl.where(k_mask, acc / sum_acc, 0.0)
    
    o_ptrs = output_ptr + s_idx * stride_o_s + k_offs * stride_o_k
    # Store with mask to prevent out-of-bounds write
    tl.store(o_ptrs, output_val, mask=k_mask)


def triton_attn_dist(p_out: torch.Tensor, sm_scale) -> torch.Tensor:
    s_q, h_q, topk = p_out.shape
    output = torch.empty((s_q, topk), device=p_out.device, dtype=p_out.dtype)

    assert topk == triton.next_power_of_2(topk)
    
    grid = (s_q,)
    
    triton_attn_dist_kernel[grid](
        p_out, output, sm_scale,
        s_q, topk,
        p_out.stride(0), p_out.stride(1), p_out.stride(2),
        output.stride(0), output.stride(1),
        H_Q=h_q,
        BLOCK_K=topk,
    )
    return output


def ref_torch_attn_sum(
    p_out: torch.Tensor,      # [s_q, h_q, topk]
    sm_scale
) -> torch.Tensor:
    attn_score = p_out * sm_scale
    attn_prob = torch.softmax(attn_score, dim=-1)  # [s_q, h_q, topk]
    attn_sum = attn_prob.sum(dim=1)  # [s_q, topk]
    attn_sum = attn_sum / attn_sum.sum(-1, keepdim=True)
    
    return attn_sum


def benchmark():
    s_q = 16 * 1024  # 16384
    h_q = 128
    topk = 2048
    sm_scale = 1.0 / math.sqrt(64)
    
    print(f"Benchmark Configuration:")
    print(f"  s_q = {s_q}, h_q = {h_q}, topk = {topk}")
    print(f"  Input shape: [{s_q}, {h_q}, {topk}]")
    print(f"  sm_scale = {sm_scale:.6f}")
    print()
    
    p_out = torch.randn(s_q, h_q, topk, device='cuda', dtype=torch.float32)
    
    print("Correctness Check:")
    ref_out = ref_torch_attn_sum(p_out, sm_scale)
    triton_out = triton_attn_dist(p_out, sm_scale)

    def calc_diff(a, b):
        abs_diff = torch.abs(a - b)
        max_diff = abs_diff.max().item()
        rel_diff = (abs_diff / (1e-4 + torch.abs(a))).mean().item()
        return max_diff, rel_diff

    max_diff, rel_diff = calc_diff(ref_out, triton_out.to(ref_out.dtype))
    print(f"{max_diff=}\n{rel_diff=}\n")
    is_close = rel_diff < 0.01
    
    print("Performance Benchmark (using triton.testing.do_bench):")
    
    torch_ms = triton.testing.do_bench(
        lambda: ref_torch_attn_sum(p_out, sm_scale),
        warmup=100,
        rep=1000,
    )
    print(f"  PyTorch:  {torch_ms:.4f} ms")
    
    triton_ms = triton.testing.do_bench(
        lambda: triton_attn_dist(p_out, sm_scale),
        warmup=100,
        rep=1000,
    )
    print(f"  Triton:   {triton_ms:.4f} ms")
    
    speedup = torch_ms / triton_ms
    print(f"  Speedup:  {speedup:.2f}x")
    print()
    
    return is_close


if __name__ == "__main__":
    benchmark()
