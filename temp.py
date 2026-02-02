@triton.jit
def triton_attn_dist_kernel(
    p_out_ptr,      
    output_ptr,     
    sm_scale,
    s_q, topk,
    stride_p_s: tl.int64, stride_p_h: tl.int64, stride_p_k: tl.int64,  # 改为 int64
    stride_o_s: tl.int64, stride_o_k: tl.int64,                        # 改为 int64
    H_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ... 其余代码保持不变
