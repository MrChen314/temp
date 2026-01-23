            // Output P matrix (after mask) to global memory
            if (params.p_out != nullptr) {
                int h_idx = cta_idx*(B_H/2) + (idx_in_warpgroup % 64);
                int k_base = k*B_TOPK + (idx_in_warpgroup < 64 ? 0 : 64);
                float* p_out_ptr = params.p_out + (int64_t)s_q_idx * params.h_q * params.topk 
                                   + h_idx * params.topk + k_base;
                CUTE_UNROLL
                for (int i = 0; i < (B_TOPK/2); i += 4) {
                    cutlass::Array<float, 4> p_vec;
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        p_vec[j] = p_float[i + j];
                    }
                    // 128-bit store (4 x fp32 = 128 bits)
                    *reinterpret_cast<cutlass::Array<float, 4>*>(p_out_ptr + i) = p_vec;
                }
            }
