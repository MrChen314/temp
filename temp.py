        auto store_rP_to_global = [&](auto warpgroup_idx, int block_idx) {
            if (params.p_out == nullptr) return;
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            // WG0 处理偶数 block，WG1 处理奇数 block
            int cur_block_idx = block_idx + (IS_WG1 ? 1 : 0);
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {
                // 通过 WGMMA 布局映射获取实际的行索引
                int real_row = get_AorC_row_idx(row_idx, idx_in_warpgroup);
                int h_idx = q_h_idx * B_H + real_row;
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    // 通过 WGMMA 布局映射获取实际的列索引
                    int real_col = get_AorC_col_idx(i, idx_in_warpgroup);
                    int topk_idx = cur_block_idx * B_TOPK + real_col;
                    // 计算输出地址：[s_q, h_q, topk] 布局
                    float* p_out_ptr = params.p_out + s_q_idx * params.h_q * params.topk 
                                       + h_idx * params.topk + topk_idx;
                    // 使用 float2 进行向量化写入，减少内存事务
                    float2 p_data = make_float2(rP(i), rP(i+1));
                    *reinterpret_cast<float2*>(p_out_ptr) = p_data;
                }
            }
        };
