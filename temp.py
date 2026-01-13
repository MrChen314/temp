#include "fwd.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cooperative_groups.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>

#include "utils.h"
#include "helpers.h"

namespace sm90 {

using namespace cute;

constexpr int D_Q = 576;
constexpr int D_K = 576;

constexpr int B_H = 64;
constexpr int B_TOPK = 64;    // TopK block size
constexpr int NUM_THREADS = 128*3;

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ = SmemLayoutQTiles<9>;
using SmemLayoutK = SmemLayoutKTiles<9>;

// Simplified SharedMemoryPlan for attn_dist kernel
struct AttnDistSharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    array_aligned<bf16, cosize_v<SmemLayoutK>> k[2];

    bool is_kv_valid[2][B_TOPK];
    
    // Head dimension reduction buffer
    float attn_dist_reduce[2][B_TOPK];  // [warpgroup_idx][topk_position]
    
    // Barriers
    transac_bar_t bar_q, bar_k0_free[2], bar_k0_ready[2], bar_k1_free[2], bar_k1_ready[2], bar_is_kv_valid_ready;
};

using TiledMMA_QK = decltype(make_tiled_mma(
    GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<_1, _1, _1>>{}
));

template<
    typename Shape_Q, typename TMA_Q
>
struct AttnDistTmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
};

enum NamedBarriers : uint32_t {
    warpgroup0_sync = 0,
    warpgroup1_sync = 1
};


template<typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
attn_dist_kernel(__grid_constant__ const AttnDistParams params, __grid_constant__ const TmaParams tma_params) {
#if IS_SM90
    const int q_h_idx = blockIdx.x % (params.h_q/B_H);
    const int s_q_idx = blockIdx.x / (params.h_q/B_H);
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warpgroup = threadIdx.x % 128;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    AttnDistSharedMemoryPlan &plan = *reinterpret_cast<AttnDistSharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});

    if (warp_idx == 0 && elect_one_sync()) {
        // Prefetch TMA descriptors
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());

        // Initialize barriers
        plan.bar_q.init(1);
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            plan.bar_k0_free[i].init(128);
            plan.bar_k0_ready[i].init(128);
            plan.bar_k1_free[i].init(128);
            plan.bar_k1_ready[i].init(128);
        }
        plan.bar_is_kv_valid_ready.init(16);
        
        // Initialize reduction buffers to zero
        CUTE_UNROLL
        for (int wg = 0; wg < 2; ++wg) {
            CUTE_UNROLL
            for (int i = 0; i < B_TOPK; ++i) {
                plan.attn_dist_reduce[wg][i] = 0.0f;
            }
        }
        
        fence_barrier_init();
    }

    __syncthreads();
    
    const int num_topk_blocks = params.topk / B_TOPK;
    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<216>();

        if (warp_idx == 0 && elect_one_sync()) {
            // Load Q
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<B_H>, Int<D_Q>>{}
            )(_, _, q_h_idx, _0{});
            launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
            plan.bar_q.arrive_and_expect_tx(B_H*D_Q*sizeof(bf16));
        }

        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});
        
        // Load LSE values for this thread's rows
        float lse_local[2];
        CUTE_UNROLL
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
            int real_row = get_AorC_row_idx(row_idx, idx_in_warpgroup);
            int global_head = q_h_idx * B_H + real_row;
            lse_local[row_idx] = params.lse[s_q_idx * params.h_q + global_head];
        }
        
        // Wait for Q
        plan.bar_q.wait(0);

        bool cur_bar_wait_phase = 0;
        
        struct Warpgroup0 {};
        struct Warpgroup1 {};

        auto qkt_gemm_one_tile = [&](auto warpgroup_idx, int tile_idx, bool clear_accum) {
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            TiledMMA tiled_mma_QK = TiledMMA_QK{};
            Tensor sQ_tile = flat_divide(sQ, Tile<Int<B_H>, Int<64>>{})(_, _, _0{}, tile_idx);
            Tensor sK_tile = make_tensor(make_smem_ptr(plan.k[(int)IS_WG1].data() + tile_idx*B_TOPK*64), SmemLayoutKTiles<1>{});
            gemm_ss(clear_accum, tiled_mma_QK, sQ_tile, sK_tile, rP, idx_in_warpgroup);
        };

        // Compute normalized attention probability: attn_prob = exp2(rP * sm_scale - lse)
        auto compute_normalized_attn_prob = [&](auto warpgroup_idx) {
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            const float scale = params.sm_scale_div_log2;
            
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    int col = 8*(i/4) + (idx_in_warpgroup%4)*2;
                    // Apply mask and compute normalized attention probability
                    float p0 = plan.is_kv_valid[IS_WG1][col] ? 
                               exp2f(rP(i) * scale - lse_local[row_idx]) : 0.0f;
                    float p1 = plan.is_kv_valid[IS_WG1][col+1] ? 
                               exp2f(rP(i+1) * scale - lse_local[row_idx]) : 0.0f;
                    rP(i) = p0;
                    rP(i+1) = p1;
                }
            }
        };

        // Reduce over heads and store to global memory
        auto reduce_heads_and_store = [&](int block_idx, int wg_idx) {
            // Step 1: Each thread sums its 2 rows (8 rows apart) for each column
            // rP layout: every 4 elements form a group, 0,1 belong to row0's two columns, 2,3 belong to row1's same two columns
            float local_sum[16];  // 16 column positions partial sums
            CUTE_UNROLL
            for (int col_local = 0; col_local < 8; ++col_local) {
                int i_row0 = col_local * 4;      // row 0's element start
                int i_row1 = col_local * 4 + 2;  // row 1's element start
                local_sum[col_local * 2] = rP(i_row0) + rP(i_row1);          // column 0
                local_sum[col_local * 2 + 1] = rP(i_row0 + 1) + rP(i_row1 + 1);  // column 1
            }
            // Now: each thread holds 16 columns' partial sums (2 rows merged)
            
            // Step 2: Warp-level column reduction (different rows of the same column)
            // Threads t and t^4, t^8, t^16 hold same column's different row data
            // shfl_xor(4), shfl_xor(8), shfl_xor(16) merge 8 thread groups within warp
            CUTE_UNROLL
            for (int i = 0; i < 16; ++i) {
                local_sum[i] += __shfl_xor_sync(0xffffffff, local_sum[i], 4);
                local_sum[i] += __shfl_xor_sync(0xffffffff, local_sum[i], 8);
                local_sum[i] += __shfl_xor_sync(0xffffffff, local_sum[i], 16);
            }
            // Now: all 32 threads in warp hold same 16 rows' sum (but hold different columns)
            // Note: each thread still only knows its responsible 16 columns
            
            // Step 3: Cross-warp reduction - 4 warps each hold different 16 rows
            // After shfl_xor(4,8,16), every 4 threads within warp hold same data
            // So each warp only needs first 4 threads (idx_in_warpgroup % 32 < 4) to write
            // 4 warps x 4 threads = 16 threads write, each thread writes 16 columns
            //
            // atomicAdd: 4 warps each contribute 16 rows' sum, finally accumulate to 64 rows' total sum
            if (idx_in_warpgroup % 32 < 4) {
                CUTE_UNROLL
                for (int i = 0; i < 16; ++i) {
                    int col_idx = 8 * (i / 2) + (idx_in_warpgroup % 4) * 2 + (i % 2);
                    atomicAdd(&plan.attn_dist_reduce[wg_idx][col_idx], local_sum[i]);
                }
            }
            
            // Warpgroup sync - wait for all 4 warps to complete writing
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, wg_idx ? NamedBarriers::warpgroup1_sync : NamedBarriers::warpgroup0_sync);
            
            // Step 4: Write to global memory (each thread writes one element)
            if (idx_in_warpgroup < B_TOPK) {
                int out_idx = q_h_idx * (params.s_q * params.topk) + 
                              s_q_idx * params.topk + 
                              block_idx * B_TOPK + idx_in_warpgroup;
                params.attn_sum[out_idx] = plan.attn_dist_reduce[wg_idx][idx_in_warpgroup];
                // Clear for next round
                plan.attn_dist_reduce[wg_idx][idx_in_warpgroup] = 0.0f;
            }
            
            // Sync again to ensure clear is complete before next iteration
            NamedBarrier::arrive_and_wait(128, wg_idx ? NamedBarriers::warpgroup1_sync : NamedBarriers::warpgroup0_sync);
        };


        if (warpgroup_idx == 0) {
            // Warpgroup 0 - handles even topk blocks (block_idx = 0, 2, 4, ...)

            auto pipelined_wait_and_qkt_gemm_l = [&]() __attribute__((always_inline)) {
                plan.bar_k0_ready[0].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup0{}, 0, true);
                qkt_gemm_one_tile(Warpgroup0{}, 1, false);
                qkt_gemm_one_tile(Warpgroup0{}, 2, false);
                qkt_gemm_one_tile(Warpgroup0{}, 3, false);
                warpgroup_commit_batch();
            };

            auto pipelined_wait_and_qkt_gemm_r = [&]() __attribute__((always_inline)) {
                plan.bar_k0_ready[1].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup0{}, 4, false);
                qkt_gemm_one_tile(Warpgroup0{}, 5, false);
                qkt_gemm_one_tile(Warpgroup0{}, 6, false);
                qkt_gemm_one_tile(Warpgroup0{}, 7, false);
                qkt_gemm_one_tile(Warpgroup0{}, 8, false);
                warpgroup_commit_batch();
            };
            
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                if (block_idx == 0) {
                    // First iteration: wait and compute QK GEMM
                    pipelined_wait_and_qkt_gemm_l();
                    pipelined_wait_and_qkt_gemm_r();
                    warpgroup_wait<0>();
                }
                
                // Compute normalized attention probability
                compute_normalized_attn_prob(Warpgroup0{});
                
                // Reduce over heads and store
                reduce_heads_and_store(block_idx, 0);

                // Release K buffer
                plan.bar_k0_free[0].arrive();
                plan.bar_k0_free[1].arrive();

                cur_bar_wait_phase ^= 1;

                if (block_idx+2 < num_topk_blocks) {
                    // Prefetch next QK GEMM
                    pipelined_wait_and_qkt_gemm_l();
                    pipelined_wait_and_qkt_gemm_r();
                    warpgroup_wait<0>();
                }
            }
        } else {
            // Warpgroup 1 - handles odd topk blocks (block_idx = 1, 3, 5, ...)

            auto pipelined_wait_and_qkt_gemm = [&]() __attribute__((always_inline)) {
                plan.bar_k1_ready[1].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup1{}, 4, true);
                qkt_gemm_one_tile(Warpgroup1{}, 5, false);
                qkt_gemm_one_tile(Warpgroup1{}, 6, false);
                qkt_gemm_one_tile(Warpgroup1{}, 7, false);
                qkt_gemm_one_tile(Warpgroup1{}, 8, false);
                plan.bar_k1_ready[0].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup1{}, 0, false);
                qkt_gemm_one_tile(Warpgroup1{}, 1, false);
                qkt_gemm_one_tile(Warpgroup1{}, 2, false);
                qkt_gemm_one_tile(Warpgroup1{}, 3, false);
                warpgroup_commit_batch();
            };
            
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                // Issue rP = sQ @ sK1, and wait
                pipelined_wait_and_qkt_gemm();
                warpgroup_wait<0>();

                // Compute normalized attention probability
                compute_normalized_attn_prob(Warpgroup1{});
                
                // Reduce over heads and store (odd block)
                reduce_heads_and_store(block_idx + 1, 1);

                // Release K buffer
                plan.bar_k1_free[0].arrive();
                plan.bar_k1_free[1].arrive();

                cur_bar_wait_phase ^= 1;
            }
        }
    } else {
        // Producer warpgroup
        cutlass::arch::warpgroup_reg_dealloc<72>();

        constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128/GROUP_SIZE;
        constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
        int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
        int group_idx = idx_in_warpgroup / GROUP_SIZE;
        int* gIndices = params.indices + s_q_idx*params.topk;   // [topk]

        bf16* my_sKV_base = &(make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTiles<1>{})(group_idx, idx_in_group*8));
        bf16* my_gKV_base = params.k + idx_in_group*8;
        
        int64_t token_indices[2][NUM_ROWS_PER_GROUP];
        bool is_token_valid[2][NUM_ROWS_PER_GROUP];
        auto load_token_indices = [&](int block_idx) {
            CUTE_UNROLL
            for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                    int offs = (block_idx+buf_idx)*B_TOPK + local_row*NUM_GROUPS + group_idx;
                    int t = __ldg(gIndices + offs);
                    token_indices[buf_idx][local_row] = t*(int64_t)params.stride_k_s_kv;   // We mult it with params.stride_k_s_kv here since it's faster
                    is_token_valid[buf_idx][local_row] = t >= 0 && t < params.s_kv;
                }
            }
        };
        
        int64_t cache_policy = createpolicy_evict_last();
        auto copy_tiles = [&](int block_idx, int buf_idx, int tile_start, int tile_end) {
            // Copy some K tiles from global memory to shared memory
            // A tile has a shape of 64 (B_TOPK) x 64
            // `buf_idx` is the index of the shared memory buffer, 0 or 1
            // `tile_idx` is the index of the tile to load, from 0 to D_K/64-1 = 8
            CUTE_UNROLL
            for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                int64_t token_index = token_indices[buf_idx][local_row];
                CUTE_UNROLL
                for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                    cp_async_cacheglobal_l2_prefetch_256B(
                        my_gKV_base + token_index + tile_idx*64,
                        my_sKV_base + (buf_idx*B_TOPK*D_K + tile_idx*(B_TOPK*64) + local_row*NUM_GROUPS*64),
                        is_token_valid[buf_idx][local_row],
                        cache_policy
                    );
                }
            }
        };

        auto commit_to_mbar = [&](transac_bar_t &bar) {
            cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar));
        };

        int cur_bar_wait_phase = 1;

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
            load_token_indices(block_idx);

            // K0L (left half for WG0's even block)
            plan.bar_k0_free[0].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+0, 0, 0, 4);
            commit_to_mbar(plan.bar_k0_ready[0]);

            // K1R (right half for WG1's odd block)
            plan.bar_k1_free[1].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+1, 1, 4, 9);
            commit_to_mbar(plan.bar_k1_ready[1]);
            
            // K0R (right half for WG0's even block)
            plan.bar_k0_free[1].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+0, 0, 4, 9);
            commit_to_mbar(plan.bar_k0_ready[1]);

            // K1L (left half for WG1's odd block)
            plan.bar_k1_free[0].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+1, 1, 0, 4);
            commit_to_mbar(plan.bar_k1_ready[0]);

            // Valid mask
            if (idx_in_group == 0) {
                CUTE_UNROLL
                for (int buf_idx = 0; buf_idx < 2; ++buf_idx)
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row)
                        plan.is_kv_valid[buf_idx][local_row*NUM_GROUPS+group_idx] = is_token_valid[buf_idx][local_row];
                plan.bar_is_kv_valid_ready.arrive();
            }

            cur_bar_wait_phase ^= 1;
        }
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
}


void run_attn_dist_kernel(const AttnDistParams& params) {
    FLASH_ASSERT(params.h_kv == 1);
    FLASH_ASSERT(params.topk % (2*B_TOPK) == 0);   // To save some boundary checkings
    FLASH_ASSERT(params.topk > 0);
    FLASH_ASSERT(params.h_q % B_H == 0);

    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    AttnDistTmaParams<
        decltype(shape_Q), decltype(tma_Q)
    > tma_params = {
        shape_Q, tma_Q
    };
    auto kernel = &attn_dist_kernel<decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(AttnDistSharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3((params.h_q/B_H)*params.s_q, 1, 1),    // NOTE We put s_q on the first dim since it can be larger than 65536 (the maximum size of griddim.y and griddim.z)
        dim3(NUM_THREADS, 1, 1),
        dim3(1, 1, 1),
        smem_size,
        params.stream
    }; 
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    );
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
