#include "mla_bwd.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <math_constants.h>

using namespace test_operator::mla_bwd;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

// Helper function: float2 subtraction using float2_add and float2_neg
CUTE_DEVICE
float2 float2_sub(const float2 &a, const float2 &b) {
    return ku::float2_add(a, ku::float2_neg(b));
}

CUTE_DEVICE
void atomic_add_float4(float* dst_ptr, const float4& v) {
    asm volatile(
        "red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(dst_ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        : "memory"
    );
}

// 手动展开的32个float原子加法 (CHUNK_SIZE = 32)
// 展开8次 float4 原子加法，明确dst和src地址计算
CUTE_DEVICE
void atomic_add_32floats_unrolled(float* dst, const float* src) {
    // i = 0:  dst[0..3]   += src[0..3]
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 0), "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]) : "memory");
    // i = 4:  dst[4..7]   += src[4..7]
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 4), "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]) : "memory");
    // i = 8:  dst[8..11]  += src[8..11]
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 8), "f"(src[8]), "f"(src[9]), "f"(src[10]), "f"(src[11]) : "memory");
    // i = 12: dst[12..15] += src[12..15]
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 12), "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]) : "memory");
    // i = 16: dst[16..19] += src[16..19]
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 16), "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]) : "memory");
    // i = 20: dst[20..23] += src[20..23]
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 20), "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]) : "memory");
    // i = 24: dst[24..27] += src[24..27]
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 24), "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]) : "memory");
    // i = 28: dst[28..31] += src[28..31]
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(dst + 28), "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]) : "memory");
}

// bf16x8 structure for vectorized operations
struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};

struct int32x8_t {
    int a0, a1, a2, a3, a4, a5, a6, a7;
};

CUTE_DEVICE int32x8_t ldg_256_indices(void* src_ptr) {
    int32x8_t val;
    asm volatile("ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(val.a0), "=r"(val.a1), "=r"(val.a2), "=r"(val.a3),
          "=r"(val.a4), "=r"(val.a5), "=r"(val.a6), "=r"(val.a7)
        : "l"(src_ptr)
    );
    return val;
}

// Preprocess delta kernel: compute delta = sum(O * dO, dim=-1)
__global__ void preprocess_delta_kernel(
    const bf16* __restrict__ O,      // [s_q, B_H, D_V]
    const bf16* __restrict__ dO,     // [s_q, B_H, D_V]
    float* __restrict__ delta,       // [s_q, B_H]
    int s_q
) {
    constexpr int D_V = 512;
    constexpr int B_H = 128;
    
    const int s_q_idx = blockIdx.x;
    const int head_idx = threadIdx.x;  // Each thread processes one head
    
    if (s_q_idx >= s_q || head_idx >= B_H) return;
    
    // Compute delta: delta[i] = sum_j(O[i,j] * dO[i,j])
    float delta_val = 0.0f;
    const int64_t row_base = ((int64_t)s_q_idx * B_H + head_idx) * D_V;
    
    // Accumulate O * dO using vectorized loads
    CUTE_UNROLL
    for (int col = 0; col < D_V; col += 8) {
        // Vectorized load of O (8 bf16 values = 128 bits)
        uint4 o_raw = __ldg((const uint4*)(O + row_base + col));
        bf16x8 o_vec;
        *(uint4*)&o_vec = o_raw;
        
        // Vectorized load of dO (8 bf16 values = 128 bits)
        uint4 do_raw = __ldg((const uint4*)(dO + row_base + col));
        bf16x8 do_vec;
        *(uint4*)&do_vec = do_raw;
        
        // Accumulate dot product
        delta_val += __bfloat162float(o_vec.a01.x) * __bfloat162float(do_vec.a01.x);
        delta_val += __bfloat162float(o_vec.a01.y) * __bfloat162float(do_vec.a01.y);
        delta_val += __bfloat162float(o_vec.a23.x) * __bfloat162float(do_vec.a23.x);
        delta_val += __bfloat162float(o_vec.a23.y) * __bfloat162float(do_vec.a23.y);
        delta_val += __bfloat162float(o_vec.a45.x) * __bfloat162float(do_vec.a45.x);
        delta_val += __bfloat162float(o_vec.a45.y) * __bfloat162float(do_vec.a45.y);
        delta_val += __bfloat162float(o_vec.a67.x) * __bfloat162float(do_vec.a67.x);
        delta_val += __bfloat162float(o_vec.a67.y) * __bfloat162float(do_vec.a67.y);
    }
    
    // Write delta to global memory
    delta[s_q_idx * B_H + head_idx] = delta_val;
}

// Kernel implementation: test mla_bwd with Q, KV, dO inputs
template<typename TmaParamsType>
__global__ __launch_bounds__(NUM_THREADS, 1) void test_mla_bwd_kernel(
    const bf16* __restrict__ q,      // [s_q, B_H, D_Q]
    const bf16* __restrict__ kv,     // [s_kv, D_K]
    const bf16* __restrict__ dO,     // [s_q, B_H, D_V]
    const float* __restrict__ lse,   // [s_q, B_H] (log-sum-exp for softmax)
    const bf16* __restrict__ O,     // [s_q, B_H, D_V] (forward output O)
    const int32_t* __restrict__ gIndices,  // [s_q, topk_length] (indices for sparse attention)
    int s_kv,                        // KV sequence length
    int topk_length,                 // TopK length
    int s_q,                         // Query sequence length
    const float* __restrict__ delta,  // [s_q, B_H] (delta = sum(O * dO))
    float* __restrict__ dKV,          // [s_kv, D_K] (dKV gradient, float32 for atomic add)
    bf16* __restrict__ dQ,            // [s_q, B_H, D_Q] (dQ gradient, bf16)
    __grid_constant__ const TmaParamsType tma_params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
    // Use cute namespace inside kernel to avoid conflicts with PyTorch's at::Layout
    using namespace cute;
    
    extern __shared__ char smem_raw[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(smem_raw);
    
    const int cta_idx = blockIdx.x % 2;  // 0 or 1
    const int s_q_idx = blockIdx.x / 2;
    const int max_kv_i = s_q_idx;
    const int tid = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();  // Global warp index
    const int lane_idx = threadIdx.x % 32;
    if (s_q_idx >= s_q) {
        return;
    }
    const int32_t* gIndices_s = gIndices + (int64_t)s_q_idx * topk_length;
    const float* lse_s = lse + (int64_t)s_q_idx * B_H;
    const float* delta_s = delta + (int64_t)s_q_idx * B_H;
    
    // Determine warpgroup: 4 warpgroups, 128 threads each
    // WG0: threads 0-127 (warpgroup_idx = 0, warp_idx = 0-3)
    // WG1: threads 128-255 (warpgroup_idx = 1, warp_idx = 4-7)
    // WG2: threads 256-383 (warpgroup_idx = 2, warp_idx = 8-11)
    // WG3: threads 384-511 (warpgroup_idx = 3, warp_idx = 12-15)
    const int warpgroup_idx = __shfl_sync(0xffffffff, tid / 128, 0);
    const int idx_in_warpgroup = tid % 128;
    const bool is_wg0 = (warpgroup_idx == 0);
    const bool is_wg1 = (warpgroup_idx == 1);
    const bool is_wg2 = (warpgroup_idx == 2);
    const bool is_wg3 = (warpgroup_idx == 3);
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);
    (void)O;

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_dQ.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
    }

    // Initialize barriers (warp 0 in CTA0)
    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_prologue_q_nope.init(1);
        plan.bar_prologue_q_rope.init(1);
        plan.bar_prologue_kv.init(1);
        plan.bar_prologue_dO.init(1);
        plan.bar_p_ready.init(1);        // WG3通知WG0 p已准备好 (2CTA sync)
        plan.bar_dp_ready.init(1);       // WG3通知WG0 dp已准备好 (2CTA sync)
        plan.bar_s_ready.init(128);        // WG0通知WG3 s已准备好
        plan.bar_ds_ready.init(128);       // WG0通知WG3 ds已准备好
        // WG3-WG2 barriers for dKV computation
        plan.bar_dkv_part0_ready.init(1);
        plan.bar_dkv_part1_ready.init(1);
        plan.bar_dkv_part2_ready.init(1);
        plan.bar_dkv_part0_done.init(128);
        plan.bar_dkv_part1_done.init(128);
        plan.bar_dkv_part2_done.init(128);
        // WG1-WG3 barriers for kv_peer cp_async
        plan.bar_kv_peer_cp_async.init(1);      // cp_async transaction barrier
        plan.bar_kv_peer_ready.init(1);         // WG1 notifies WG3 kv_peer is ready
        // WG3-WG0 barrier for dQ computation
        plan.bar_k_valid_ready.init(8);
        plan.bar_k_valid_free.init(128);
        plan.bar_dq_ready.init(1);              // WG3 notifies WG0 dQ is ready
        fence_barrier_init();
    }
    
    // Cluster sync before accessing peer SMEM - all CTAs must participate
    __syncthreads();
    cluster_sync();
    
    // Construct SMEM Tensors
    // Q and K are split into NoPE and RoPE parts
    // Q NoPE: [B_H/2, D_V] = [64, 512], Q RoPE: [B_H/2, D_ROPE] = [64, 64]
    // K NoPE: [B_TOPK/2, D_V] = [32, 512], K RoPE: [B_TOPK/2, D_ROPE] = [32, 64]
    Tensor sQNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQNoPE{});
    Tensor sQRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPE{});
    Tensor sKNoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKNoPE{});
    Tensor sKRoPE = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPE{});
    Tensor sdO = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutdO{});
    // kv_peer: [B_TOPK/2, D_K] = [32, 576] for cross-CTA K loading
    Tensor sKV_peer = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data()), SmemLayoutKV{});

    // Launch prologue TMA loads and allocate TMEM (warp 0 in each CTA)
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Q_NoPE: [B_H, D_V] split by CTA on first dim
            Tensor gQNoPE = flat_divide(
                tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_nope, gQNoPE, sQNoPE, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

            // Q_RoPE: [B_H, D_ROPE] split by CTA on first dim
            Tensor gQRoPE = flat_divide(
                tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_Q_rope, gQRoPE, sQRoPE, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);

            // dO: [B_H, D_V] split by CTA on first dim
            Tensor gdO = flat_divide(
                tma_params.tma_dO.get_tma_tensor(tma_params.shape_dO)(_, _, s_q_idx),
                Tile<Int<B_H/2>>{}
            )(_, cta_idx, _);
            ku::launch_tma_copy(tma_params.tma_dO, gdO, sdO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);

            // arrive_and_expect_tx is issued in MMA warp before actual use
        }

        TMEM::Allocator2Sm().allocate(512, plan.tmem_start_addr.data());
        TMEM::Allocator2Sm().release_allocation_lock();
    }
    __syncthreads();  // Wait for TMEM allocation

    // ========================================
    // Warpgroup 0: Softmax and dS Computation (WG0)
    // Responsibility: Compute softmax(P), load delta, compute ds
    // ========================================
    if (is_wg0) {        
        // Load LSE from global memory (needed for softmax computation)
        float row_lse = 0.0f;
        const int global_row_idx = cta_idx * (B_H/2) + idx_in_warpgroup % (B_H/2);
        row_lse = __ldg(lse_s + global_row_idx);

        const float sm_scale = 1.0f / sqrtf(576.0f);  // softmax scale
        const float scale = sm_scale * 1.44269504f;  // sm_scale * log2(e) for exp2
        Tensor sS = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
        Tensor sDS = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutS{});
        
        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;

            // Step 1: Wait for WG3 to compute P for current block
            plan.bar_p_ready.wait(phase);
            ku::tcgen05_after_thread_sync();
            
            // Step 2: Load P from TMEM (current tile) for softmax
            float2 p[(B_TOPK/2)/2];
            uint32_t tmem_base = plan.tmem_start_addr.data()[0];
            uint32_t tmem_lane = idx_in_warpgroup % (B_H/2);  // 0-63
            uint32_t tmem_col = tmem_cols::P;
            uint32_t tmem_addr = tmem_base + (tmem_lane << 16) + tmem_col;
            
            ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_addr, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            plan.bar_k_valid_ready.wait(phase);
            uint32_t is_k_valid_lo = *(uint32_t*)(plan.is_k_valid + (idx_in_warpgroup>=64?B_TOPK/8/2:0));
            float* p_float = (float*)p;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; i += 1) {
                if (!(is_k_valid_lo >> i & 1))
                    p_float[i] = -CUDART_INF_F;
            }
            plan.bar_k_valid_free.arrive();

            // Step 3: Compute softmax(P) = exp2(P*scale - LSE)
            // Compute softmax values: s = exp2(P*scale - LSE)
            float2 s_fp32[(B_TOPK/2)/2];
            float2 neg_lse = make_float2(-row_lse, -row_lse);
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                float2 scaled_p = ku::float2_fma(p[i], make_float2(scale, scale), neg_lse);
                scaled_p.x = exp2f(scaled_p.x);
                scaled_p.y = exp2f(scaled_p.y);
                // s_fp32[i] = __float22bfloat162_rn(scaled_p);
                s_fp32[i] = scaled_p;
            }
            
            // Step 4: Store s to SMEM (convert fp32 to bf16)
            // uint128_t* sS_base = (uint128_t*)plan.s_ds.s.data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*8);
            // CUTE_UNROLL
            // for (int i = 0; i < B_TOPK/2/8; ++i) {
            //     sS_base[64*i] = *(uint128_t*)(s_fp32 + i*4);
            // }
            // fence_view_async_shared();

            int s_col_offset = (idx_in_warpgroup / 64) * 32;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                sS(i*2+s_col_offset, idx_in_warpgroup%64) = bf16(s_fp32[i].x);
                sS(i*2+1+s_col_offset, idx_in_warpgroup%64) = bf16(s_fp32[i].y);
            }
            fence_view_async_shared();

            if (is_wg0) {
                __threadfence_block();
            }
            
            // Step 5: Notify WG3 that s is ready.
            if (cta_idx == 0) {
                plan.bar_s_ready.arrive(0u);
            } else {
                plan.bar_s_ready.arrive(1u);
            }

            // Step 6: Load delta from global memory
            float delta_val = 0.0f;
            delta_val = __ldg(delta_s + global_row_idx);
            
            // Step 7: Wait for WG3 to compute dP for current block
            plan.bar_dp_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            // Debug output dP is removed; consume dP directly from TMEM below.
            // Step 8: Load dp from TMEM
            float2 dp[(B_TOPK/2)/2];
            uint32_t dp_tmem_addr = tmem_base + (tmem_lane << 16) + tmem_cols::dP;
            ku::tmem_ld_32dp32bNx<B_TOPK/2>(dp_tmem_addr, dp);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            
            // Step 9: Compute ds = s * (dp - delta) * sm_scale
            // Note: Use fp32 format of s (s_fp32), not bf16 format stored in SMEM
            // Note: Use sm_scale (not scale which includes log2(e)) for ds computation
            __nv_bfloat162 ds_fp32[(B_TOPK/2)/2];
            float2 delta_float2 = make_float2(delta_val, delta_val);
            float2 sm_scale_float2 = make_float2(sm_scale, sm_scale);
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                // Convert s_fp32 back to float2 for computation
                // float2 s_val = __bfloat1622float2(s_fp32[i]);
                float2 s_val = s_fp32[i];
                float2 dp_val = dp[i];
                float2 dp_minus_delta = float2_sub(dp_val, delta_float2);
                float2 ds_val = ku::float2_mul(ku::float2_mul(s_val, dp_minus_delta), sm_scale_float2);
                ds_fp32[i] = __float22bfloat162_rn(ds_val);
            }
            
            // Step 10: Store ds to SMEM (convert fp32 to bf16)
            // uint128_t* sDS_base = (uint128_t*)plan.s_ds.ds.data() + idx_in_warpgroup%64 + 64*((idx_in_warpgroup/64)*8);
            // CUTE_UNROLL
            // for (int i = 0; i < B_TOPK/2/8; ++i) {
            //     sDS_base[64*i] = *(uint128_t*)(ds_fp32 + i*4);
            // }
            // fence_view_async_shared();
            int ds_col_offset = (idx_in_warpgroup / 64) * 32;
            CUTE_UNROLL
            for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                sDS(i*2+ds_col_offset, idx_in_warpgroup%64) = bf16(ds_fp32[i].x);
                sDS(i*2+1+ds_col_offset, idx_in_warpgroup%64) = bf16(ds_fp32[i].y);
            }
            fence_view_async_shared();
        
            if (is_wg0) {
                __threadfence_block();
            }
            
            // Step 11: Notify WG3 that ds is ready.
            if (cta_idx == 0) {
                plan.bar_ds_ready.arrive(0u);
            } else {
                plan.bar_ds_ready.arrive(1u);
            }
        }
        
        // ========================================
        // WG0 Step 12-13: Wait for dQ and transfer to global memory
        // ========================================
        // Step 12: Wait for WG3 to complete all dQ accumulations
        const int final_phase = (num_k_blocks - 1) & 1;
        plan.bar_dq_ready.wait(final_phase);
        ku::tcgen05_after_thread_sync();
        
        // Step 13: Read dQ (fp32) from TMEM, convert to bf16 in SMEM, then TMA-store to global memory
        // dQ shape: [B_H, D_Q] = [128, 576], each CTA handles [B_H/2, D_Q] = [64, 576]
        {
            constexpr int dQ_ROWS = B_H / 2;  // 64
            constexpr int NOPE_FLOATS_PER_HALF = 256 / 2;            // 128 floats/thread per NoPE tile-half
            constexpr int NOPE_CHUNKS = 8;
            constexpr int NOPE_CHUNK_FLOATS = NOPE_FLOATS_PER_HALF / NOPE_CHUNKS;  // 16 floats/chunk
            constexpr int NOPE_CHUNK_FLOAT2 = NOPE_CHUNK_FLOATS / 2;                 // 8 float2/chunk
            constexpr int ROPE_FLOAT2_PER_ROW = D_ROPE / 2 / 2;                      // 16 float2/row

            Tensor sdQ = make_tensor(make_smem_ptr(plan.u.dq.data()), SmemLayoutQ{});

            int row_in_cta = idx_in_warpgroup % dQ_ROWS;   // 0-63
            int col_half = idx_in_warpgroup / dQ_ROWS;     // 0 or 1

            uint32_t tmem_base_dq = plan.tmem_start_addr.data()[0];
            uint32_t tmem_addr_dq0 = tmem_base_dq + (row_in_cta << 16) + tmem_cols::dQ;
            uint32_t tmem_addr_dq1 = tmem_base_dq + (row_in_cta << 16) + (tmem_cols::dQ + 128);

            // dQ_NoPE part0: cols [0, 255]
            CUTE_UNROLL
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq0 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            // dQ_NoPE part1: cols [256, 511]
            CUTE_UNROLL
            for (int chunk = 0; chunk < NOPE_CHUNKS; ++chunk) {
                const int chunk_col_base = chunk * NOPE_CHUNK_FLOATS;
                float2 dq_chunk[NOPE_CHUNK_FLOAT2];
                ku::tmem_ld_32dp32bNx<NOPE_CHUNK_FLOATS>(tmem_addr_dq1 + chunk_col_base, dq_chunk);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                CUTE_UNROLL
                for (int i = 0; i < NOPE_CHUNK_FLOAT2; ++i) {
                    int col = 256 + col_half * NOPE_FLOATS_PER_HALF + chunk_col_base + i * 2;
                    sdQ(row_in_cta, col) = bf16(dq_chunk[i].x);
                    sdQ(row_in_cta, col + 1) = bf16(dq_chunk[i].y);
                }
            }

            // dQ_RoPE: cols [512, 575]
            float2 dq_rope[ROPE_FLOAT2_PER_ROW];
            uint32_t tmem_addr_dq_rope = tmem_base_dq + (row_in_cta << 16) + tmem_cols::dQ_RoPE;
            ku::tmem_ld_32dp32bNx<D_ROPE/2>(tmem_addr_dq_rope, dq_rope);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            CUTE_UNROLL
            for (int i = 0; i < ROPE_FLOAT2_PER_ROW; ++i) {
                int col = D_V + col_half * (D_ROPE / 2) + i * 2;
                sdQ(row_in_cta, col) = bf16(dq_rope[i].x);
                sdQ(row_in_cta, col + 1) = bf16(dq_rope[i].y);
            }

            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, 0);

            if (warp_idx == 0 && elect_one_sync()) {
                Tensor gdQ = flat_divide(
                    tma_params.tma_dQ.get_tma_tensor(tma_params.shape_dQ)(_, _, s_q_idx),
                    Tile<Int<B_H/2>>{}
                )(_, cta_idx, _);
                auto thr_tma_dq = tma_params.tma_dQ.get_slice(_0{});
                cute::copy(
                    tma_params.tma_dQ,
                    thr_tma_dq.partition_S(sdQ),
                    thr_tma_dq.partition_D(gdQ)
                );
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
        }
    }
    
    // ========================================
    // Warpgroup 1: KV Loading (WG1)
    // Responsibility: Maintain per-block KV tile and kv_peer transfer
    // ========================================
    if (is_wg1) {
        const int local_warp_idx = warp_idx - 4;  // WG1 has global warps [4, 7]
        constexpr int NUM_WARPS = 4;
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK / 2) / 4 / NUM_WARPS;
        constexpr int KV_PEER_ELEMENTS = (B_TOPK / 2) * D_K;  // 32 * 576 = 18432

        // Use lane0 of each warp (4 warps in WG1) to issue gather4 TMA loads.
        if (elect_one_sync()) {
            bf16* sKV_base = plan.u.q_kv.k_nope.data() + local_warp_idx * 4 * 64;

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int phase = k_block & 1;

                if (k_block > 0) {
                    // Keep producer/consumer order between iterations for all producer warps.
                    plan.bar_dq_ready.wait((k_block - 1) & 1);
                }

                int4 indices4[NUM_LOCAL_ROWS_PER_WARP];
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices4[local_row] = __ldg(
                        (const int4*)(gIndices_s + k_block * B_TOPK + cta_idx * (B_TOPK / 2)) +
                        local_row * NUM_WARPS + local_warp_idx
                    );
                }

                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    CUTE_UNROLL
                    for (int local_col = 0; local_col < D_K / 64; ++local_col) {
                        ku::tma_gather4_cta_group_2<true>(
                            &(tma_params.tensor_map_kv),
                            plan.bar_prologue_kv,
                            sKV_base + local_row * (4 * NUM_WARPS) * 64 + local_col * ((B_TOPK / 2) * 64),
                            local_col * 64,
                            indices4[local_row],
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                    }
                }
                // Load kv_peer from peer CTA using cp_async every iteration.
                if (local_warp_idx == 0) {
                    // Wait until WG3 has consumed this K block for P, then copy to kv_peer.
                    plan.bar_p_ready.wait(phase);

                    plan.bar_kv_peer_cp_async.arrive_and_expect_tx(sizeof(bf16) * KV_PEER_ELEMENTS);
                    bf16* peer_kv_peer_ptr = kerutils::get_peer_addr(plan.u.q_kv.kv_peer.data());
                    transac_bar_t* peer_bar_ptr = kerutils::get_peer_addr(&plan.bar_kv_peer_cp_async);
                    kerutils::cp_async_bulk_shared_cta_to_shared_cluster(
                        peer_kv_peer_ptr,
                        plan.u.q_kv.k_nope.data(),
                        sizeof(bf16) * KV_PEER_ELEMENTS,
                        *peer_bar_ptr
                    );
                    fence_view_async_shared();
                    plan.bar_kv_peer_cp_async.wait(phase);
                    if (cta_idx == 0) {
                        plan.bar_kv_peer_ready.arrive(0u);
                    } else {
                        plan.bar_kv_peer_ready.arrive(1u);
                    }
                }
            }
        }
    }
    
    // ========================================
    // Warpgroup 2: dKV Transfer (WG2)
    // Responsibility: Read dKV from TMEM and atomicAdd to global memory
    // ========================================
    if (is_wg2) {
        const int row = idx_in_warpgroup % B_TOPK;   // 0-63: which KV row
        const int half = idx_in_warpgroup / B_TOPK;   // 0 or 1: which column half

        CUTE_NO_UNROLL
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            const int phase = k_block & 1;
            const int row_base = k_block * B_TOPK;
            const int row_global = row_base + row;
            int kv_idx = -1;
            if (row_global < topk_length) {
                kv_idx = __ldg(gIndices_s + row_global);
            }
            const bool row_valid = kv_idx >= 0 && kv_idx < s_kv;

            plan.bar_dkv_part0_ready.wait(phase);
            ku::tcgen05_after_thread_sync();

            constexpr int COLS_PER_HALF = 256 / 2;
            constexpr int CHUNK_SIZE = COLS_PER_HALF / 4;
            CUTE_UNROLL
            for (int chunk = 0; chunk < 4; ++chunk) {
                float2 dkv_data[CHUNK_SIZE / 2];
                ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dKV + chunk * CHUNK_SIZE, dkv_data);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                if (row_valid) {
                    float* dst = dKV + kv_idx * D_K + half * COLS_PER_HALF + chunk * CHUNK_SIZE;
                    float* src = (float*)dkv_data;
                    // 手动展开的PTX原子加法，确保dst和src寻址正确
                    atomic_add_32floats_unrolled(dst, src);
                }
            }
            if (cta_idx == 0) {
                plan.bar_dkv_part0_done.arrive(0u);
            } else {
                plan.bar_dkv_part0_done.arrive(1u);
            }

            plan.bar_dkv_part1_ready.wait(phase);
            ku::tcgen05_after_thread_sync();
            CUTE_UNROLL
            for (int chunk = 0; chunk < 4; ++chunk) {
                float2 dkv_data[CHUNK_SIZE / 2];
                ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::dKV + chunk * CHUNK_SIZE, dkv_data);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                if (row_valid) {
                    float* dst = dKV + kv_idx * D_K + 256 + half * COLS_PER_HALF + chunk * CHUNK_SIZE;
                    float* src = (float*)dkv_data;
                    // 手动展开的PTX原子加法，确保dst和src寻址正确
                    atomic_add_32floats_unrolled(dst, src);
                }
            }
            if (cta_idx == 0) {
                plan.bar_dkv_part1_done.arrive(0u);
            } else {
                plan.bar_dkv_part1_done.arrive(1u);
            }

            plan.bar_dkv_part2_ready.wait(phase);
            ku::tcgen05_after_thread_sync();
            constexpr int ROPE_COLS_PER_HALF = D_ROPE / 2;
            float2 dkv_rope_data[ROPE_COLS_PER_HALF / 2];
            ku::tmem_ld_32dp32bNx<ROPE_COLS_PER_HALF>(tmem_cols::dKV_RoPE, dkv_rope_data);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            if (row_valid) {
                float* dst = dKV + kv_idx * D_K + D_V + half * ROPE_COLS_PER_HALF;
                float* src = (float*)dkv_rope_data;
                // 手动展开的PTX原子加法，确保dst和src寻址正确
                // ROPE_COLS_PER_HALF = 32，与CHUNK_SIZE相同
                atomic_add_32floats_unrolled(dst, src);
            }
            if (cta_idx == 0) {
                plan.bar_dkv_part2_done.arrive(0u);
            } else {
                plan.bar_dkv_part2_done.arrive(1u);
            }
        }
    }

    // ========================================
    // Warpgroup 3: Matrix Multiplication (WG3)
    // Responsibility: Compute P, dP, and dKV
    // ========================================
    if (is_wg3) {
        // Allocate TMEM tensors for P and dP
        TiledMMA_P tiled_mma_P{};
        TiledMMA_dP tiled_mma_dP{};
        Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_H/2>, Int<B_TOPK>>{});
        Tensor tdP = partition_fragment_C(tiled_mma_dP, Shape<Int<B_H/2>, Int<B_TOPK>>{});
        tP.data().get() = tmem_cols::P;
        tdP.data().get() = tmem_cols::dP;

        // Allocate TMEM tensors for dKV
        TiledMMA_dKV tiled_mma_dKV{};
        TiledMMA_dKV_RoPE tiled_mma_dKV_RoPE{};
        Tensor tdKV = partition_fragment_C(tiled_mma_dKV, Shape<Int<B_TOPK>, Int<256>>{});
        tdKV.data().get() = tmem_cols::dKV;
        Tensor tdKV_RoPE = partition_fragment_C(tiled_mma_dKV_RoPE, Shape<Int<B_TOPK>, Int<D_ROPE>>{});
        tdKV_RoPE.data().get() = tmem_cols::dKV_RoPE;

        // Extract V from memory: V uses the same layout as K_NoPE
        Tensor sV = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutV{});

        // Loop body is executed by one elected warp in each CTA.
        if (warp_idx == 12 && elect_one_sync()) {
            if (cta_idx == 0) {
                // Q and dO prologue synchronization is consumed in WG3 before any MMA use.
                plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H * D_ROPE * sizeof(bf16));
                plan.bar_prologue_dO.arrive_and_expect_tx(B_H * D_V * sizeof(bf16));
                plan.bar_prologue_q_nope.wait(0);
                plan.bar_prologue_q_rope.wait(0);
                plan.bar_prologue_dO.wait(0);
                ku::tcgen05_after_thread_sync();
            }

            // S and dS tensors for MMA A operand
            Tensor sS_mma = make_tensor(make_smem_ptr(plan.s_ds.s.data()), SmemLayoutS{});
            Tensor sDS_mma = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdS{});

            // dO transposed: full [D_V, B_H/2] = [512, 64], then flat_divide into [256, 64] tiles
            Tensor sdO_t_full = make_tensor(make_smem_ptr(plan.dO.data()), SmemLayoutQTilesTransposed<D_V/64>{});
            auto sdO_t_div = flat_divide(sdO_t_full, Shape<Int<256>, Int<B_H/2>>{});

            // Q NoPE transposed: full [D_V, B_H/2] = [512, 64], then flat_divide into [256, 64] tiles
            Tensor sQ_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.q_nope.data()), SmemLayoutQTilesTransposed<D_V/64>{});
            auto sQ_t_div = flat_divide(sQ_t_full, Shape<Int<256>, Int<B_H/2>>{});

            // Q RoPE transposed: [D_ROPE, B_H/2] = [64, 64]
            Tensor sQ_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.q_rope.data()), SmemLayoutQRoPETransposed{});

            // Allocate TMEM tensors for dQ
            TiledMMA_dQ tiled_mma_dQ{};
            TiledMMA_dQ_RoPE tiled_mma_dQ_RoPE{};
            Tensor tdQ_part0 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H/2>, Int<256>>{});
            tdQ_part0.data().get() = tmem_cols::dQ;
            Tensor tdQ_part1 = partition_fragment_C(tiled_mma_dQ, Shape<Int<B_H/2>, Int<256>>{});
            tdQ_part1.data().get() = tmem_cols::dQ + 128;
            Tensor tdQ_RoPE = partition_fragment_C(tiled_mma_dQ_RoPE, Shape<Int<B_H/2>, Int<D_ROPE>>{});
            tdQ_RoPE.data().get() = tmem_cols::dQ_RoPE;

            // dS transposed tensor for MMA A operand: [B_H/2, B_TOPK] -> A matrix
            Tensor sDS_t = make_tensor(make_smem_ptr(plan.s_ds.ds.data()), SmemLayoutdSTransposed{});
            auto sDS_t_div = flat_divide(sDS_t, Shape<Int<B_H/2>, Int<B_TOPK/2>>{});

            // K NoPE/RoPE transposed
            Tensor sK_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.k_nope.data()), SmemLayoutKVTilesTransposed<D_V/64>{});
            auto sK_nope_t_div = flat_divide(sK_nope_t_full, Shape<Int<256>, Int<B_TOPK/2>>{});
            Tensor sK_rope_t = make_tensor(make_smem_ptr(plan.u.q_kv.k_rope.data()), SmemLayoutKRoPETransposed{});

            // K_peer transposed
            Tensor sK_peer_nope_t_full = make_tensor(make_smem_ptr(plan.u.q_kv.kv_peer.data()), SmemLayoutKVTilesTransposed<D_V/64>{});
            auto sK_peer_nope_t_div = flat_divide(sK_peer_nope_t_full, Shape<Int<256>, Int<B_TOPK/2>>{});
            Tensor sK_peer_rope_t = make_tensor(
                make_smem_ptr(plan.u.q_kv.kv_peer.data() + (B_TOPK/2) * D_V),
                SmemLayoutKRoPETransposed{}
            );

            CUTE_NO_UNROLL
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int phase = k_block & 1;
                const bool dq_clear = (k_block == 0);

                // Pipeline dependency:
                // Reuse dKV(0:511) TMEM buffer for part0(k) only after part1(k-1) is fully drained by WG2.
                if (k_block > 0) {
                    const int prev_phase = (k_block - 1) & 1;
                    plan.bar_dkv_part1_done.wait(prev_phase);
                    ku::tcgen05_after_thread_sync();
                }

                // CTA0 computes P/dP and notifies WG0.
                if (cta_idx == 0) {
                    // Wait for WG1 KV TMA completion of current block.
                    plan.bar_prologue_kv.arrive_and_expect_tx(B_TOPK * D_K * sizeof(bf16));
                    plan.bar_prologue_kv.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_P, sQNoPE, sKNoPE, tP, true);
                    ku::utcmma_ss(tiled_mma_P, sQRoPE, sKRoPE, tP, false);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_p_ready, 1|2);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_dP, sdO, sV, tdP, true);
                    ku::umma_arrive_multicast_2x1SM_noelect(plan.bar_dp_ready, 1|2);
                    ku::tcgen05_after_thread_sync();
                }

                // dKV part0: dV[0:256] + dK[0:256]
                plan.bar_s_ready.wait(phase);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_dKV, sS_mma, sdO_t_div(_, _, _0{}, _0{}), tdKV, true);

                plan.bar_ds_ready.wait(phase);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_dKV, sDS_mma, sQ_t_div(_, _, _0{}, _0{}), tdKV, false);
                ku::umma_arrive_noelect(plan.bar_dkv_part0_ready);
                ku::tcgen05_after_thread_sync();

                // dQ: k==0 clear, k>0 accumulate
                if (cta_idx == 0) {
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, dq_clear);
                    plan.bar_kv_peer_ready.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);

                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);

                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_rope_t, tdQ_RoPE, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_peer_rope_t, tdQ_RoPE, false);
                } else {
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, dq_clear);
                    plan.bar_kv_peer_ready.wait(phase);
                    ku::tcgen05_after_thread_sync();
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _0{}, _0{}), tdQ_part0, false);

                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _1{}), sK_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ, sDS_t_div(_, _, _0{}, _0{}), sK_peer_nope_t_div(_, _, _1{}, _0{}), tdQ_part1, false);

                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _1{}), sK_rope_t, tdQ_RoPE, dq_clear);
                    ku::utcmma_ss(tiled_mma_dQ_RoPE, sDS_t_div(_, _, _0{}, _0{}), sK_peer_rope_t, tdQ_RoPE, false);
                }
                ku::umma_arrive_noelect(plan.bar_dq_ready);
                ku::tcgen05_after_thread_sync();

                // dKV part1: dV[256:512] + dK[256:512]
                plan.bar_dkv_part0_done.wait(phase);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_dKV, sS_mma, sdO_t_div(_, _, _1{}, _0{}), tdKV, true);
                ku::utcmma_ss(tiled_mma_dKV, sDS_mma, sQ_t_div(_, _, _1{}, _0{}), tdKV, false);
                ku::umma_arrive_noelect(plan.bar_dkv_part1_ready);
                ku::tcgen05_after_thread_sync();

                // dKV part2: dK_rope
                // Reuse dKV_RoPE TMEM buffer for part2(k) only after part2(k-1) is drained by WG2.
                if (k_block > 0) {
                    const int prev_phase = (k_block - 1) & 1;
                    plan.bar_dkv_part2_done.wait(prev_phase);
                    ku::tcgen05_after_thread_sync();
                }
                ku::utcmma_ss(tiled_mma_dKV_RoPE, sDS_mma, sQ_rope_t, tdKV_RoPE, true);
                ku::umma_arrive_noelect(plan.bar_dkv_part2_ready);
                ku::tcgen05_after_thread_sync();
            }

            // Drain outstanding part1/part2 writes of the final k-block before TMEM free.
            if (num_k_blocks > 0) {
                const int final_phase = (num_k_blocks - 1) & 1;
                plan.bar_dkv_part1_done.wait(final_phase);
                ku::tcgen05_after_thread_sync();
                plan.bar_dkv_part2_done.wait(final_phase);
                ku::tcgen05_after_thread_sync();
            }

            (void)sdO_t_full;
        } else if (warp_idx == 13) {
            // KV valid loading warp
            static_assert(B_TOPK == 64);
            if (lane_idx < 8) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int32x8_t indices = ldg_256_indices((void*)(gIndices_s + k * B_TOPK + lane_idx * 8));
                    auto is_valid = [&](int index) -> char {
                        return index >= 0 && index < s_kv && index <= max_kv_i;
                    };
                    char is_ks_valid_mask =
                        is_valid(indices.a7) << 7 |
                        is_valid(indices.a6) << 6 |
                        is_valid(indices.a5) << 5 |
                        is_valid(indices.a4) << 4 |
                        is_valid(indices.a3) << 3 |
                        is_valid(indices.a2) << 2 |
                        is_valid(indices.a1) << 1 |
                        is_valid(indices.a0) << 0;

                    plan.bar_k_valid_free.wait(k & 1 ^ 1);
                    plan.is_k_valid[lane_idx] = is_ks_valid_mask;
                    plan.bar_k_valid_ready.arrive();
                }
            }
        }
    }

    // All threads must sync before proceeding
    __syncthreads();
    cluster_sync();
    
    // Free TMEM
    if (warp_idx == 0 && elect_one_sync()) {
        TMEM::Allocator2Sm().free(plan.tmem_start_addr.data()[0], 512);
    }
#endif
}

// Preprocess delta wrapper
void launch_preprocess_delta(
    const bf16* O,
    const bf16* dO,
    float* delta,
    int s_q,
    cudaStream_t stream
) {
    constexpr int B_H = 128;
    dim3 grid(s_q, 1, 1);  // One block per query token
    dim3 block(B_H, 1, 1);  // 128 threads
    
    preprocess_delta_kernel<<<grid, block, 0, stream>>>(O, dO, delta, s_q);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "preprocess_delta_kernel failed with error: %s\n", cudaGetErrorString(err));
        return;
    }
}

// C++ wrapper
void launch_test_mla_bwd(
    const bf16* q,
    const bf16* kv,
    const bf16* dO,
    const float* lse,
    const bf16* O,
    const int32_t* gIndices,
    int s_kv,
    int topk_length,
    int s_q,
    const float* delta,
    float* dKV,
    bf16* dQ,
    cudaStream_t stream
) {
    // Create TMA descriptors for prologue loads (Q/KV/dO)
    auto shape_Q_nope = cute::make_shape(B_H, D_V, s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)q),
            cute::make_layout(shape_Q_nope, cute::make_stride(D_Q, cute::_1{}, B_H * D_Q))
        ),
        SmemLayoutQNoPE{}
    );

    auto shape_Q_rope = cute::make_shape(B_H, D_ROPE, s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)q + D_V),
            cute::make_layout(shape_Q_rope, cute::make_stride(D_Q, cute::_1{}, B_H * D_Q))
        ),
        SmemLayoutQRoPE{}
    );

    auto shape_KV = cute::make_shape(s_kv, D_K);
    auto tma_KV = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)kv),
            cute::make_layout(shape_KV, cute::make_stride(D_K, cute::_1{}))
        ),
        SmemLayoutKV{}
    );

    auto shape_dO = cute::make_shape(B_H, D_V, s_q);
    auto tma_dO = cute::make_tma_copy(
        cute::SM100_TMA_2SM_LOAD_NOSPLIT{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)dO),
            cute::make_layout(shape_dO, cute::make_stride(D_V, cute::_1{}, B_H * D_V))
        ),
        SmemLayoutdO{}
    );

    auto shape_dQ = cute::make_shape(B_H, D_Q, s_q);
    auto tma_dQ = cute::make_tma_copy(
        cute::SM90_TMA_STORE{},
        cute::make_tensor(
            cute::make_gmem_ptr((bf16*)dQ),
            cute::make_layout(shape_dQ, cute::make_stride(D_Q, cute::_1{}, B_H * D_Q))
        ),
        SmemLayoutQ{}
    );

    CUtensorMap tensor_map_kv;
    {
        uint64_t size[2] = {(uint64_t)D_K, (unsigned long)s_kv};  // [D_K, s_kv]
        uint64_t stride[1] = {(uint64_t)D_K * sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<bf16*>(kv),
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (res != CUresult::CUDA_SUCCESS) {
            fprintf(stderr, "cuTensorMapEncodeTiled failed with error code: %d\n", static_cast<int>(res));
            return;
        }
    }

    using KernelTmaParams = TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_KV), decltype(tma_KV),
        decltype(shape_dO), decltype(tma_dO),
        decltype(shape_dQ), decltype(tma_dQ)
    >;

    KernelTmaParams tma_params = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_KV, tma_KV,
        shape_dO, tma_dO,
        shape_dQ, tma_dQ,
        tensor_map_kv
    };

    auto kernel = &test_mla_bwd_kernel<KernelTmaParams>;

    dim3 grid(2 * s_q, 1, 1);  // 2 CTAs per query token
    dim3 block(NUM_THREADS, 1, 1);
    
    // Set dynamic shared memory size
    cudaError_t attr_err = cudaFuncSetAttribute(
        (const void*)kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        SMEM_SIZE
    );
    if (attr_err != cudaSuccess) {
        fprintf(stderr, "cudaFuncSetAttribute failed with error: %s\n", cudaGetErrorString(attr_err));
        return;
    }
    
    // Cluster configuration
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaError_t err = cudaLaunchKernelEx(&config, kernel, 
        q, kv, dO, lse, O, gIndices, s_kv, topk_length, s_q, delta, dKV, dQ, tma_params);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchKernelEx failed with error: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Python binding
std::tuple<torch::Tensor, torch::Tensor> mla_bwd_forward(
    torch::Tensor q, torch::Tensor kv, torch::Tensor dO, torch::Tensor lse, torch::Tensor O, torch::Tensor indices) {
    // Check inputs
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(kv.is_cuda(), "kv must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");
    TORCH_CHECK(lse.is_cuda(), "lse must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.dtype() == torch::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(dO.dtype() == torch::kBFloat16, "dO must be bfloat16");
    TORCH_CHECK(lse.dtype() == torch::kFloat32, "lse must be float32");
    TORCH_CHECK(O.dtype() == torch::kBFloat16, "O must be bfloat16");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == B_H && q.size(2) == D_Q, 
                "q shape must be [s_q, 128, 576]");
    TORCH_CHECK(kv.dim() == 2 && kv.size(1) == D_K,
                "kv shape must be [s_kv, 576]");
    const int64_t s_q = q.size(0);
    TORCH_CHECK(s_q > 0, "s_q must be > 0");
    TORCH_CHECK(dO.dim() == 3 && dO.size(0) == s_q && dO.size(1) == B_H && dO.size(2) == D_V,
                "dO shape must be [s_q, 128, 512]");
    TORCH_CHECK(lse.dim() == 2 && lse.size(0) == s_q && lse.size(1) == B_H,
                "lse shape must be [s_q, 128]");
    TORCH_CHECK(O.dim() == 3 && O.size(0) == s_q && O.size(1) == B_H && O.size(2) == D_V,
                "O shape must be [s_q, 128, 512]");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == s_q,
                "indices must be [s_q, topk_length]");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(kv.is_contiguous(), "kv must be contiguous");
    TORCH_CHECK(dO.is_contiguous(), "dO must be contiguous");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    const int64_t s_kv = kv.size(0);
    const int64_t topk_length = indices.size(1);
    TORCH_CHECK(topk_length > 0, "topk_length must be > 0");
    TORCH_CHECK(topk_length % B_TOPK == 0,
                "topk_length must be a multiple of tile B_TOPK=64");
    TORCH_CHECK(s_kv > 0, "s_kv must be > 0");
    TORCH_CHECK(topk_length <= s_kv,
                "indices.size(1) must be <= kv.size(0)");
    
    // Create output tensors
    auto options_f32 = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(q.device());
    
    torch::Tensor delta = torch::empty({s_q, B_H}, options_f32);
    auto options_bf16 = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(q.device());

    torch::Tensor dKV = torch::zeros({s_kv, D_K}, options_f32);
    torch::Tensor dQ = torch::zeros({s_q, B_H, D_Q}, options_bf16);
    
    // Get data pointers
    const bf16* q_ptr = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
    const bf16* kv_ptr = reinterpret_cast<const bf16*>(kv.data_ptr<at::BFloat16>());
    const bf16* dO_ptr = reinterpret_cast<const bf16*>(dO.data_ptr<at::BFloat16>());
    const float* lse_ptr = lse.data_ptr<float>();
    const bf16* O_ptr = reinterpret_cast<const bf16*>(O.data_ptr<at::BFloat16>());
    const int32_t* indices_ptr = indices.data_ptr<int32_t>();
    float* delta_ptr = delta.data_ptr<float>();
    float* dKV_ptr = dKV.data_ptr<float>();
    bf16* dQ_ptr = reinterpret_cast<bf16*>(dQ.data_ptr<at::BFloat16>());
    
    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Preprocess delta: compute delta = sum(O * dO, dim=-1)
    const int s_q_i = static_cast<int>(s_q);
    launch_preprocess_delta(O_ptr, dO_ptr, delta_ptr, s_q_i, stream);
    
    const int s_kv_i = static_cast<int>(s_kv);
    const int topk_length_i = static_cast<int>(topk_length);
    
    // Call kernel once; topk loop is handled inside WG0/WG1/WG2/WG3.
    launch_test_mla_bwd(q_ptr, kv_ptr, dO_ptr, lse_ptr, O_ptr, indices_ptr, s_kv_i, topk_length_i, s_q_i,
        delta_ptr, dKV_ptr, dQ_ptr, stream);

    return std::make_tuple(dQ, dKV);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_bwd", &mla_bwd_forward, 
          "Test mla_bwd kernel (CUDA). Returns (dQ, dKV)");
}
