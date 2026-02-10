#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>
#include <kerutils/kerutils.cuh>
#include <cutlass/arch/arch.h>
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
namespace test_operator::k_major_vs_mn_major {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

// Matrix shape: dV = s^T @ dO
// s:  [128, 64]
// dO: [128, 256]
// dV: [64, 256]
static constexpr int M = 64;
static constexpr int K_DIM = 128;
static constexpr int N = 256;
static constexpr int NUM_THREADS = 128;

// A operand layout: K major, shape [M, K] = [64, 128]
using SmemLayoutS = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<M>, Int<K_DIM>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// B operand layouts, both shape [N, K] = [256, 128]
// 1) K major
using SmemLayoutDO_K = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<N>, Int<K_DIM>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// 2) MN major
using SmemLayoutDO_MN = decltype(coalesce(tile_to_shape(
    UMMA::Layout_MN_SW128_Atom<bf16>{},
    Shape<Int<N>, Int<K_DIM>>{},
    Step<_2, _1>{}
), Shape<_1, _1>{}));

using TiledMMA_dKV_K = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_dKV_MN = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, M, N, UMMA::Major::K, UMMA::Major::MN>{}
));

struct tmem_cols {
    static constexpr int dV = 0;
    static_assert(dV + (M * N / 128) <= 512, "TMEM column overflow");
};

struct alignas(128) SharedMemory {
    array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    array_aligned<bf16, cosize_v<SmemLayoutDO_K>> dO_k;
    array_aligned<bf16, cosize_v<SmemLayoutDO_MN>> dO_mn;
    transac_bar_t bar_c_ready;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

static constexpr size_t SMEM_SIZE = sizeof(SharedMemory);

}  // namespace test_operator::k_major_vs_mn_major

__global__ void test_k_mn_major_kernel(
    const test_operator::k_major_vs_mn_major::bf16* __restrict__ s_bf16,  // [128, 64]
    const test_operator::k_major_vs_mn_major::bf16* __restrict__ dO,      // [128, 256]
    float* __restrict__ dV_cuda_k,                                         // [64, 256]
    float* __restrict__ dV_cuda_mn                                         // [64, 256]
);

void launch_test_k_mn_major(
    const test_operator::k_major_vs_mn_major::bf16* s_bf16,
    const test_operator::k_major_vs_mn_major::bf16* dO,
    float* dV_cuda_k,
    float* dV_cuda_mn,
    cudaStream_t stream = nullptr
);
