============================================================
Testing sparse_mla_fwd_debug with small parameters
B=1, S=4, SKV=8, H=16, HKV=1
DQK=576, DV=512, topk=64
============================================================

[INFO] Running kernel with T.print debug statements...
============================================================
2025-12-19 12:18:32  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-2, -1]`
ptxas error   : Entry function 'main_kernel' uses too much shared data (0x12400 bytes, 0xc000 max)
Traceback (most recent call last):
  File "/home/users/chenquanlin/workspace/a0p6b_dsa_stage2/test2.py", line 287, in <module>
    test_sparse_mla_fwd_debug()
  File "/home/users/chenquanlin/workspace/a0p6b_dsa_stage2/test2.py", line 271, in test_sparse_mla_fwd_debug
    tl_out, tl_lse = sparse_mla_fwd_debug_interface(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/a0p6b_dsa_stage2/test2.py", line 225, in sparse_mla_fwd_debug_interface
    kernel = sparse_mla_fwd_debug(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/tilelang/jit/__init__.py", line 205, in wrapper
    kernel_result = compile(
                    ^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/tilelang/jit/__init__.py", line 70, in compile
    return cached(
           ^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/tilelang/cache/__init__.py", line 29, in cached
    return _kernel_cache_instance.cached(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/tilelang/cache/kernel_cache.py", line 185, in cached
    kernel = JITKernel(
             ^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/tilelang/jit/kernel.py", line 121, in __init__
    adapter = self._compile_and_create_adapter(func, out_idx)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/tilelang/jit/kernel.py", line 250, in _compile_and_create_adapter
    adapter = CythonKernelAdapter(
              ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/tilelang/jit/adapter/cython/adapter.py", line 128, in __init__
    self.lib_generator.compile_lib()
  File "/usr/local/lib/python3.12/dist-packages/tilelang/jit/adapter/libgen.py", line 164, in compile_lib
    raise RuntimeError(f"Compilation Failed! {command}"
RuntimeError: Compilation Failed! ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-w', '-Xcudafe', '--diag_suppress=177', '--compiler-options', '-fPIC', '-lineinfo', '--shared', '/tmp/tmpvkpsdjzd.cu', '-lcuda', '-gencode', 'arch=compute_90a,code=sm_90a', '-I/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/cutlass/include', '-I/usr/local/lib/python3.12/dist-packages/tilelang/3rdparty/../src', '-o', '/tmp/tmpvkpsdjzd.so']
 #include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(int* __restrict__ Indices, bfloat16_t* __restrict__ KV, float* __restrict__ Lse, bfloat16_t* __restrict__ Output, bfloat16_t* __restrict__ Q, int batch, int seq_len, int seq_len_kv);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(int* __restrict__ Indices, bfloat16_t* __restrict__ KV, float* __restrict__ Lse, bfloat16_t* __restrict__ Output, bfloat16_t* __restrict__ Q, int batch, int seq_len, int seq_len_kv) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[32];
  float sumexp[2];
  float m_i[2];
  signed char mask[2];
  float acc_s[4];
  __shared__ float smem[1024];
  float m_i_prev[2];
  float alpha[2];
  float sumexp_i[2];
  __shared__ float smem_1[16];
  __shared__ float smem_2[16];
  __shared__ float smem_3[16];
  __shared__ float smem_4[1024];
  __shared__ float smem_5[8192];
  __shared__ float smem_6[8192];
  __shared__ float smem_7[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float2*)(acc_o + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    sumexp[i_1] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    m_i[i_2] = -0x1p+30f/*-1.073742e+09*/;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 63) >> 3) * 1024) + (i_3 * 256)) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int64_t)((int)blockIdx.x)) * (int64_t)9216) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)9216)) + (((int64_t)i_3) * (int64_t)2304)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)576)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)));
  }
  *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 4) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8192)) = *(uint2*)(Q + (((((((int64_t)((int)blockIdx.x)) * (int64_t)9216) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)9216)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)576)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)4)) + (int64_t)512));
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    debug_print_var("[DEBUG] b_i:", ((int)blockIdx.y));
    debug_print_var("[DEBUG] s_i:", ((int)blockIdx.x));
    debug_print_var("[DEBUG] H0:", 0);
    debug_print_var("[DEBUG] H1:", 16);
    if (((int)threadIdx.x) == 0) {
      for (int i_4 = 0; i_4 < 8192; ++i_4) {
        debug_print_buffer_value("[DEBUG] Q_shared after copy:", "Q_shared", i_4, ((bfloat16_t*)buf_dyn_shmem)[((((((((i_4 & 511) >> 6) * 1024) + ((i_4 >> 9) * 64)) + (((((i_4 & 4095) >> 11) + ((i_4 & 63) >> 5)) & 1) * 32)) + (((((i_4 & 2047) >> 10) + ((i_4 & 31) >> 4)) & 1) * 16)) + (((((i_4 & 1023) >> 9) + ((i_4 & 15) >> 3)) & 1) * 8)) + (i_4 & 7))]);
      }
      for (int i_5 = 0; i_5 < 1024; ++i_5) {
        debug_print_buffer_value("[DEBUG] Q_tail_shared after copy:", "Q_tail_shared", i_5, ((bfloat16_t*)buf_dyn_shmem)[(((((((i_5 >> 6) * 64) + (((((i_5 & 511) >> 8) + ((i_5 & 63) >> 5)) & 1) * 32)) + (((((i_5 & 255) >> 7) + ((i_5 & 31) >> 4)) & 1) * 16)) + (((((i_5 & 127) >> 6) + ((i_5 & 15) >> 3)) & 1) * 8)) + (i_5 & 7)) + 8192)]);
      }
    }
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_6 * 4096) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 18432), KV+((((((int64_t)Indices[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)64)) + (((int64_t)i_6) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3))]) * (int64_t)576) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)576)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)512), ((0 <= Indices[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)64)) + (((int64_t)i_6) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3))]) && (Indices[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)64)) + (((int64_t)i_6) * (int64_t)32)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)3))] < seq_len_kv)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 63) >> 3) * 8192) + (i_7 * 512)) + ((((int)threadIdx.x) >> 6) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_7 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 34816), KV+(((((int64_t)Indices[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)64)) + (((int64_t)i_7) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6))]) * (int64_t)576) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len_kv)) * (int64_t)576)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)63) * (int64_t)8)), ((0 <= Indices[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)64)) + (((int64_t)i_7) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6))]) && (Indices[((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)64)) + (((int64_t)i_7) * (int64_t)4)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)6))] < seq_len_kv)));
  }
  tl::cp_async_commit();
  tl::fence_proxy_async();
  tl::fence_proxy_async();
  char2 __1;
  ushort2 __2;
    int2 v_ = *(int2*)(Indices + ((((((int64_t)((int)blockIdx.x)) * (int64_t)64) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)));
    int2 v__1 = make_int2(((int)blockIdx.x), ((int)blockIdx.x));
    __2.x = (v_.x<=v__1.x);
    __2.y = (v_.y<=v__1.y);
  __1.x=((signed char)(__2.x));
  __1.y=((signed char)(__2.y));
  *(char2*)(mask + 0) = __1;
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    debug_print_var("[DEBUG] i_i (iteration):", 0);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if (((int)threadIdx.x) == 0) {
      for (int i_8 = 0; i_8 < 32768; ++i_8) {
        debug_print_buffer_value("[DEBUG] KV_shared after copy:", "KV_shared", i_8, ((bfloat16_t*)buf_dyn_shmem)[(((((((((i_8 & 511) >> 6) * 4096) + ((i_8 >> 9) * 64)) + (((((i_8 & 4095) >> 11) + ((i_8 & 63) >> 5)) & 1) * 32)) + (((((i_8 & 2047) >> 10) + ((i_8 & 31) >> 4)) & 1) * 16)) + (((((i_8 & 1023) >> 9) + ((i_8 & 15) >> 3)) & 1) * 8)) + (i_8 & 7)) + 17408)]);
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if (((int)threadIdx.x) == 0) {
      for (int i_9 = 0; i_9 < 4096; ++i_9) {
        debug_print_buffer_value("[DEBUG] K_tail_shared after copy:", "K_tail_shared", i_9, ((bfloat16_t*)buf_dyn_shmem)[(((((((i_9 >> 6) * 64) + (((((i_9 & 511) >> 8) + ((i_9 & 63) >> 5)) & 1) * 32)) + (((((i_9 & 255) >> 7) + ((i_9 & 31) >> 4)) & 1) * 16)) + (((((i_9 & 127) >> 6) + ((i_9 & 15) >> 3)) & 1) * 8)) + (i_9 & 7)) + 9216)]);
      }
    }
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 4; ++i_10) {
    float condval;
    if (((bool)mask[(i_10 & 1)])) {
      condval = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval = -CUDART_INF_F;
    }
    acc_s[i_10] = condval;
  }
  tl::fence_proxy_async();
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<16, 64, 512, 1, 8, 0, 1, 0, 512, 512, 0, 0, false>((&(((bfloat16_t*)buf_dyn_shmem)[0])), (&(((bfloat16_t*)buf_dyn_shmem)[17408])), (&(acc_s[0])));
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<16, 64, 64, 1, 8, 0, 1, 0, 64, 64, 0, 0, false>((&(((bfloat16_t*)buf_dyn_shmem)[8192])), (&(((bfloat16_t*)buf_dyn_shmem)[9216])), (&(acc_s[0])));
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      *(float2*)(smem + ((((i_11 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_11 * 2));
    }
  }
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if (((int)threadIdx.x) == 0) {
      for (int i_12 = 0; i_12 < 1024; ++i_12) {
        debug_print_buffer_value("[DEBUG] acc_s after GEMM (QK^T):", "acc_s", i_12, smem[i_12]);
      }
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    m_i_prev[i_13] = m_i[i_13];
  }
  __syncthreads();
  #pragma unroll
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    #pragma unroll
    for (int rv = 0; rv < 2; ++rv) {
      m_i[i_14] = max(m_i[i_14], acc_s[((i_14 * 2) + rv)]);
    }
    m_i[i_14] = tl::AllReduce<tl::MaxOp, 256, 32, 0, 256>::run_hopper(m_i[i_14], (&(((float*)buf_dyn_shmem)[0])));
    m_i[i_14] = tl::AllReduce<tl::MaxOp, 4, 1, 0, 256>::run_hopper(m_i[i_14]);
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 2; ++i_15) {
    m_i[i_15] = max(m_i[i_15], m_i_prev[i_15]);
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 2; ++i_16) {
    alpha[i_16] = exp2f(((m_i_prev[i_16] - m_i[i_16]) * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/));
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 4; ++i_17) {
    acc_s[i_17] = exp2f(((acc_s[i_17] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/) - (m_i[(i_17 >> 1)] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    sumexp_i[i_18] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
      sumexp_i[i_18] = (sumexp_i[i_18] + acc_s[((i_18 * 2) + rv_1)]);
    }
    sumexp_i[i_18] = tl::AllReduce<tl::SumOp, 256, 32, 0, 256>::run_hopper(sumexp_i[i_18], (&(((float*)buf_dyn_shmem)[0])));
    sumexp_i[i_18] = tl::AllReduce<tl::SumOp, 4, 1, 0, 256>::run_hopper(sumexp_i[i_18]);
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 2; ++i_19) {
    sumexp[i_19] = ((sumexp[i_19] * alpha[i_19]) + sumexp_i[i_19]);
  }
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        smem_1[((i_20 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = m_i[i_20];
      }
    }
  }
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if (((int)threadIdx.x) == 0) {
      for (int i_21 = 0; i_21 < 16; ++i_21) {
        debug_print_buffer_value("[DEBUG] m_i (row max):", "m_i", i_21, smem_1[i_21]);
      }
    }
  }
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
      #pragma unroll
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        smem_2[((i_22 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = alpha[i_22];
      }
    }
  }
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if (((int)threadIdx.x) == 0) {
      for (int i_23 = 0; i_23 < 16; ++i_23) {
        debug_print_buffer_value("[DEBUG] alpha (rescale factor):", "alpha", i_23, smem_2[i_23]);
      }
    }
  }
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        smem_3[((i_24 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = sumexp[i_24];
      }
    }
  }
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if (((int)threadIdx.x) == 0) {
      for (int i_25 = 0; i_25 < 16; ++i_25) {
        debug_print_buffer_value("[DEBUG] sumexp (sum of exp):", "sumexp", i_25, smem_3[i_25]);
      }
    }
  }
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      *(float2*)(smem_4 + ((((i_26 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_s + (i_26 * 2));
    }
  }
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if (((int)threadIdx.x) == 0) {
      for (int i_27 = 0; i_27 < 1024; ++i_27) {
        debug_print_buffer_value("[DEBUG] acc_s after softmax (P):", "acc_s", i_27, smem_4[i_27]);
      }
    }
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 32; ++i_28) {
    acc_o[i_28] = (acc_o[i_28] * alpha[((i_28 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 1; ++i_29) {
    tl::ptx_stmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)acc_s[0]), ((bfloat16_t)acc_s[1])), __pack_half2(((bfloat16_t)acc_s[2]), ((bfloat16_t)acc_s[3])));
  }
  tl::fence_proxy_async();
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<16, 512, 64, 1, 8, 0, 0, 0, 64, 512, 0, 0, false>((&(((bfloat16_t*)buf_dyn_shmem)[0])), (&(((bfloat16_t*)buf_dyn_shmem)[17408])), (&(acc_o[0])));
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 16; ++i_30) {
      *(float2*)(smem_5 + ((((((i_30 & 1) * 4096) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((i_30 >> 1) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o + (i_30 * 2));
    }
  }
  __syncthreads();
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    if (((int)threadIdx.x) == 0) {
      for (int i_31 = 0; i_31 < 8192; ++i_31) {
        debug_print_buffer_value("[DEBUG] acc_o after PV GEMM (first iter):", "acc_o", i_31, smem_5[i_31]);
      }
    }
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 32; ++i_32) {
    acc_o[i_32] = (acc_o[i_32] / sumexp[((i_32 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 2; ++i_33) {
    sumexp[i_33] = (log2f(sumexp[i_33]) + (m_i[i_33] * 0x1.ec709dbe8903ep-5f/*6.011229e-02*/));
  }
  if ((((int)blockIdx.x) == 0) && (((int)blockIdx.y) == 0)) {
    #pragma unroll
    for (int i_34 = 0; i_34 < 16; ++i_34) {
      *(float2*)(smem_6 + ((((((i_34 & 1) * 4096) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + ((i_34 >> 1) * 64)) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(acc_o + (i_34 * 2));
    }
    __syncthreads();
    if (((int)threadIdx.x) == 0) {
      for (int i_35 = 0; i_35 < 8192; ++i_35) {
        debug_print_buffer_value("[DEBUG] acc_o (final output after rescale):", "acc_o", i_35, smem_6[i_35]);
      }
    }
    if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
      #pragma unroll
      for (int i_36 = 0; i_36 < 2; ++i_36) {
        smem_7[((i_36 * 8) + ((((int)threadIdx.x) & 31) >> 2))] = sumexp[i_36];
      }
    }
    __syncthreads();
    if (((int)threadIdx.x) == 0) {
      for (int i_37 = 0; i_37 < 16; ++i_37) {
        debug_print_buffer_value("[DEBUG] sumexp (final LSE):", "sumexp", i_37, smem_7[i_37]);
      }
    }
  }
  #pragma unroll
  for (int i_38 = 0; i_38 < 16; ++i_38) {
    uint1 __3;
    float2 v__2 = *(float2*)(acc_o + (i_38 * 2));
    *reinterpret_cast<__nv_bfloat162*>(&(__3)) = __float22bfloat162_rn(*(float2*)(&(v__2)));
    *(uint1*)(Output + (((((((((int64_t)((int)blockIdx.x)) * (int64_t)8192) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)8192)) + ((((int64_t)i_38) & (int64_t)1) * (int64_t)4096)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)512)) + ((((int64_t)i_38) >> (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = __3;
  }
  if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
    #pragma unroll
    for (int i_39 = 0; i_39 < 2; ++i_39) {
      Lse[((((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_len)) * (int64_t)16)) + (((int64_t)i_39) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2))] = sumexp[i_39];
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 165888);
    if (result_main_kernel != cudaSuccess) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 165888, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ KV, int* __restrict__ Indices, bfloat16_t* __restrict__ Output, float* __restrict__ Lse, int batch, int seq_len, int seq_len_kv, cudaStream_t stream=cudaStreamDefault) {
        main_kernel<<<dim3(seq_len, batch, 1), dim3(256, 1, 1), 165888, stream>>>(Indices, KV, Lse, Output, Q, batch, seq_len, seq_len_kv);
        TILELANG_CHECK_LAST_ERROR("main_kernel");

        return 0;
}
