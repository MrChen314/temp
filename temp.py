
测试: 原始方案 (优化1配置)
----------------------------------------
2025-12-18 14:19:36  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-2, -1]`
2025-12-18 14:20:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
  ✓ 正确性验证通过
  平均时间: 1.0651 ms
  带宽: 2.52 TB/s
  TFLOPS: 9.07

测试: 优化2: 减少同步开销
----------------------------------------
2025-12-18 14:20:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-2, -1]`
  ✗ 错误: Layout infer conflict between acc_s and S_local in T.Parallel loop:
    loop Fragment([16, 64] -> [32], replicate: 4, thread: 128, forward_thread: _rep * 32 + _i % 8 * 4 + _j % 8 // 2, forward_index: [_j // 8 * 4 + _i // 8 * 2 + _j % 2], thread_range: I.Range(0, 128))
    fragment Fragment([16, 64] -> [8], replicate: 1, thread: 128, forward_thread: _j % 32 // 8 * 32 + _i % 8 * 4 + _j % 8 // 2, forward_index: [_j // 32 * 4 + _i // 8 * 2 + _j % 2], thread_range: I.Range(0, 128))


测试: 优化3: 预加载 Indices
----------------------------------------
2025-12-18 14:20:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-2, -1]`
2025-12-18 14:20:29  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
  ✗ 正确性验证失败，最大差异: nan
  平均时间: 1.1543 ms
  带宽: 2.33 TB/s
  TFLOPS: 8.37

测试: 优化4: 小 H 特化
----------------------------------------
2025-12-18 14:20:29  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-2, -1]`
  ✗ 错误: Check failed: (M % kMPerWarp == 0) is false: M must be divisible by 16, but got 2

测试: 综合优化 (2+3+4)
----------------------------------------
2025-12-18 14:20:29  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-2, -1]`
  ✗ 错误: Check failed: (M % kMPerWarp == 0) is false: M must be divisible by 16, but got 2

================================================================================
性能汇总
================================================================================
方案                             时间(ms)       带宽(TB/s)     TFLOPS     相对基准      
--------------------------------------------------------------------------------
原始方案 (优化1配置)                   1.0651       2.52         9.07       1.00      x
优化2: 减少同步开销                    失败          
优化3: 预加载 Indices               1.1543       2.33         8.37       0.92      x
优化4: 小 H 特化                    失败          
综合优化 (2+3+4)                   失败          
================================================================================
