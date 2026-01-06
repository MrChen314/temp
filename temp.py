root@gajl-55668-20250606chenyuanfang7:/home/users/chenquanlin/workspace/indexer_loss# python tilelang_kernel_pipeline.py 
2026-01-06 13:45:48  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /home/users/chenquanlin/workspace/tilelang/build

====================================================================================================
前向精度测试 (PyTorch attn_sum vs TileLang attn_sum)
====================================================================================================
2026-01-06 13:45:54  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-1]`
2026-01-06 13:45:54  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-1]`
2026-01-06 13:45:56  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-1]`

Name         Config                                                  RefMax       TLMax        RelDiff      Pass  
-------------------------------------------------------------------------------------------------------------
小规模          batch=1, heads=16, chunk=32, seq=64, dim=64, topk=32    Error
  Error: Loop layout is not injective: Fragment([16] -> [2], replicate: 1, thread: 64 + tk_i // 16 * 32 + tk_i % 8 * 4 + 3 + 1, forward_thread: _i // 8 * 64 + tk_i // 16 * 32 + tk_i % 8 * 4 + _i % 8 // 2, forw
中等规模         batch=1, heads=32, chunk=64, seq=128, dim=64, topk=64   Error
  Error: Layout infer conflict between m_block and alpha in T.Parallel loop:
    loop Fragment([16] -> [4], replicate: 1, thread: tk_i // 16 * 32 + tk_i % 8 * 4 + 3 + 1, forward_thread: tk_i // 16 * 32 + tk_i 
大规模          batch=1, heads=64, chunk=128, seq=256, dim=128, topk=128 Error
  Error: Layout infer conflict between m_block and alpha in T.Parallel loop:
    loop Fragment([16] -> [4], replicate: 1, thread: tk_i // 16 * 32 + tk_i % 8 * 4 + 3 + 1, forward_thread: tk_i // 16 * 32 + tk_i 
-------------------------------------------------------------------------------------------------------------
前向测试 (attn_sum): 0/3 通过

================================================================================
性能对比测试 (Triton vs TileLang)
================================================================================
参数: batch=1, heads=128, chunk=4096, seq=8192, dim=576, topk=2048
================================================================================

>>> Triton 性能: 18.402 ms
2026-01-06 13:45:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[-1]`

>>> TileLang 执行失败: Layout infer conflict between m_block and alpha in T.Parallel loop:
    loop Fragment([16] -> [4], replicate: 1, thread: tk_i // 16 * 32 + tk_i % 8 * 4 + 3 + 1, forward_thread: tk_i // 16 * 32 + tk_i % 8 * 4 + _i % 8 // 2, forward_index: [_i // 8 * 2 + _i % 2], thread_range: I.Range(0, 128))
    fragment Fragment([16] -> [4], replicate: 32, thread: 128, forward_thread: _rep % 4 * 32 + _rep // 4 * 4 + _i % 8 // 2, forward_index: [_i // 8 * 2 + _i % 2], thread_range: I.Range(0, 128))

Traceback (most recent call last):
  File "/home/users/chenquanlin/workspace/indexer_loss/tilelang_kernel_pipeline.py", line 795, in test_performance_comparison
    _ = indexer_loss_fwd_tilelang(query, key, topk_indices, scaling, chunk_offset=chunk_offset)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/indexer_loss/tilelang_kernel_pipeline.py", line 483, in indexer_loss_fwd_tilelang
    kernel = indexer_loss_fwd_kernel(
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/jit/__init__.py", line 414, in __call__
    kernel = self.compile(*args, **kwargs, **tune_params)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/jit/__init__.py", line 349, in compile
    kernel_result = compile(
                    ^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/jit/__init__.py", line 98, in compile
    return cached(
           ^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/cache/__init__.py", line 74, in cached
    return _dispatch_map[execution_backend].cached(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/cache/kernel_cache.py", line 204, in cached
    kernel = JITKernel(
             ^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/jit/kernel.py", line 137, in __init__
    adapter = self._compile_and_create_adapter(func, out_idx)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/jit/kernel.py", line 242, in _compile_and_create_adapter
    artifact = tilelang.lower(
               ^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/engine/lower.py", line 267, in lower
    mod = LowerAndLegalize(mod, target)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/tilelang/engine/phase.py", line 168, in LowerAndLegalize
    mod = tilelang.transform.LayoutInference()(mod)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/tilelang/3rdparty/tvm/python/tvm/ir/transform.py", line 167, in __call__
    return _ffi_transform_api.RunPass(self, mod)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "python/tvm_ffi/cython/function.pxi", line 923, in tvm_ffi.core.Function.__call__
tvm.error.InternalError: Layout infer conflict between m_block and alpha in T.Parallel loop:
    loop Fragment([16] -> [4], replicate: 1, thread: tk_i // 16 * 32 + tk_i % 8 * 4 + 3 + 1, forward_thread: tk_i // 16 * 32 + tk_i % 8 * 4 + _i % 8 // 2, forward_index: [_i // 8 * 2 + _i % 2], thread_range: I.Range(0, 128))
    fragment Fragment([16] -> [4], replicate: 32, thread: 128, forward_thread: _rep % 4 * 32 + _rep // 4 * 4 + _i % 8 // 2, forward_index: [_i // 8 * 2 + _i % 2], thread_range: I.Range(0, 128))
