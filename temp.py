root@gajl-bbc-onlinec-com-1585189:/home/users/chenquanlin# python /home/users/chenquanlin/workspace/chunk_loss/test.py

======================================================================


======================================================================
Sparse Triton H20优化 性能测试
======================================================================
参数: batch=1, heads=16, seq=4096, dim=256, topk=512
理论复杂度: O(seq * topk * head_dim) = O(536,870,912)
======================================================================
Traceback (most recent call last):
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 690, in <module>
    test_performance(
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 569, in test_performance
    _ = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_multirow=True)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 411, in compute_index_loss_sparse
    attn_scores = sparse_attention_softmax_fused(query, key, indices, scaling, use_multirow)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 287, in sparse_attention_softmax_fused
    _sparse_attn_multirow_kernel[grid](
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 347, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/autotuner.py", line 189, in run
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/autotuner.py", line 167, in _bench
    return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/testing.py", line 145, in do_bench
    fn()
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/autotuner.py", line 153, in kernel_call
    self.fn.run(
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 569, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 278, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 81, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 66:12:
            idx_base = idx_batch_base + row * stride_is
            out_base = out_batch_head_base + row * stride_os

            # 加载indices
            indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
            causal_mask = indices > row

            # 计算QK - 分块处理head_dim
            qk = tl.zeros([topk], dtype=tl.float32)

            num_d_blocks: tl.constexpr = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in tl.static_range(num_d_blocks):
            ^
TypeError("'tensor' object cannot be interpreted as an integer")
