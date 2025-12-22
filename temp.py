root@gajl-bbc-onlinec-com-1585189:/home/users/chenquanlin# python /home/users/chenquanlin/workspace/chunk_loss/test.py

======================================================================
精度测试
======================================================================

[小规模测试]
Traceback (most recent call last):
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 627, in <module>
    test_kernel1_accuracy(batch_size=1, num_heads=4, seq_len=64, head_dim=32, topk=16)
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 446, in test_kernel1_accuracy
    tri_attn = sparse_attention_softmax_fused(query, key, indices, scaling)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 239, in sparse_attention_softmax_fused
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
triton.compiler.errors.UnsupportedLanguageConstruct: at 35:12:
    pid_batch = pid // num_per_batch
    pid_temp = pid % num_per_batch
    pid_head = pid_temp // num_per_head
    pid_m = pid_temp % num_per_head

    NEG_INF = -1e9

    # 处理BLOCK_M行
    for mi in range(BLOCK_M):
        row = pid_m * BLOCK_M + mi
        if row >= seq_len:
            continue
            ^
unsupported AST node type: Continue
