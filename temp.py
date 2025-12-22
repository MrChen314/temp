root@gajl-bbc-onlinec-com-1585189:/home/users/chenquanlin# python /home/users/chenquanlin/workspace/chunk_loss/test.py

============================================================
单独测试 Kernel 1
============================================================
Kernel 1 (Attention Softmax):
  Max diff: nan
  Pass: False

============================================================
单独测试 Kernel 2
============================================================
Kernel 2 (Post-Reduce Loss):
  Ref: 3979.246094, Triton: 3979.246094
  Diff: 0.000000e+00
  Pass: True


============================================================
Triton Fused 测试
============================================================
参数: batch=1, heads=16, seq=4096, dim=256, topk=2048
============================================================

>>> 精度测试

[Kernel 1: Attention Softmax]
Traceback (most recent call last):
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 586, in <module>
    test_fused(
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 513, in test_fused
    attn_tri = attention_softmax_fused(query, key, mask, scaling)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/chunk_loss/test.py", line 183, in attention_softmax_fused
    _attention_softmax_kernel_v2[grid](
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 347, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 591, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata,
    ^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 413, in __getattribute__
    self._init_handles()
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 401, in _init_handles
    raise OutOfResources(self.metadata.shared, max_shared, "shared memory")
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 558080, Hardware limit: 232448. Reducing block sizes or `num_stages` may help.
