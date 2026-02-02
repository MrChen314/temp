root@gajl-55668-20250606chenyuanfang7:/home/users/chenquanlin/workspace# CUDA_LAUNCH_BLOCKING=1 python /home/users/chenquanlin/workspace/operat
or/attn_dist/test_attn_dist.py
Benchmark Configuration:
  s_q = 16384, h_q = 128, topk = 2048
  Input shape: [16384, 128, 2048]
  sm_scale = 0.125000

Correctness Check:
Traceback (most recent call last):
  File "/home/users/chenquanlin/workspace/operator/attn_dist/test_attn_dist.py", line 123, in <module>
    benchmark()
  File "/home/users/chenquanlin/workspace/operator/attn_dist/test_attn_dist.py", line 86, in benchmark
    triton_out = triton_attn_dist(p_out, sm_scale)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/operator/attn_dist/test_attn_dist.py", line 48, in triton_attn_dist
    triton_attn_dist_kernel[grid](
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 347, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 591, in run
    kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata,
  File "/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py", line 529, in __call__
    self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, global_scratch, *args)
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
