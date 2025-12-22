============================================================
Triton Fused 测试
============================================================
参数: batch=1, heads=16, seq=4096, dim=256, topk=2048
============================================================

>>> 精度测试

[Kernel 1: Attention Softmax]
  Max diff: 2.145469e-04

[Kernel 2: Post-Reduce Loss]
  Ref: 2242.793945, Triton: 2242.793945
  Diff: 0.000000e+00

[完整流程]
  PyTorch: 2242.793945
  Triton Fused: 2242.793945
  Diff: 0.000000e+00

>>> 性能测试 (warmup=5, iterations=10)

  PyTorch: 7.902 ms
  Triton Fused: 2005.404 ms
  加速比: 0.00x
