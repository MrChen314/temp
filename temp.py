root@gajl-bbc-onlinec-com-1585189:/home/users/chenquanlin# python /home/users/chenquanlin/workspace/chunk_loss/test.py

============================================================
单独测试 Kernel 1
============================================================
Kernel 1 (Attention Softmax):
  Max diff: 8.034706e-04
  Pass: True

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
参数: batch=1, heads=16, seq=8192, dim=512, topk=2048
============================================================

>>> 精度测试

[Kernel 1: Attention Softmax]
  Max diff: nan

[Kernel 2: Post-Reduce Loss]
  Ref: 4488.436523, Triton: nan
  Diff: nan

[完整流程]
  PyTorch: 4488.436523
  Triton Fused: nan
  Diff: nan

>>> 性能测试 (warmup=2, iterations=3)

  PyTorch: 41.151 ms
  Triton Fused: 190.472 ms
  加速比: 0.22x
