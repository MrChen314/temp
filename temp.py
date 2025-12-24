root@gajl-55668-20250606chenyuanfang104:/home/users/chenquanlin# python test.py

======================================================================
精度测试 (PyTorch Full vs Triton Sparse)
======================================================================


======================================================================
Triton Sparse vs PyTorch Full 性能测试
======================================================================
参数: batch=1, heads=128, chunk=8192, seq=16384, dim=512, topk=2048
Sparse复杂度: O(chunk * topk * head_dim * num_heads) = O(1,099,511,627,776)
Full复杂度:   O(chunk * seq * head_dim * num_heads) = O(8,796,093,022,208)
理论加速比:   seq / topk = 16384 / 2048 = 8.00x
======================================================================

[测试] Triton V1 (单kernel, 串行heads)...
[测试] Triton V2 (两阶段kernel, 并行heads)...
[测试] PyTorch Full 参考...

>>> 性能结果 (warmup=1, iters=3)
  Triton V1 (串行heads):   1084.897 ms
  Triton V2 (并行heads):   1059.508 ms (vs V1: 1.02x)
  PyTorch Full:            257.236 ms
  V2 vs PyTorch:           0.24x 减速
