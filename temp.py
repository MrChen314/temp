root@gajl-55668-20250606chenyuanfang104:/workspace# python /home/users/chenquanlin/test2.py

======================================================================
精度测试 (PyTorch Full vs Triton Sparse)
======================================================================

[小规模测试]
Accuracy - PyTorch(Full): 9.687500, Triton(Sparse): 9.694802, AbsDiff: 7.302284e-03, RelDiff: 7.537842e-04, Pass: True

[中等规模测试]
Accuracy - PyTorch(Full): 32.000000, Triton(Sparse): 32.123486, AbsDiff: 1.234856e-01, RelDiff: 3.858924e-03, Pass: False

[大规模测试]
Accuracy - PyTorch(Full): 116.000000, Triton(Sparse): 116.129959, AbsDiff: 1.299591e-01, RelDiff: 1.120337e-03, Pass: False

[大规模测试]
Accuracy - PyTorch(Full): 1656.000000, Triton(Sparse): 1657.537842, AbsDiff: 1.537842e+00, RelDiff: 9.286484e-04, Pass: True

[大规模测试]
Accuracy - PyTorch(Full): 744.000000, Triton(Sparse): 751.012573, AbsDiff: 7.012573e+00, RelDiff: 9.425502e-03, Pass: False

[大规模测试]
Accuracy - PyTorch(Full): 2400.000000, Triton(Sparse): 2413.944336, AbsDiff: 1.394434e+01, RelDiff: 5.810140e-03, Pass: False
