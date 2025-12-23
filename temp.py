root@gajl-bbc-onlinec-com-1585209:/home/users/chenquanlin/workspace/a0p6b_dsa_stage2# python test.py 

======================================================================
精度测试 - V2 版本 (简化版)
======================================================================

[小规模测试]
Traceback (most recent call last):
  File "/home/users/chenquanlin/workspace/a0p6b_dsa_stage2/test.py", line 611, in <module>
    test_full_accuracy(batch_size=1, num_heads=4, seq_len=64, head_dim=32, topk=16, use_v2=True)
  File "/home/users/chenquanlin/workspace/a0p6b_dsa_stage2/test.py", line 524, in test_full_accuracy
    tri = compute_index_loss_sparse(query, key, index_score, indices, scaling, use_v2=use_v2)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/a0p6b_dsa_stage2/test.py", line 421, in compute_index_loss_sparse
    _sparse_attn_loss_fused_kernel_v2[grid](
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 347, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/runtime/jit.py", line 569, in run
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 278, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/compiler/compiler.py", line 81, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 45:8:
    indices = tl.load(idx_base + offs_topk * stride_ik).to(tl.int64)
    causal_mask = indices > pid_row

    attn_sum = tl.zeros([topk], dtype=tl.float32)

    for h in tl.static_range(num_heads):
        q_base = q_batch_base + h * stride_qh + pid_row * stride_qs

        # 计算 QK^T - 小块 BLOCK_D
        qk = tl.zeros([topk], dtype=tl.float32)

        num_d_blocks: tl.constexpr = (head_dim + BLOCK_D - 1) // BLOCK_D
        ^
ValueError('num_d_blocks is already defined. constexpr cannot be reassigned.')
