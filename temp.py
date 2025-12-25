root@gajl-55668-20250606chenyuanfang104:/workspace# python /home/users/chenquanlin/test.py

==========================================================================================
精度测试 (PyTorch Full vs Triton Sparse)
==========================================================================================
Traceback (most recent call last):
  File "/home/users/chenquanlin/test.py", line 679, in <module>
    test_full_accuracy(accuracy_configs)
  File "/home/users/chenquanlin/test.py", line 523, in test_full_accuracy
    result = run_single_accuracy_test(config)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/test.py", line 493, in run_single_accuracy_test
    tri = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, 
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/test.py", line 328, in compute_index_loss_sparse
    _sparse_attn_loss_fused_kernel[grid](
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
triton.compiler.errors.CompilationError: at 69:8:
    for h_block in range(num_head_blocks):
        h_start = h_block * BLOCK_H
        h_mask = (h_start + offs_h) < num_heads

        # -----------------------------------------------------------------
        # Pass 1: 计算全局 max 和 sum (Online Softmax) - 每个 head 独立
        # -----------------------------------------------------------------
        # m_global, l_global: [BLOCK_H] - 每个 head 一个值
        m_global = tl.full([BLOCK_H], NEG_INF, dtype=tl.bfloat16)
        l_global = tl.zeros([BLOCK_H], dtype=tl.bfloat16)

        for tk_idx in range(num_topk_blocks):
        ^
AssertionError("Loop-carried variable m_global has initial type <['16'], bf16> but is re-assigned to <['16'], fp32> in loop! Please make sure that the type stays consistent.")
