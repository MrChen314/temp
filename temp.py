======================================================================
精度测试 (PyTorch Full vs Triton Sparse)
======================================================================

[小规模测试]
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 34, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/core.py", line 1814, in dot
    return semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype, _builder)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/triton/language/semantic.py", line 1570, in dot
    assert lhs.shape[-2].value >= min_dot_size[0] and lhs.shape[-1].value >= min_dot_size[2] \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Input shapes should have M >= 16, N >= 16 and K >= 16

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/users/chenquanlin/workspace/../test.py", line 570, in <module>
    test_full_accuracy(batch_size=1, num_heads=4, chunk_size=32, seq_len=64, head_dim=32, topk=16)
  File "/home/users/chenquanlin/workspace/../test.py", line 432, in test_full_accuracy
    tri = compute_index_loss_sparse(query, key, index_score_sparse, topk_indices, scaling, chunk_offset=chunk_offset)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/chenquanlin/workspace/../test.py", line 280, in compute_index_loss_sparse
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
triton.compiler.errors.CompilationError: at 82:22:
            # 计算 QK
            qk = tl.zeros([BLOCK_TOPK, 1], dtype=tl.float32)
            num_d_blocks = tl.cdiv(head_dim, BLOCK_D)
            for d_idx in range(num_d_blocks):
                d_start = d_idx * BLOCK_D
                offs_d = d_start + tl.arange(0, BLOCK_D)
                d_mask = offs_d < head_dim

                q = tl.load(q_base + offs_d * stride_qd, mask=d_mask, other=0.0)  # [BLOCK_D]
                k_ptrs = k_batch_base + indices_block[:, None] * stride_ks + offs_d[None, :] * stride_kd
                k_gathered = tl.load(k_ptrs, mask=tk_mask[:, None] & d_mask[None, :], other=0.0)  # [BLOCK_TOPK, BLOCK_D]
                qk += tl.dot(k_gathered, q[:, None])  # [BLOCK_TOPK, 1]
