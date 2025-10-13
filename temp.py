[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/c30061641/Veomni/VeOmni_Qwen3Moe/tasks/train_torch.py", line 405, in <module>
[rank1]:     main()
[rank1]:   File "/home/c30061641/Veomni/VeOmni_Qwen3Moe/tasks/train_torch.py", line 299, in main
[rank1]:     loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss.mean() / len(micro_batches)
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1857, in _call_impl
[rank1]:     return inner()
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in inner
[rank1]:     args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_state.py", line 62, in fsdp_hook_wrapper
[rank1]:     return torch._dynamo.disable(func, recursive=True)(*args, **kwargs)
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 838, in _fn
[rank1]:     return fn(*args, **kwargs)
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_state.py", line 222, in _pre_forward
[rank1]:     args, kwargs = self._root_pre_forward(module, args, kwargs)
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch_npu/distributed/fsdp/_add_fsdp_patch.py", line 95, in _patched_root_pre_forward
[rank1]:     self._lazy_init()
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_state.py", line 178, in _lazy_init
[rank1]:     state._fsdp_param_group.lazy_init()
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py", line 244, in lazy_init
[rank1]:     self._validate_no_meta_params()
[rank1]:   File "/root/anaconda3/envs/veomni/lib/python3.10/site-packages/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py", line 667, in _validate_no_meta_params
[rank1]:     raise RuntimeError(
[rank1]: RuntimeError: FSDP parameters should be materialized from meta device before training, but the following were still on meta device: ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight']
