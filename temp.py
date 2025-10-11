The expanded size of the tensor (2048) must match the existing size (768) at non-singleton dimension 1.  Target sizes: [768, 2048].  Tensor sizes: [2048, 768]
  File "/home/c30061641/Veomni/VeOmni_Qwen3Moe/veomni/models/transformers/qwen3_moe/modeling_qwen3_moe.py", line 391, in post_init
    fused_experts.gate_proj.data[expert_idx] = expert.gate_proj.weight.data.T.clone()
  File "/home/c30061641/Veomni/VeOmni_Qwen3Moe/veomni/models/auto.py", line 130, in build_foundation_model
    module.post_init()
  File "/home/c30061641/Veomni/VeOmni_Qwen3Moe/tasks/train_torch.py", line 152, in main
    model = build_foundation_model(
  File "/home/c30061641/Veomni/VeOmni_Qwen3Moe/tasks/train_torch.py", line 405, in <module>
    main()
RuntimeError: The expanded size of the tensor (2048) must match the existing size (768) at non-singleton dimension 1.  Target sizes: [768, 2048].  Tensor sizes: [2048, 768]
