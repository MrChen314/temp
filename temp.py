model:
  model_path: /home/c30061641/data/model/Qwen3-30B-A3B-Instruct-2507
  attn_implementation: flash_attention_2
  moe_implementation: fused  # fused

data:
  train_path: /home/c30061641/data/dataset/fineweb/sample/10BT
  train_size: 1000000000000
  dataloader_type: native
  datasets_type: iterable
  data_type: plaintext
  max_seq_len: 8192
  text_keys: text
  drop_last: true

train:
  output_dir: Qwen3-30B-A3B_CT
  data_parallel_mode: fsdp2
  ulysses_parallel_size: 1
  expert_parallel_size: 4
  global_batch_size: 32
  micro_batch_size: 1
  rmpad: false
  rmpad_with_pos_ids: true
  bsz_warmup_ratio: 0.007
  dyn_bsz_margin: 0
  dyn_bsz_buffer_size: 200
  optimizer: adamw
  lr: 3.0e-4
  lr_warmup_ratio: 0.007
  lr_decay_style: constant
  lr_decay_ratio: 1.0
  weight_decay: 0.01
  max_grad_norm: 1.0
  enable_mixed_precision: false
  enable_gradient_checkpointing: true
  enable_full_shard: true
  enable_fsdp_offload: false
  enable_activation_offload: false
  init_device: meta
  enable_full_determinism: false
  empty_cache_steps: 500
  ckpt_manager: dcp
  load_checkpoint_path: ""
  max_steps: 20
  save_steps: 1000000
  save_hf_weights: true
  use_wandb: false
  wandb_project: Qwen3-30B-A3B
  wandb_name: Qwen3-30B-A3B-CT
