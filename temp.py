model:
  model_path: /home/c30061641/data/model/Qwen2.5-VL-7B-Instruct

data:
  dataloader_type: native
  datasets_type: mapping
  train_path: /home/c30061641/data/dataset/ShareGPT4V/subset
  chat_template: qwen2_5vl
  max_seq_len: 2048
  train_size: 80000000
  source_name: sharegpt4v_pretrain

train:
  output_dir: qwen2.5vl_sft
  data_parallel_mode: fsdp2
  wandb_project: qwen2_5vl
  wandb_name: qwen2_5vl
  rmpad: false
  rmpad_with_pos_ids: true
  ulysses_parallel_size: 1
  freeze_vit: false
  lr: 1.0e-5
  lr_decay_style: constant
  num_train_epochs: 1
  micro_batch_size: 1
  global_batch_size: 32
  max_steps: 5000
  use_wandb: false
  ckpt_manager: dcp






def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attn_output = torch_npu.npu_fusion_attention(
                q, k, v, q.shape[1],
                pse=None,
                padding_mask=None,
                atten_mask=None,
                scale=1.0 / math.sqrt(q.shape[-1]),
                keep_prob=1,
                input_layout='TND',
                actual_seq_qlen=cu_seqlens.tolist()[1:],
                actual_seq_kvlen=cu_seqlens.tolist()[1:],
                pre_tockens=2147483647,
                next_tockens=2147483647,
                sparse_mode=0)[0]

        # attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        # q = q.transpose(0, 1)
        # k = k.transpose(0, 1)
        # v = v.transpose(0, 1)
        # attn_output = F.scaled_dot_product_attention(
        #     q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
        # )
        # attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output
