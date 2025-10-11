
class Qwen3MoeSparseFusedMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.config = config

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # Use nn.ModuleList for loading compatibility with transformers
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def post_init(self):
        """
        Convert nn.ModuleList experts to fused Qwen3MoeExperts after model loading.
        This method should be called after the model weights are loaded.
        """
        if isinstance(self.experts, nn.ModuleList):
            # Create Qwen3MoeExperts instance
            fused_experts = Qwen3MoeExperts(self.config)
            
            # Copy weights from ModuleList to fused experts
            for expert_idx, expert in enumerate(self.experts):
                fused_experts.gate_proj.data[expert_idx] = expert.gate_proj.weight.data.T.clone()
                fused_experts.up_proj.data[expert_idx] = expert.up_proj.weight.data.T.clone()
                fused_experts.down_proj.data[expert_idx] = expert.down_proj.weight.data.T.clone()
            
            # Replace ModuleList with fused experts
            self.experts = fused_experts
            logger.info("Converted nn.ModuleList experts to fused Qwen3MoeExperts")


    for module in model.modules():
        # Check if the module has post_init method and is a MoE block
        if hasattr(module, "post_init") and hasattr(module, "experts") and isinstance(module.experts, torch.nn.ModuleList):
            module.post_init()
            logger.info_rank0(f"Called post_init for {module.__class__.__name__}")
