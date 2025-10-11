    def post_init(self):
        """
        Convert nn.ModuleList experts to fused Qwen3MoeExperts after model loading.
        This method should be called after the model weights are loaded.
        """
        if isinstance(self.experts, nn.ModuleList):
            # Check if experts are on meta device (no actual data yet)
            if len(self.experts) > 0 and self.experts[0].gate_proj.weight.device.type == "meta":
                logger.info("Skipping post_init for Qwen3MoeSparseMoeBlock: experts are still on meta device")
                return
            
            # Create Qwen3MoeExperts instance
            fused_experts = Qwen3MoeExperts(self.config)
            
            # Get device and dtype from the first expert
            device = self.experts[0].gate_proj.weight.device
            dtype = self.experts[0].gate_proj.weight.dtype
            
            # Copy weights from ModuleList to fused experts in batch for efficiency
            # Linear.weight shape: [out_features, in_features]
            # gate_proj: [intermediate_size, hidden_dim]
            # up_proj: [intermediate_size, hidden_dim]
            # down_proj: [hidden_dim, intermediate_size]
            # Stack all expert weights at once
            gate_proj_weights = torch.stack([expert.gate_proj.weight.data for expert in self.experts], dim=0)
            up_proj_weights = torch.stack([expert.up_proj.weight.data for expert in self.experts], dim=0)
            down_proj_weights = torch.stack([expert.down_proj.weight.data for expert in self.experts], dim=0)
            
            # Move to target device and copy
            fused_experts.gate_proj.data = gate_proj_weights.to(device=device, dtype=dtype)
            fused_experts.up_proj.data = up_proj_weights.to(device=device, dtype=dtype)
            fused_experts.down_proj.data = down_proj_weights.to(device=device, dtype=dtype)
            
            # Replace ModuleList with fused experts
            self.experts = fused_experts
            logger.info("Converted nn.ModuleList experts to fused Qwen3MoeExperts")
