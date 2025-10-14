
class EPGroupGemm_eager(torch.autograd.Function):
    """
    EPGroupGemm 的 eager 实现，使用 PyTorch 原生 API，数学上与 EPGroupGemm 等价。
    不使用 group_gemm_same_nk 或 group_gemm_same_mn，而是通过切片和循环实现。
    """

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        # permute_tokens: [tokens, hidden_dim]
        # cumsum: [local_experts] - 每个专家处理的 token 累积数量
        # fc1_1_weight: [num_experts, intermediate_dim, hidden_dim]
        # fc1_2_weight: [num_experts, intermediate_dim, hidden_dim]
        # fc2_weight: [num_experts, hidden_dim, intermediate_dim]

        num_experts = fc1_1_weight.shape[0]
        total_tokens = permute_tokens.shape[0]
        hidden_dim = permute_tokens.shape[1]
        intermediate_dim = fc1_1_weight.shape[1]

        # 初始化输出
        fc1_1_output = torch.zeros(total_tokens, intermediate_dim, dtype=permute_tokens.dtype, device=permute_tokens.device)
        fc1_2_output = torch.zeros(total_tokens, intermediate_dim, dtype=permute_tokens.dtype, device=permute_tokens.device)
        fc2_output = torch.zeros(total_tokens, hidden_dim, dtype=permute_tokens.dtype, device=permute_tokens.device)

        # 将 cumsum 移到 CPU 以便索引
        cumsum_cpu = cumsum.cpu()

        # 对每个专家分别处理
        start_idx = 0
        for expert_id in range(num_experts):
            end_idx = int(cumsum_cpu[expert_id].item())
            if start_idx >= end_idx:
                continue

            # 获取该专家处理的 token
            expert_tokens = permute_tokens[start_idx:end_idx]  # [expert_tokens, hidden_dim]

            # 获取该专家的权重
            expert_fc1_1_weight = fc1_1_weight[expert_id]  # [intermediate_dim, hidden_dim]
            expert_fc1_2_weight = fc1_2_weight[expert_id]  # [intermediate_dim, hidden_dim]
            expert_fc2_weight = fc2_weight[expert_id]  # [hidden_dim, intermediate_dim]

            # compute linear layer fc1-1: expert_tokens @ fc1_1_weight.T
            fc1_1_output[start_idx:end_idx] = torch.matmul(expert_tokens, expert_fc1_1_weight.t())

            # compute linear layer fc1-2: expert_tokens @ fc1_2_weight.T
            fc1_2_output[start_idx:end_idx] = torch.matmul(expert_tokens, expert_fc1_2_weight.t())

            start_idx = end_idx

        # compute the activation of linear layer fc1-1
        fc1_1_activation = torch.nn.functional.silu(fc1_1_output)

        # compute final result of linear layer fc1 (SwiGLU)
        fc1_output = fc1_1_activation * fc1_2_output

        # compute linear layer fc2
        start_idx = 0
        for expert_id in range(num_experts):
            end_idx = int(cumsum_cpu[expert_id].item())
            if start_idx >= end_idx:
                continue

            # 获取该专家处理的 fc1 输出
            expert_fc1_output = fc1_output[start_idx:end_idx]  # [expert_tokens, intermediate_dim]

            # 获取该专家的 fc2 权重
            expert_fc2_weight = fc2_weight[expert_id]  # [hidden_dim, intermediate_dim]

            # compute linear layer fc2: expert_fc1_output @ fc2_weight.T
            fc2_output[start_idx:end_idx] = torch.matmul(expert_fc1_output, expert_fc2_weight.t())

            start_idx = end_idx

        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [tokens, hidden_dim]
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
        ) = ctx.saved_tensors

        num_experts = fc1_1_weight.shape[0]
        total_tokens = permute_tokens.shape[0]
        hidden_dim = permute_tokens.shape[1]
        intermediate_dim = fc1_1_weight.shape[1]

        # 初始化梯度
        grad_permute_tokens = torch.zeros_like(permute_tokens)
        grad_fc1_1_weight = torch.zeros_like(fc1_1_weight) if fc1_1_weight.requires_grad else None
        grad_fc1_2_weight = torch.zeros_like(fc1_2_weight) if fc1_2_weight.requires_grad else None
        grad_fc2_weight = torch.zeros_like(fc2_weight) if fc2_weight.requires_grad else None

        grad_fc1_output = torch.zeros(total_tokens, intermediate_dim, dtype=grad_output.dtype, device=grad_output.device)

        # 将 cumsum 移到 CPU 以便索引
        cumsum_cpu = cumsum.cpu()

        # recompute for backward
        fc1_1_activation = torch.nn.functional.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output

        # backward through fc2
        start_idx = 0
        for expert_id in range(num_experts):
            end_idx = int(cumsum_cpu[expert_id].item())
            if start_idx >= end_idx:
                continue

            # 获取该专家的梯度
            expert_grad_output = grad_output[start_idx:end_idx]  # [expert_tokens, hidden_dim]
            expert_fc1_output = fc1_output[start_idx:end_idx]  # [expert_tokens, intermediate_dim]
            expert_fc2_weight = fc2_weight[expert_id]  # [hidden_dim, intermediate_dim]

            # dgrad fc2: grad_output @ fc2_weight (不转置)
            grad_fc1_output[start_idx:end_idx] = torch.matmul(expert_grad_output, expert_fc2_weight)

            # wgrad fc2: grad_output.T @ fc1_output
            if grad_fc2_weight is not None:
                grad_fc2_weight[expert_id] = torch.matmul(expert_grad_output.t(), expert_fc1_output)

            start_idx = end_idx

        # backward through SwiGLU
        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output

        # backward through silu
        # silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        # PyTorch 提供了 silu_backward
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # backward through fc1-1 and fc1-2
        start_idx = 0
        for expert_id in range(num_experts):
            end_idx = int(cumsum_cpu[expert_id].item())
            if start_idx >= end_idx:
                continue

            expert_tokens = permute_tokens[start_idx:end_idx]  # [expert_tokens, hidden_dim]
            expert_grad_fc1_1_output = grad_fc1_1_output[start_idx:end_idx]  # [expert_tokens, intermediate_dim]
            expert_grad_fc1_2_output = grad_fc1_2_output[start_idx:end_idx]  # [expert_tokens, intermediate_dim]
            expert_fc1_1_weight = fc1_1_weight[expert_id]  # [intermediate_dim, hidden_dim]
            expert_fc1_2_weight = fc1_2_weight[expert_id]  # [intermediate_dim, hidden_dim]

            # dgrad fc1-1: grad_fc1_1_output @ fc1_1_weight (不转置)
            grad_scatter_output_1 = torch.matmul(expert_grad_fc1_1_output, expert_fc1_1_weight)

            # wgrad fc1-1: grad_fc1_1_output.T @ expert_tokens
            if grad_fc1_1_weight is not None:
                grad_fc1_1_weight[expert_id] = torch.matmul(expert_grad_fc1_1_output.t(), expert_tokens)

            # dgrad fc1-2: grad_fc1_2_output @ fc1_2_weight (不转置)
            grad_scatter_output_2 = torch.matmul(expert_grad_fc1_2_output, expert_fc1_2_weight)

            # wgrad fc1-2: grad_fc1_2_output.T @ expert_tokens
            if grad_fc1_2_weight is not None:
                grad_fc1_2_weight[expert_id] = torch.matmul(expert_grad_fc1_2_output.t(), expert_tokens)

            # 累加输入梯度
            grad_permute_tokens[start_idx:end_idx] = grad_scatter_output_1 + grad_scatter_output_2

            start_idx = end_idx

        return (
            grad_permute_tokens,  # permute_tokens
            None,  # cumsum
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )
