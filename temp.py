# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch

from ..distributed.moe import EPGroupGemm, preprocess, token_pre_all2all, tokens_post_all2all
from ..distributed.parallel_state import get_parallel_state
from ..utils.import_utils import is_fused_moe_available, is_seed_kernels_available


if is_fused_moe_available():
    from .group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk
    from .group_gemm.kernel.moe import expert_histogram, moe_gather, moe_scatter


class FusedMoeExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    ):
        # MOE Step 3: dispatch input tokens to the experts
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        # MOE Step 3-1: compute the token num for each expert
        # splits shape (num_experts)
        splits = expert_histogram(expert_index, num_experts)

        # MOE Step 3-2: compute the each token's index in result
        # scatter_index shape (batch_size * sequence_len, topk)
        # TODO(wenyawei): opt it
        scatter_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)

        # MOE Step 3-3: compute the result, select tokens by scatter_index, and put them together
        # scatter_output shape (batch_size * sequence_len * topk, hidden_size)
        scatter_output = moe_scatter(hidden_states, scatter_index)

        # MOE Step 4: compute linear layer 1-1
        # Not consistent.
        cumsum_t = torch.cumsum(splits, dim=0)
        fc1_1_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 6: compute linear layer 1-2
        # fc1_2_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_2_output = group_gemm_same_nk(
            a=scatter_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 5: compute the actication of linear layer 1-1
        # TODO(wenyawei): act function
        # fc1_1_activation shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7: compute final result of linear layer 1
        fc1_activation = fc1_1_activation * fc1_2_output

        # MOE Step 8: compute the the weighted linear layer 1 result
        # MOE Step 8-1: compute scattered_gate_weight, shape is (batch_size * sequence_len * topk)
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        # MOE Step 8-2: multiply activate with scattered_gate_weight
        # fc1_weighted_output shape is (batch_size * sequence_len * topk, ffn_dim)
        fc1_weighted_output = fc1_activation * scattered_gate_weight

        # MOE Step 9: compute linear layer 2
        # result shape is (batch_size * sequence_len * topk, hidden_size)
        fc2_output = group_gemm_same_nk(
            a=fc1_weighted_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        # MOE Step 10: gather the final token result by averate the the topk token results
        expert_output = moe_gather(fc2_output, scatter_index)

        # reshape the output with input shape
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.save_for_backward(
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
        ) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        # MOE Step 10
        grad_fc2_output = moe_scatter(grad_output, scatter_index)

        # MOE Step 9
        # grad_fc1_weighted_output = torch.empty_like(fc1_weighted_output)

        # dgrad
        grad_fc1_weighted_output = group_gemm_same_nk(
            a=grad_fc2_output,
            b=fc2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(
                a=grad_fc2_output,
                b=fc1_weighted_output,
                c=grad_fc2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 8
        # MOE Step 8-2
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight

        # MOE Step 8-1
        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        # recompute during backward
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        # MOE Step 7
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        # MOE Step 6
        # grad_scatter_output_2 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_2 = group_gemm_same_nk(
            a=grad_fc1_2_output,
            b=fc1_2_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(
                a=grad_fc1_2_output,
                b=scatter_output,
                c=grad_fc1_2_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 5
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        # MOE Step 4
        # grad_scatter_output_1 = torch.empty_like(scatter_output)

        # dgrad
        grad_scatter_output_1 = group_gemm_same_nk(
            a=grad_fc1_1_output,
            b=fc1_1_weight,
            cumsum_M=cumsum_t,
            max_M=grad_output.shape[0],
            transpose_b=False,
        )

        # wgrad
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(
                a=grad_fc1_1_output,
                b=scatter_output,
                c=grad_fc1_1_weight,
                cumsum_K=cumsum_t,
                max_K=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        # MOE Step 3
        # MOE Step 3-3
        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)

        # MOE Step 3-2: no grad
        # MOE Step 3-1: no grad

        # reshape the result with input shape
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,  # num_experts
            grad_gate_weight,  # gate_weights
            None,  # expert_index
            grad_hidden_states,  # hidden_states
            grad_fc1_1_weight,  # fc1_1_weight
            grad_fc1_2_weight,  # fc1_2_weight
            grad_fc2_weight,  # fc2_weight
        )


def fused_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
):
    if get_parallel_state().ep_enabled:
        # use seed kernels
        if is_seed_kernels_available():
            from seed_kernels.transformers.functional import seed_fused_moe

            seed_ep_implementation = os.getenv("SEED_EP_IMPLEMENTATION")
            if seed_ep_implementation is not None:
                assert seed_ep_implementation in [
                    "bumi",
                    "flux",
                    "ring_ep",
                    "local_ep",
                    "chunked_overlap",
                    "agrs",
                    "flux_gpu",
                ]

            final_hidden_states = seed_fused_moe(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
                ep_group=get_parallel_state().ep_group,
                ep_implementation=seed_ep_implementation if seed_ep_implementation is not None else "bumi",
            )
        # use open source
        else:
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
            # preprocess, permute token for ep
            input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
                preprocess(
                    expert_mask=expert_mask,
                    num_experts=num_experts,
                    ep_group=get_parallel_state().ep_group,
                )
            )
            permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
                hidden_states=hidden_states,
                expert_mask=expert_mask,
                num_experts=num_experts,
                input_splits=input_splits,
                output_splits=output_splits,
                num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
                ep_group=get_parallel_state().ep_group,
            )

            final_permute_tokens = torch.zeros(
                (permute_tokens.shape),
                dtype=permute_tokens.dtype,
                device=permute_tokens.device,
            )

            cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, dim=0).to(permute_tokens.device)

            final_permute_tokens = EPGroupGemm_eager.apply(
                permute_tokens,
                cumsum,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
            )

            # unpermute with routing_weight
            final_hidden_states = tokens_post_all2all(
                expert_outputs=final_permute_tokens,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                num_experts=num_experts,
                input_splits=input_splits,
                output_splits=output_splits,
                num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
                routing_map=routing_map,
                local_input_permutation_mapping=local_input_permutation_mapping,
                org_hidden_states_shape=org_hidden_states_shape,
                ep_group=get_parallel_state().ep_group,
            )
    else:
        routing_weights = routing_weights.bfloat16()
        hidden_states = hidden_states.bfloat16()
        if is_seed_kernels_available():
            from seed_kernels.transformers.functional import seed_fused_moe

            final_hidden_states = seed_fused_moe(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
            )
        # use open source
        else:
            final_hidden_states = FusedMoeExpertFunction.apply(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
            )

    return final_hidden_states


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
