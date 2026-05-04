# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM,
    _expand_gpt_oss_nvfp4_scale,
    _reshape_gpt_oss_nvfp4_weight,
)


def test_gpt_oss_nvfp4_mapper_handles_modelopt_moe_names():
    mapper = GptOssForCausalLM.hf_to_vllm_mapper

    assert (
        mapper._map_name("model.layers.0.mlp.experts.gate_up_proj_weight_scale_2")
        == "model.layers.0.mlp.experts.w13_weight_scale_2"
    )
    assert (
        mapper._map_name("model.layers.0.mlp.experts.down_proj_input_scale")
        == "model.layers.0.mlp.experts.w2_input_scale"
    )
    assert (
        mapper._map_name("model.layers.0.mlp.experts.gate_up_proj.weight_scale_2")
        == "model.layers.0.mlp.experts.w13_weight_scale_2"
    )


def test_gpt_oss_nvfp4_gate_up_weight_accepts_checkpoint_orientation():
    num_experts = 2
    hidden_size = 16
    intermediate_size = 8
    checkpoint_weight = torch.arange(
        num_experts * (hidden_size // 2) * (2 * intermediate_size),
        dtype=torch.uint8,
    ).reshape(num_experts, hidden_size // 2, 2 * intermediate_size)

    reshaped = _reshape_gpt_oss_nvfp4_weight(
        checkpoint_weight,
        weight_name="w13_weight",
        is_w13=True,
        is_scale=False,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        group_size=16,
        expert_id=None,
    )

    assert reshaped.shape == (num_experts, 2 * intermediate_size, hidden_size // 2)
    assert torch.equal(reshaped, checkpoint_weight.transpose(1, 2).contiguous())


def test_gpt_oss_nvfp4_down_weight_accepts_checkpoint_orientation():
    num_experts = 2
    hidden_size = 16
    intermediate_size = 8
    checkpoint_weight = torch.arange(
        num_experts * (intermediate_size // 2) * hidden_size,
        dtype=torch.uint8,
    ).reshape(num_experts, intermediate_size // 2, hidden_size)

    reshaped = _reshape_gpt_oss_nvfp4_weight(
        checkpoint_weight,
        weight_name="w2_weight",
        is_w13=False,
        is_scale=False,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        group_size=16,
        expert_id=None,
    )

    assert reshaped.shape == (num_experts, hidden_size, intermediate_size // 2)
    assert torch.equal(reshaped, checkpoint_weight.transpose(1, 2).contiguous())


def test_gpt_oss_nvfp4_scale_expansion():
    per_expert = torch.tensor([1.0, 2.0, 3.0])
    expanded = _expand_gpt_oss_nvfp4_scale(
        per_expert,
        torch.Size((3, 2)),
        "w13_weight_scale_2",
    )
    assert torch.equal(
        expanded,
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )

    per_shard = torch.tensor([4.0, 5.0])
    expanded = _expand_gpt_oss_nvfp4_scale(
        per_shard,
        torch.Size((3, 2)),
        "w13_input_scale",
    )
    assert torch.equal(
        expanded,
        torch.tensor([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]]),
    )
