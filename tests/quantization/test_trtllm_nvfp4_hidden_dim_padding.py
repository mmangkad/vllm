# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsBase,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_trtllm_fp4_moe_biases,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
    align_fp4_moe_weights_for_fi,
    align_trtllm_fp4_moe_hidden_dim_for_fi,
)


def test_flashinfer_activation_maps_swigluoai_to_swiglu():
    core = pytest.importorskip("flashinfer.fused_moe.core")

    assert activation_to_flashinfer_int(MoEActivation.SWIGLUOAI) == (
        core.ActivationType.Swiglu.value
    )


def test_trtllm_nvfp4_rescales_alpha_dependent_params_once():
    w1_bias = torch.tensor([[2.0, 4.0], [8.0, 16.0]])
    w2_bias = torch.tensor([[10.0, 20.0], [60.0, 80.0]])

    experts = object.__new__(TrtLlmNvFp4ExpertsBase)
    experts._scale_dependent_params_adjusted = False
    experts.quant_config = SimpleNamespace(
        g1_alphas=torch.tensor([2.0, 4.0]),
        g2_alphas=torch.tensor([10.0, 20.0]),
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
    experts.gemm1_beta = torch.tensor([1.0, 1.0])
    experts.gemm1_clamp_limit = torch.tensor([7.0, 7.0])

    experts._adjust_scale_dependent_params()

    torch.testing.assert_close(w1_bias, torch.tensor([[1.0, 2.0], [2.0, 4.0]]))
    torch.testing.assert_close(w2_bias, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(experts.gemm1_beta, torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(experts.gemm1_clamp_limit, torch.tensor([3.5, 1.75]))

    experts._adjust_scale_dependent_params()

    torch.testing.assert_close(w1_bias, torch.tensor([[1.0, 2.0], [2.0, 4.0]]))
    torch.testing.assert_close(w2_bias, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(experts.gemm1_beta, torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(experts.gemm1_clamp_limit, torch.tensor([3.5, 1.75]))


def test_prepare_trtllm_fp4_moe_biases_pads_and_shuffles_like_weights():
    core = pytest.importorskip("flashinfer.fused_moe.core")

    gemm1_bias = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    gemm2_bias = torch.arange(20, dtype=torch.float32).reshape(1, 20)

    out_gemm1_bias, out_gemm2_bias = prepare_trtllm_fp4_moe_biases(
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        hidden_size=32,
        intermediate_size=16,
        num_experts=1,
        is_gated_activation=True,
    )

    assert out_gemm1_bias is not None
    assert out_gemm2_bias is not None
    assert out_gemm1_bias.shape == (1, 32)
    assert out_gemm2_bias.shape == (1, 32)

    cache: dict[tuple[str, torch.Size], torch.Tensor] = {}
    gemm1_permute = core._maybe_get_cached_w3_w1_permute_indices(
        cache,
        gemm1_bias[0].reshape(-1, 1),
        128,
        is_gated_act_gemm=True,
    )
    padded_gemm2_bias = torch.zeros(1, 32, dtype=torch.float32)
    padded_gemm2_bias[:, :20] = gemm2_bias
    gemm2_permute = core.get_w2_permute_indices_with_cache(
        cache,
        padded_gemm2_bias[0].reshape(-1, 1),
        128,
    )

    torch.testing.assert_close(out_gemm1_bias[0], gemm1_bias[0][gemm1_permute])
    torch.testing.assert_close(
        out_gemm2_bias[0],
        padded_gemm2_bias[0][gemm2_permute],
    )


def test_align_fp4_moe_weights_pads_gated_halves_independently():
    num_experts = 1
    hidden_dim = 32
    intermediate = 24
    padded_intermediate = 32

    w13 = torch.zeros(num_experts, 2 * intermediate, hidden_dim // 2, dtype=torch.uint8)
    w13[:, :intermediate, :] = 1
    w13[:, intermediate:, :] = 2
    w13_scale = torch.zeros(
        num_experts,
        2 * intermediate,
        hidden_dim // 16,
        dtype=torch.uint8,
    )
    w13_scale[:, :intermediate, :] = 3
    w13_scale[:, intermediate:, :] = 4
    w2 = torch.ones(num_experts, hidden_dim, intermediate // 2, dtype=torch.uint8)
    w2_scale = torch.ones(
        num_experts, hidden_dim, intermediate // 16, dtype=torch.uint8
    )

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_intermediate = (
        align_fp4_moe_weights_for_fi(
            w13,
            w13_scale,
            w2,
            w2_scale,
            is_act_and_mul=True,
            min_alignment=padded_intermediate,
        )
    )

    assert out_intermediate == padded_intermediate
    assert out_w13.shape == (num_experts, 2 * padded_intermediate, hidden_dim // 2)
    assert out_w13_scale.shape == (
        num_experts,
        2 * padded_intermediate,
        hidden_dim // 16,
    )

    assert torch.all(out_w13[:, :intermediate, :] == 1)
    assert torch.count_nonzero(out_w13[:, intermediate:padded_intermediate, :]) == 0
    assert torch.all(out_w13[:, padded_intermediate:-8, :] == 2)
    assert torch.count_nonzero(out_w13[:, -8:, :]) == 0

    assert torch.all(out_w13_scale[:, :intermediate, :] == 3)
    assert (
        torch.count_nonzero(out_w13_scale[:, intermediate:padded_intermediate, :]) == 0
    )
    assert torch.all(out_w13_scale[:, padded_intermediate:-8, :] == 4)
    assert torch.count_nonzero(out_w13_scale[:, -8:, :]) == 0
    assert out_w2.shape == (num_experts, hidden_dim, padded_intermediate // 2)
    assert out_w2_scale.shape == (
        num_experts,
        hidden_dim,
        padded_intermediate // 16,
    )


def test_align_trtllm_fp4_moe_hidden_dim_noop():
    w13 = torch.arange(2 * 8 * 256, dtype=torch.uint8).reshape(2, 8, 256)
    w13_scale = torch.arange(2 * 8 * 32, dtype=torch.uint8).reshape(2, 8, 32)
    w2 = torch.arange(2 * 512 * 4, dtype=torch.uint8).reshape(2, 512, 4)
    w2_scale = torch.arange(2 * 512 * 1, dtype=torch.uint8).reshape(2, 512, 1)

    out_w13, out_w13_scale, out_w2, out_w2_scale, padded_hidden = (
        align_trtllm_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert padded_hidden == 512
    assert out_w13 is w13
    assert out_w13_scale is w13_scale
    assert out_w2 is w2
    assert out_w2_scale is w2_scale


def test_align_trtllm_fp4_moe_hidden_dim_pads_to_256_multiple():
    hidden_dim = 2688
    padded_hidden_dim = 2816

    w13 = torch.arange(2 * 12 * (hidden_dim // 2), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 2
    )
    w13_scale = torch.arange(2 * 12 * (hidden_dim // 16), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 16
    )

    w2 = torch.arange(2 * hidden_dim * 6, dtype=torch.uint8).reshape(2, hidden_dim, 6)
    w2_scale = torch.arange(2 * hidden_dim * 2, dtype=torch.uint8).reshape(
        2, hidden_dim, 2
    )

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_hidden_dim = (
        align_trtllm_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert out_hidden_dim == padded_hidden_dim
    assert out_w13.shape == (2, 12, padded_hidden_dim // 2)
    assert out_w13_scale.shape == (2, 12, padded_hidden_dim // 16)
    assert out_w2.shape == (2, padded_hidden_dim, 6)
    assert out_w2_scale.shape == (2, padded_hidden_dim, 2)

    torch.testing.assert_close(out_w13[:, :, : hidden_dim // 2], w13)
    torch.testing.assert_close(out_w13_scale[:, :, : hidden_dim // 16], w13_scale)
    torch.testing.assert_close(out_w2[:, :hidden_dim, :], w2)
    torch.testing.assert_close(out_w2_scale[:, :hidden_dim, :], w2_scale)

    assert torch.count_nonzero(out_w13[:, :, hidden_dim // 2 :]) == 0
    assert torch.count_nonzero(out_w13_scale[:, :, hidden_dim // 16 :]) == 0
    assert torch.count_nonzero(out_w2[:, hidden_dim:, :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, hidden_dim:, :]) == 0
