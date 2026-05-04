# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsBase,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
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
