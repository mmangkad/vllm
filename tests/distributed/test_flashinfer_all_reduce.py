# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.device_communicators import flashinfer_all_reduce as fi_ar


class _FakeWorkspace:
    def __init__(self, backend: str, use_multicast: bool = True) -> None:
        self.backend = backend
        self.use_multicast = use_multicast
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


def _reset_workspace_cache() -> None:
    fi_ar._fi_ar_workspace = None
    fi_ar._fi_ar_quant_workspace = None


def test_quant_workspace_uses_mnnvl_backend_on_multinode(monkeypatch):
    _reset_workspace_cache()
    created_backends = []

    def fake_create_workspace(backend, *args, **kwargs):
        created_backends.append(backend)
        return _FakeWorkspace(backend)

    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl")
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 2)
    monkeypatch.setattr(fi_ar, "_create_workspace", fake_create_workspace)
    monkeypatch.setattr(fi_ar, "fi_mnnvl_quant_available", True)

    workspace = fi_ar.get_fi_ar_quant_workspace(
        world_size=16,
        rank=0,
        max_token_num=8,
        hidden_dim=64,
        dtype=torch.bfloat16,
        group=object(),
    )

    assert workspace is not None
    assert workspace.backend == "mnnvl"
    assert created_backends == ["mnnvl"]
    assert (
        fi_ar.get_fi_ar_workspace(
            world_size=16,
            rank=0,
            max_token_num=8,
            hidden_dim=64,
            dtype=torch.bfloat16,
            group=object(),
        )
        is workspace
    )
    _reset_workspace_cache()


def test_quant_workspace_rejects_trtllm_backend_on_multinode(monkeypatch):
    _reset_workspace_cache()
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "trtllm")
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 2)

    try:
        fi_ar.get_fi_ar_quant_workspace(
            world_size=16,
            rank=0,
            max_token_num=8,
            hidden_dim=64,
            dtype=torch.bfloat16,
            group=object(),
        )
    except ValueError as e:
        assert "Please use 'mnnvl' backend" in str(e)
    else:
        raise AssertionError("Expected TRT-LLM quant workspace to fail on multi-node")
    finally:
        _reset_workspace_cache()


def test_quant_workspace_reuses_existing_mnnvl_workspace(monkeypatch):
    _reset_workspace_cache()
    workspace = _FakeWorkspace("mnnvl")

    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl")
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 1)
    monkeypatch.setattr(fi_ar, "fi_mnnvl_quant_available", True)
    monkeypatch.setattr(
        fi_ar,
        "_create_workspace",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("workspace should have been reused")
        ),
    )
    fi_ar._fi_ar_workspace = workspace

    assert (
        fi_ar.get_fi_ar_quant_workspace(
            world_size=2,
            rank=0,
            max_token_num=8,
            hidden_dim=64,
            dtype=torch.bfloat16,
            group=object(),
        )
        is workspace
    )
    _reset_workspace_cache()


def test_quant_workspace_falls_back_to_trtllm_when_mnnvl_quant_unavailable(
    monkeypatch,
):
    _reset_workspace_cache()
    created_backends = []

    def fake_create_workspace(backend, *args, **kwargs):
        created_backends.append(backend)
        return _FakeWorkspace(backend)

    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl")
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 1)
    monkeypatch.setattr(fi_ar, "fi_mnnvl_quant_available", False)
    monkeypatch.setattr(fi_ar, "_create_workspace", fake_create_workspace)

    workspace = fi_ar.get_fi_ar_quant_workspace(
        world_size=2,
        rank=0,
        max_token_num=8,
        hidden_dim=64,
        dtype=torch.bfloat16,
        group=object(),
    )

    assert workspace is not None
    assert workspace.backend == "trtllm"
    assert created_backends == ["trtllm"]
    _reset_workspace_cache()


def test_quant_workspace_disabled_on_multinode_when_mnnvl_quant_unavailable(
    monkeypatch,
):
    _reset_workspace_cache()
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl")
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 2)
    monkeypatch.setattr(fi_ar, "fi_mnnvl_quant_available", False)
    monkeypatch.setattr(
        fi_ar,
        "_create_workspace",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("workspace should not be created")
        ),
    )

    assert (
        fi_ar.get_fi_ar_quant_workspace(
            world_size=16,
            rank=0,
            max_token_num=8,
            hidden_dim=64,
            dtype=torch.bfloat16,
            group=object(),
        )
        is None
    )
    _reset_workspace_cache()


def test_auto_backend_uses_mnnvl_when_unicast_requested(monkeypatch):
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "auto")
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_USE_MULTICAST", "0")
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 1)

    assert fi_ar._resolve_fi_ar_backend() == "mnnvl"


def test_workspace_creation_passes_unicast_transport(monkeypatch):
    _reset_workspace_cache()
    created_kwargs = []

    def fake_create_workspace(**kwargs):
        created_kwargs.append(kwargs)
        return _FakeWorkspace(kwargs["backend"], kwargs["use_multicast"])

    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl")
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_USE_MULTICAST", "0")
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 1)
    monkeypatch.setattr(fi_ar, "fi_mnnvl_quant_available", True)
    monkeypatch.setattr(
        fi_ar.flashinfer_comm,
        "create_allreduce_fusion_workspace",
        fake_create_workspace,
    )
    monkeypatch.setattr(fi_ar, "TorchDistBackend", lambda group: object())

    workspace = fi_ar.get_fi_ar_workspace(
        world_size=8,
        rank=0,
        max_token_num=8,
        hidden_dim=64,
        dtype=torch.bfloat16,
        group=object(),
    )

    assert workspace is not None
    assert workspace.backend == "mnnvl"
    assert workspace.use_multicast is False
    assert len(created_kwargs) == 1
    assert created_kwargs[0]["backend"] == "mnnvl"
    assert created_kwargs[0]["world_size"] == 8
    assert created_kwargs[0]["rank"] == 0
    assert created_kwargs[0]["max_token_num"] == 8
    assert created_kwargs[0]["hidden_dim"] == 64
    assert created_kwargs[0]["dtype"] is torch.bfloat16
    assert created_kwargs[0]["gpus_per_node"] == 8
    assert created_kwargs[0]["use_multicast"] is False
    _reset_workspace_cache()


def test_mnnvl_unicast_rejected_on_multinode(monkeypatch):
    _reset_workspace_cache()
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", "mnnvl")
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_USE_MULTICAST", "0")
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 2)

    try:
        fi_ar.get_fi_ar_workspace(
            world_size=16,
            rank=0,
            max_token_num=8,
            hidden_dim=64,
            dtype=torch.bfloat16,
            group=object(),
        )
    except ValueError as e:
        assert "single-node only" in str(e)
    else:
        raise AssertionError("Expected MNNVL unicast workspace to fail on multi-node")
    finally:
        _reset_workspace_cache()
