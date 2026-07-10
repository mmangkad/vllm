# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


class FlashAttnMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLASparseMetadataBuilder"]:
        return FlashAttnMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[SparseMLAAttentionImpl[Any]]:
        return FlashAttnMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype not in (None, "auto", "float16", "bfloat16"):
            return (
                "FlashAttention MLA Sparse currently supports only FP16/BF16 KV cache"
            )

        if not flash_attn_supports_mla():
            return "FlashAttention MLA not supported on this device"

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            if vllm_config.parallel_config.decode_context_parallel_size > 1:
                return "FlashAttention MLA Sparse does not support DCP for now"

            hf_config = vllm_config.model_config.hf_config
            if not hasattr(hf_config, "index_topk"):
                return "FlashAttention MLA Sparse requires model with index_topk"
            index_topk = hf_config.index_topk
            if index_topk <= 0 or index_topk % 128 != 0:
                return (
                    "FlashAttention MLA Sparse requires index_topk to be a "
                    "positive multiple of 128"
                )
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


@dataclass
class FlashAttnMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    cu_seqlens_q: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048


class FlashAttnMLASparseMetadataBuilder(
    AttentionMetadataBuilder[FlashAttnMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        self.device = device

        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )
        self.cu_seqlens_q_buffer = torch.arange(
            max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self._reserve_conversion_workspace(max_num_batched_tokens)

    def _reserve_conversion_workspace(self, max_num_batched_tokens: int) -> None:
        """Reserve the largest conversion buffers before workspace locking."""
        if not is_workspace_manager_initialized():
            return
        current_workspace_manager().reserve_for_all_ubatches(
            ((max_num_batched_tokens, self.topk_tokens), torch.int32),
            ((max_num_batched_tokens,), torch.int32),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttnMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        req_id_per_token = cm.token_to_req_indices(self.req_id_per_token_buffer)

        return FlashAttnMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=req_id_per_token,
            cu_seqlens_q=self.cu_seqlens_q_buffer[: num_tokens + 1],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class FlashAttnMLASparseImpl(SparseMLAAttentionImpl[FlashAttnMLASparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        assert flash_attn_supports_mla(), (
            "FlashAttnMLASparseImpl requires FA3 MLA support"
        )
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl does not support alibi, sliding window, "
                "or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl currently supports only FP16/BF16 KV cache."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FlashAttnMLASparseImpl only supports decoder attention."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        assert self.topk_indices_buffer is not None, (
            "Indexer or topk_indices_buffer required for sparse MLA"
        )
        self.supports_quant_query_input = False
        self.dcp_world_size = -1
        self.q_pad_num_heads = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl expects split (q_nope, q_rope) input."
            )
        q_nope, q_rope = q
        num_actual_toks = q_rope.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        converted_indices_buffer = None
        valid_counts_buffer = None
        if is_workspace_manager_initialized():
            converted_indices_buffer, valid_counts_buffer = (
                current_workspace_manager().get_simultaneous(
                    (tuple(topk_indices.shape), torch.int32),
                    ((num_actual_toks,), torch.int32),
                )
            )
        topk_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
            out=converted_indices_buffer,
            valid_counts_out=valid_counts_buffer,
        )

        kv_cache = kv_c_and_k_pe_cache.view(
            -1, attn_metadata.block_size, self.head_size
        )
        k_cache = kv_cache[:, :, self.kv_lora_rank :].view(
            -1, 1, 1, self.qk_rope_head_dim
        )
        v_cache = kv_cache[:, :, : self.kv_lora_rank].view(-1, 1, 1, self.kv_lora_rank)

        out = flash_attn_varlen_func(
            q=q_rope,
            k=k_cache,
            v=v_cache,
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=attn_metadata.cu_seqlens_q[: num_actual_toks + 1],
            max_seqlen_k=topk_indices.shape[1],
            seqused_k=valid_counts,
            block_table=topk_indices,
            softmax_scale=self.scale,
            # The indexer already bounds each top-k set causally. With q_len=1,
            # FA3's causal mask admits the same set, so use the cheaper variant.
            causal=False,
            fa_version=3,
        )
        return out, None
