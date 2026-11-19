# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from sglang.srt.environ import envs


class PrefillQWithStoredKV(NamedTuple):
    query: torch.Tensor


def _canonical_metadata(strategy, batch, latent):
    metadata = batch.attn_cp_metadata
    if metadata is None or getattr(metadata, "gather_index", None) is not None:
        return False
    actual = metadata.per_rank_actual_token
    logical = metadata.per_rank_logical_token
    if logical is None:
        logical = actual
    total = metadata.total_seq_lens
    cp_size, cp_rank = strategy.cp_size, strategy.cp_rank
    if (
        cp_size != 8
        or not isinstance(cp_rank, int)
        or not 0 <= cp_rank < cp_size
        or not isinstance(total, int)
        or total <= 0
        or not isinstance(logical, (list, tuple))
        or not isinstance(actual, (list, tuple))
        or len(logical) != cp_size
        or len(actual) != cp_size
        or any(not isinstance(n, int) or n <= 0 for n in logical)
        or any(not isinstance(n, int) or n <= 0 for n in actual)
        or any(p < n for p, n in zip(actual, logical))
        or list(logical) != [len(range(rank, total, cp_size)) for rank in range(cp_size)]
        or latent.shape[0] != actual[cp_rank]
    ):
        return False
    loc = batch.out_cache_loc
    return (
        isinstance(loc, torch.Tensor)
        and loc.dtype == torch.int64
        and loc.ndim == 1
        and loc.is_contiguous()
        and loc.numel() == total
        and loc.device == latent.device
    )


def _vendor_function():
    from lightop import rmsnorm

    function = getattr(rmsnorm, "fused_rms_norm_rope_contiguous_q_cat", None)
    if not callable(function) or not callable(getattr(rmsnorm.op, "fused_rms_norm_rope_contiguous_q_cat", None)):
        raise RuntimeError("GLM-5.3 FP8 KV producer requires the paired LightOp fused API")
    if 256 not in getattr(rmsnorm.op, "MLA_ROPE_SUPPORTED_Q_HEAD_STRIDES", ()):
        raise RuntimeError("GLM-5.3 FP8 KV producer requires LightOp Q head stride 256")
    return function


def use_fused_kv_producer(attn, batch, positions, latent):
    if not envs.SGLANG_HCU_GLM53_FP8_KV_ROPE_STORE.get():
        return False
    from sglang.srt.layers.cp.utils import get_cp_strategy
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool, has_forward_context

    if not has_forward_context():
        return False
    backend, pool = get_attn_backend(), get_token_to_kv_pool()
    ordinary = getattr(backend, "_ordinary_glm53_cp_prefill", None)
    if not callable(ordinary) or not ordinary(batch):
        return False
    if (
        attn.is_nextn
        or not attn.use_dsa
        or attn.dcp_size != 1
        or attn.use_deep_gemm_bmm
        or attn.attn_mqa.is_cross_attention
        or attn.current_attention_backend not in ("dsa", "nsa")
        or attn.num_local_heads != 64
        or attn.kv_lora_rank != 512
        or attn.qk_rope_head_dim != 64
        or attn.qk_head_dim != 256
        or not isinstance(latent, torch.Tensor)
        or not latent.is_cuda
        or latent.dtype != torch.bfloat16
        or latent.ndim != 2
        or latent.shape[1] != 576
        or latent.stride(1) != 1
        or latent.stride(0) % 8
        or latent.data_ptr() % 16
        or not isinstance(positions, torch.Tensor)
        or positions.dtype != torch.int64
        or positions.ndim != 1
        or positions.numel() != latent.shape[0]
        or not positions.is_contiguous()
        or positions.device != latent.device
        or not _canonical_metadata(get_cp_strategy(), batch, latent)
    ):
        return False
    norm, rope = attn.kv_a_layernorm, attn.rotary_emb
    if type(norm) is not RMSNorm or type(rope) is not RotaryEmbedding:
        return False
    weight, cache = norm.weight, rope.cos_sin_cache
    if (
        norm.cast_x_before_out_mul
        or norm.variance_size_override is not None
        or norm.fp32_residual
        or not norm.has_weight
        or norm.override_orig_dtype is not None
        or getattr(norm, "x_pad_to_multiple", 0)
        or getattr(norm, "_quant_linear", None) is not None
        or not math.isfinite(norm.variance_epsilon)
        or norm.variance_epsilon <= 0
        or weight.dtype != torch.bfloat16
        or weight.shape != (512,)
        or not weight.is_contiguous()
        or weight.device != latent.device
        or weight.data_ptr() % 16
        or rope._force_native
        or rope.rotary_dim != 64
        or rope.is_neox_style
        or cache.dtype != torch.bfloat16
        or cache.ndim != 2
        or cache.shape[1] != 64
        or not cache.is_contiguous()
        or cache.device != latent.device
        or cache.data_ptr() % 16
        or attn._fuse_rope_for_trtllm_mla(batch)
    ):
        return False
    stored = pool.get_key_buffer(attn.attn_mqa.layer_id)
    if (
        not stored.is_contiguous()
        or stored.numel() == 0
        or stored.device != latent.device
        or stored.numel() * stored.element_size() % 656
        or stored.data_ptr() % 16
        or torch.cuda.get_device_properties(latent.device).gcnArchName.split(":", 1)[0] != "gfx938"
    ):
        return False
    _vendor_function()
    return True


def fuse_kv_rope_store(attn, batch, positions, query_nope, query_rope, latent):
    from sglang.srt.layers.cp.utils import get_cp_strategy
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    from hcu_sglang.layers.attention.current_dsa.packed_kv_exchange import (
        gather_packed_kv_cache_and_store,
    )

    tokens = latent.shape[0]
    if (
        query_nope.shape != (tokens, 64, 512)
        or query_rope.shape != (tokens, 64, 64)
        or query_nope.dtype != torch.bfloat16
        or query_rope.dtype != torch.bfloat16
        or query_nope.device != latent.device
        or query_rope.device != latent.device
        or query_nope.stride(-1) != 1
        or query_rope.stride(-1) != 1
        or query_rope.stride(1) != 256
        or query_nope.data_ptr() % 16
        or query_rope.data_ptr() % 16
    ):
        raise RuntimeError("GLM-5.3 FP8 KV producer received unsupported Q geometry")
    packed_kv = torch.empty((tokens, 656), dtype=torch.uint8, device=latent.device)
    local_slots = torch.arange(tokens, dtype=torch.int64, device=latent.device)
    normalized = latent.new_empty((tokens, 512))
    query = query_nope.new_empty((tokens, 64, 576))
    with torch.profiler.record_function("glm_prefill_kv_norm_rope_pack"):
        _vendor_function()(
            positions, query_rope, latent[:, 512:], latent[:, :512], normalized,
            attn.kv_a_layernorm.weight, attn.rotary_emb.cos_sin_cache,
            local_slots, packed_kv, query_nope, query, "fp8_e4m3", 1.0,
            is_neox_style=attn.rotary_emb.is_neox_style,
            epsilon=attn.kv_a_layernorm.variance_epsilon,
            flashmla_fp8_layout=True, kv_quant_epsilon=1e-6,
        )
    with torch.profiler.record_function("glm_prefill_packed_kv_cp_gather_store"):
        cache = get_token_to_kv_pool().get_key_buffer(attn.attn_mqa.layer_id)
        gather_packed_kv_cache_and_store(
            get_cp_strategy(), packed_kv, batch, cache.view(torch.uint8).view(-1, 656),
        )
    return PrefillQWithStoredKV(query)
