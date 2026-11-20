"""HCU MLA KV-pool compatibility and write hooks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def get_key_buffer_deepseek_v2(pool: Any, layer_id: int):
    if pool.layer_transfer_counter is not None:
        pool.layer_transfer_counter.wait_until(layer_id - pool.start_layer)
    buffer = pool.kv_buffer[layer_id - pool.start_layer]
    if pool.store_dtype != pool.dtype and pool.dtype not in (
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ):
        return buffer.view(pool.dtype)
    return buffer, pool.dtype


def set_kv_buffer_opt(
    pool: Any,
    layer: Any,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    layer_id = layer.layer_id
    assert not (pool.use_dsa and pool.dsa_kv_cache_store_fp8)
    cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
    if cache_k.dtype != pool.dtype:
        cache_k = cache_k.to(pool.dtype)
    if pool.store_dtype != pool.dtype:
        cache_k = cache_k.view(pool.store_dtype)
    pool.kv_buffer[layer_id - pool.start_layer][loc] = cache_k


def _write_triton(
    dst_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> None:
    from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton

    set_mla_kv_buffer_triton(
        dst_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
    )


def _write_lightop(
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    dst_buffer: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    from lightop import kvcache

    kvcache.fused_quantize_and_store_mla_kv_cache(
        cache_k_nope,
        cache_k_rope,
        dst_buffer,
        loc,
        "fp8_e4m3",
        1e-6,
    )


def around_create_mla_buffers(original, pool):
    if not is_hcu() or not pool.dsa_kv_cache_store_fp8:
        return original(pool)
    from sglang.srt.runtime_context import (
        get_exec,
        get_parallel,
        get_spec,
        max_speculative_num_draft_tokens,
    )

    if not get_parallel().dcp_enabled or pool.write_loc_is_dcp_resolved:
        return original(pool)
    graph = get_exec().graph.cuda_graph_config
    decode_rows = 0
    if graph.decode.backend != "disabled":
        decode_bs = max(graph.decode.bs or (), default=graph.decode.max_bs or 0)
        spec = get_spec()
        tokens_per_req = 1
        if spec.speculative_algorithm == "EAGLE":
            tokens_per_req = max(
                tokens_per_req,
                spec.speculative_eagle_topk or 0,
                max_speculative_num_draft_tokens() or 0,
            )
        decode_rows = decode_bs * tokens_per_req
    prefill_rows = 0
    if graph.prefill.backend != "disabled":
        prefill_rows = max(graph.prefill.bs or (), default=0)
    rows = max(decode_rows, prefill_rows)
    page_size = pool.page_size
    capacity = (rows + page_size - 1) // page_size * page_size
    pool._hcu_dcp_padding_capacity = capacity
    if capacity == 0:
        return original(pool)
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

    with pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(pool.custom_mem_pool)
            if pool.custom_mem_pool
            else nullcontext()
        ):
            pool.kv_buffer = [
                torch.zeros(
                    (pool.size + page_size + capacity, 1, pool.kv_cache_dim),
                    dtype=pool.store_dtype,
                    device=pool.device,
                )
                for _ in range(pool.layer_num)
            ]


def _localize_dcp_write(pool, loc, cache_k_nope, cache_k_rope):
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if not parallel.dcp_enabled or pool.write_loc_is_dcp_resolved:
        return loc, cache_k_nope, cache_k_rope
    valid_mask = (loc >= 0) & (loc % parallel.attn_dcp_size == parallel.attn_dcp_rank)
    if get_is_capture_mode():
        num_rows = loc.numel()
        capacity = pool._hcu_dcp_padding_capacity
        assert num_rows <= capacity, (
            f"HCU DCP graph KV rows ({num_rows}) exceed padding capacity ({capacity})"
        )
        padding_loc = pool.size + pool.page_size + torch.arange(
            num_rows, dtype=loc.dtype, device=loc.device
        )
        loc = torch.where(valid_mask, loc // parallel.attn_dcp_size, padding_loc)
    else:
        loc = loc[valid_mask] // parallel.attn_dcp_size
        cache_k_nope = cache_k_nope[valid_mask]
        cache_k_rope = cache_k_rope[valid_mask]
    return loc, cache_k_nope, cache_k_rope


def around_should_elide_dsa_index_k(original, *, is_draft_worker):
    if is_hcu():
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().dcp_enabled:
            return False
    return original(is_draft_worker=is_draft_worker)


def around_write_mla_kv_buffer(
    original: Callable[..., None],
    pool: Any,
    dst_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> None:
    if not is_hcu():
        return original(pool, dst_buffer, loc, cache_k_nope, cache_k_rope)
    if pool.dsa_kv_cache_store_fp8:
        loc, cache_k_nope, cache_k_rope = _localize_dcp_write(
            pool, loc, cache_k_nope, cache_k_rope
        )
        if loc.numel() == 0:
            return None
        _write_lightop(cache_k_nope, cache_k_rope, dst_buffer, loc)
        return None

    if cache_k_nope.dtype != pool.dtype:
        cache_k_nope = cache_k_nope.to(pool.dtype)
        cache_k_rope = cache_k_rope.to(pool.dtype)
    if pool.store_dtype != pool.dtype:
        cache_k_nope = cache_k_nope.view(pool.store_dtype)
        cache_k_rope = cache_k_rope.view(pool.store_dtype)
    if hasattr(pool, "full_to_hisparse_device_index_mapping"):
        from hcu_sglang.layers.attention.hisparse import enabled as hisparse_enabled

        if hisparse_enabled():
            loc = loc.clamp_min(0)
    _write_triton(dst_buffer, loc, cache_k_nope, cache_k_rope)
    return None


def _install_missing_methods() -> None:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    if not hasattr(MLATokenToKVPool, "get_key_buffer_DeepSeekV2"):
        MLATokenToKVPool.get_key_buffer_DeepSeekV2 = get_key_buffer_deepseek_v2
    if not hasattr(MLATokenToKVPool, "set_kv_buffer_opt"):
        MLATokenToKVPool.set_kv_buffer_opt = set_kv_buffer_opt


def around_calculate_mla_kv_cache_dim(original, *, model_config, kv_cache_dtype):
    if not is_hcu():
        return original(model_config=model_config, kv_cache_dtype=kv_cache_dtype)
    from sglang.srt.configs.model_config import is_deepseek_dsa

    if is_deepseek_dsa(model_config.hf_config):
        return original(model_config=model_config, kv_cache_dtype=kv_cache_dtype)
    from sglang.srt.runtime_context import get_exec
    from hcu_sglang.layers.attention.hcu_mla_utils import get_hcu_mla_fp8_kv_cache_dim

    kernel = get_exec().kernel
    return get_hcu_mla_fp8_kv_cache_dim(
        model_config.kv_lora_rank + model_config.qk_rope_head_dim,
        model_config.qk_rope_head_dim,
        kv_cache_dtype,
        uses_hcu_mla=any(
            getattr(kernel, name, None) == "hcu_mla"
            for name in (
                "attention_backend",
                "prefill_attention_backend",
                "decode_attention_backend",
            )
        ),
    )


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _install_missing_methods()
    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MLATokenToKVPool._create_buffers",
        around_create_mla_buffers,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.kv_cache_configurator._should_elide_dsa_index_k",
        around_should_elide_dsa_index_k,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MLATokenToKVPool._write_mla_kv_buffer",
        around_write_mla_kv_buffer,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        around_calculate_mla_kv_cache_dim,
        HookType.AROUND,
    )
