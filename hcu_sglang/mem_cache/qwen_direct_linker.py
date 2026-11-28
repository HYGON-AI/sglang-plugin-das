# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def _mha_buffers(pool, page_size):
    from hcu_sglang.mem_cache.hcu_mha_pool import HCUMHATokenToKVPool

    if not isinstance(pool, HCUMHATokenToKVPool):
        raise ValueError("Qwen direct linker requires HCU page-major MHA pools.")
    if pool.page_size != page_size or page_size <= 0:
        raise ValueError("Qwen direct linker requires matching tree and KV pages.")
    if any(
        getattr(pool, name, None) is not None
        for name in (
            "k_scale_buffer", "v_scale_buffer",
            "native_k_scale_buffer", "native_v_scale_buffer",
        )
    ) or (
        getattr(pool, "quant_method", None) is not None and pool.is_quantized_kv_cache
    ):
        raise ValueError("Qwen direct linker does not support scaled KV storage.")
    if not pool.k_buffer or len(pool.k_buffer) != len(pool.v_buffer):
        raise ValueError("Qwen direct linker requires paired K/V buffers.")
    buffers = [*pool.k_buffer, *pool.v_buffer]
    if any(
        buffer.ndim != 4
        or buffer.shape[2] != page_size
        or not buffer.is_contiguous()
        for buffer in buffers
    ):
        raise ValueError("Qwen direct linker requires contiguous BHSD pages.")
    return buffers


def build_qwen_pool_group(kvcache, params, page_size):
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
        DevicePoolEntry,
        DevicePoolGroup,
    )
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    if params.pp_size != 1 or params.attn_cp_size != 1:
        raise ValueError("Qwen direct linker currently supports TP without PP/CP.")
    if params.mtp_draft_mamba_pools:
        raise ValueError("Qwen direct linker does not support separate draft GDN state.")
    req_pool = params.req_to_token_pool
    if (
        req_pool.mamba_ckpt_pool is not None
        or req_pool.short_conv_pool.enabled
        or req_pool.ngram_pool.enabled
        or req_pool.mamba_v2p_table is not None
    ):
        raise ValueError(
            "Qwen direct linker requires physical, unquantized GDN slots "
            "without side states."
        )
    buffers = _mha_buffers(kvcache.full_kv_pool, page_size)
    n = len(kvcache.full_kv_pool.k_buffer)
    full_mapping = {
        gid - kvcache.start_layer: (lid, n + lid)
        for gid, lid in kvcache.full_attention_layer_id_mapping.items()
    }
    if sorted(kvcache.full_attention_layer_id_mapping.values()) != list(range(n)):
        raise ValueError("Qwen full-attention mapping does not cover its physical buffers.")
    mamba_pool = req_pool.mamba_pool
    mamba_mapping = {
        gid - kvcache.start_layer: lid for gid, lid in req_pool.mamba_map.items()
    }
    if sorted(mamba_mapping.values()) != list(range(mamba_pool.num_mamba_layers)):
        raise ValueError("Qwen GDN mapping does not cover its physical buffers.")
    layers = set(full_mapping) | set(mamba_mapping)
    if (
        not layers
        or set(full_mapping) & set(mamba_mapping)
        or layers != set(range(max(layers) + 1))
    ):
        raise ValueError(
            "Qwen direct linker requires a complete disjoint GQA/GDN layer mapping."
        )
    final_layer = max(layers)
    for draft in params.mtp_draft_device_pools:
        draft_kv = draft
        if isinstance(draft, HybridLinearKVPool):
            if draft.mamba_pool is not mamba_pool:
                raise ValueError("Qwen direct linker requires shared target/draft GDN state.")
            draft_kv = draft.full_kv_pool
        draft_buffers = _mha_buffers(draft_kv, page_size)
        indices = tuple(range(len(buffers), len(buffers) + len(draft_buffers)))
        full_mapping[final_layer] = (*full_mapping.get(final_layer, ()), *indices)
        buffers.extend(draft_buffers)
    kv_entry = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=kvcache.full_kv_pool,
        components=[buffers],
        layer_mapping=full_mapping,
        page_size=page_size,
        rows_are_pages=True,
    )
    state = mamba_pool.mamba_cache
    components = [
        [tensor[layer] for layer in range(mamba_pool.num_mamba_layers)]
        for tensor in (state.temporal, *state.conv)
        if tensor.numel()
    ]
    if any(not tensor.is_contiguous() for group in components for tensor in group):
        raise ValueError("Qwen direct linker requires contiguous GDN state slots.")
    mamba_entry = DevicePoolEntry(
        name=PoolName.MAMBA,
        indices_from_pool=PoolName.MAMBA,
        device_pool=mamba_pool,
        components=components,
        layer_mapping=mamba_mapping,
        page_size=1,
        rows_are_pages=True,
    )
    group = DevicePoolGroup([kv_entry, mamba_entry], final_layer + 1, page_size)
    schema = [
        str(getattr(kvcache.full_kv_pool, "dtype", None)),
        [
            (
                str(entry.name), entry.page_size, sorted(entry.layer_mapping.items()),
                [
                    [(list(t.shape[1:]), str(t.dtype)) for t in component]
                    for component in entry.components
                ],
            )
            for entry in group.entries
        ],
    ]
    group.hcu_qwen_schema = hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode()
    ).hexdigest()[:24]
    group.hcu_qwen_drafts = params.mtp_draft_device_pools
    return group


def around_resolve_pool_group(original, *, kvcache, page_size, params, components):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.runtime_context import get_memory, get_parallel

    if not isinstance(kvcache, HybridLinearKVPool) or components != {
        ComponentType.FULL, ComponentType.MAMBA,
    }:
        return original(
            kvcache=kvcache, page_size=page_size, params=params, components=components
        )
    if get_memory().unified_cache_external_linker_backend != "mooncake":
        raise ValueError("Qwen direct linker currently requires the Mooncake backend.")
    if (
        get_memory().enable_unified_memory
        or getattr(get_parallel(), "attn_dcp_size", 1) != 1
    ):
        raise ValueError("Qwen direct linker does not support unified memory or DCP.")
    return build_qwen_pool_group(kvcache, params, page_size)


class _DraftLoadCounter:
    def __init__(self, counter, target_layer):
        self.counter = counter
        self.target_layer = target_layer

    def wait_until(self, threshold):
        self.counter.wait_until(self.target_layer)


def around_linker_init(original, self, server_args, params, **kwargs):
    result = original(self, server_args, params, **kwargs)
    group = self.pool_group
    if hasattr(group, "hcu_qwen_schema"):
        from sglang.srt.runtime_context import get_parallel

        suffix = f"_hcu_qwen_v1_tp{get_parallel().tp_size}_{group.hcu_qwen_schema}"
        self.storage.mla_suffix += suffix
        self.storage.mha_suffix += suffix
        counter = _DraftLoadCounter(self.layer_done_counter, group.num_layers - 1)
        for draft in group.hcu_qwen_drafts:
            draft.register_layer_transfer_counter(counter)
        logger.info(
            "HCU Qwen direct L3 enabled: layers=%d, page_size=%d, drafts=%d, suffix=%s",
            group.num_layers,
            group.page_size,
            len(group.hcu_qwen_drafts),
            self.storage.mla_suffix,
        )
    return result


def around_component_keys(original, self, page_keys, transfer):
    from sglang.srt.mem_cache.hicache_storage import PoolName

    if hasattr(getattr(self, "mem_pool_host", None), "hcu_qwen_schema"):
        name = "k" if transfer.name == PoolName.KV else transfer.name
        return [f"{key}_{self.mla_suffix}_{name}" for key in page_keys], 1
    return original(self, page_keys, transfer)


def around_build_transfer(original, self, phase, node, keys):
    from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
    from sglang.srt.mem_cache.unified_cache.components.base import LinkerTransferPhase

    if phase == LinkerTransferPhase.OFFLOAD:
        if node is None or not node.hash_value:
            return None
        value = node.component_data[self.component_type].value
        if value is None:
            return None
        return PoolTransfer(
            name=PoolName.MAMBA,
            device_indices=value,
            keys=[node.hash_value[-1]],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
    if not keys:
        return None
    transfer = PoolTransfer(
        name=PoolName.MAMBA,
        keys=[keys[-1]],
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    if phase == LinkerTransferPhase.LOAD:
        transfer.device_indices = self._alloc_mamba_slot()
    return transfer


def around_update_load(original, self, phase, req, full_transfer, transfer, prefix_len):
    from sglang.srt.mem_cache.unified_cache.components.base import ExternalLinkerLoadPhase

    pool = self.cache.req_to_token_pool
    if phase == ExternalLinkerLoadPhase.ABORT:
        pool.mamba_allocator.free(transfer.device_indices)
        return None
    if phase != ExternalLinkerLoadPhase.PREPARE:
        return original(self, phase, req, full_transfer, transfer, prefix_len)
    if req.kv.holds_mamba:
        pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
    req.kv.mamba_pool_idx = transfer.device_indices[0]
    write_pos = pool.mamba_pool.replayssm_write_pos
    if write_pos is not None:
        write_pos[pool.translate_mamba_indices(req.kv.mamba_pool_idx.unsqueeze(0))] = 0
    req.kv.mamba_cow_src_index = None
    req.kv.mamba_needs_clear = False
    return transfer


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return
    from sglang.srt.mem_cache.unified_cache import unified_cache_linker
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    unified_cache_linker._EXTERNAL_LINKER_SUPPORTED_COMPONENTS = (
        unified_cache_linker._EXTERNAL_LINKER_SUPPORTED_COMPONENTS | {ComponentType.MAMBA}
    )
    targets = (
        (
            "hybrid_cache.linker_pool_assembler.resolve_hybrid_device_pool_group",
            around_resolve_pool_group,
        ),
        (
            "storage.mooncake_store.mooncake_direct_linker.MooncakeDirectLinker.__init__",
            around_linker_init,
        ),
        (
            "storage.mooncake_store.mooncake_store."
            "MooncakeStore._get_hybrid_page_component_keys",
            around_component_keys,
        ),
        (
            "unified_cache.components.mamba.MambaComponent.build_external_linker_transfer",
            around_build_transfer,
        ),
        (
            "unified_cache.components.mamba.MambaComponent.update_external_linker_load",
            around_update_load,
        ),
    )
    for target, hook in targets:
        HookRegistry.register(f"sglang.srt.mem_cache.{target}", hook, HookType.AROUND)
