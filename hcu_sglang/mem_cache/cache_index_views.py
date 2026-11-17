# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from hcu_sglang.utils.detection import is_hcu


def around_full_index_split(original, self, new_parent, child):
    value = child.component_data[self.component_type].value
    if not is_hcu() or value is None or not value.is_cuda or value.ndim != 1 or not value.is_contiguous() or value.dtype not in (torch.int32, torch.int64):
        return original(self, new_parent, child)
    split_len = len(new_parent.key)
    if split_len < len(value) - split_len or value.untyped_storage().nbytes() > 2 * split_len * value.element_size():
        return original(self, new_parent, child)
    from hcu_sglang.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
        CpCacheLayerSplitDeepSeekV4TokenToKVPool,
    )
    if not isinstance(self.cache.token_to_kv_pool_allocator.get_kvcache(), CpCacheLayerSplitDeepSeekV4TokenToKVPool):
        return original(self, new_parent, child)
    return _split_full_index_views(self, new_parent, child)


def _split_full_index_views(
    self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
):
    ct = self.component_type
    new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
    new_parent.component_data[ct].session_ref = child.component_data[ct].session_ref
    child_cd = child.component_data[ct]
    new_parent.component_data[ct].host_lock_ref = child_cd.host_lock_ref
    # A Full host lock starts as a one-node segment. Splitting that node
    # extends the segment through the inserted prefix, so its boundary
    # moves to the prefix's older edge just like the SWA segment boundary.
    host_uuid = child_cd.metadata.pop("host_uuid", None)
    if host_uuid is not None:
        new_parent.component_data[ct].metadata["host_uuid"] = host_uuid
    assert new_parent.component_data[ct].session_ids is None
    split_len = len(new_parent.key)
    if child_cd.value is not None:
        new_parent.component_data[ct].value = child_cd.value[:split_len]
        child_cd.value = child_cd.value[split_len:].clone()
    if child_cd.host_value is not None:
        new_parent.component_data[ct].host_value = child_cd.host_value[
            :split_len
        ].clone()
        child_cd.host_value = child_cd.host_value[split_len:].clone()


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        "sglang.srt.mem_cache.unified_cache.components.full.FullComponent.redistribute_on_node_split",
        around_full_index_split,
        HookType.AROUND,
    )
