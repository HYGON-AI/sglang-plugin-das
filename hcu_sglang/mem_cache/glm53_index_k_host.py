# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from msgspec.structs import replace
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerHostPoolBuilder,
    DSAIndexerPoolHost,
)


class _IndexKPoolByModelLayer:
    def __init__(self, pool):
        self._pool = pool
        buffers = pool.index_k_with_scale_buffer
        if buffers is None:
            buffers = [
                buffer.view(torch.uint8).flatten(1) for buffer in pool.index_k_buffer
            ]
        model_layer_buffers = [None] * pool.layer_num
        for global_layer, buffer in zip(pool.indexer_layer_ids, buffers, strict=True):
            model_layer_buffers[global_layer - pool.start_layer] = buffer
        self.index_k_with_scale_buffer = tuple(model_layer_buffers)

    def __getattr__(self, name):
        return getattr(self._pool, name)


class _GLM53IndexerPoolHost(DSAIndexerPoolHost):
    def __init__(self, *, decl, packed_draft_device_pools, **kwargs):
        draft_views = tuple(
            _IndexKPoolByModelLayer(pool) for pool in packed_draft_device_pools
        )
        self._draft_transfer_pools = {
            id(pool): view
            for pool, view in zip(packed_draft_device_pools, draft_views, strict=True)
        }
        super().__init__(
            decl=replace(decl, device_pool=_IndexKPoolByModelLayer(decl.device_pool)),
            packed_draft_device_pools=draft_views,
            **kwargs,
        )

    def init_kv_buffer(self):
        if self.layer_num:
            return super().init_kv_buffer()
        if self.layout == "layer_first":
            shape = (0, self.indexer_page_num, self.indexer_page_stride_size)
        elif self.layout in ("page_first", "page_first_direct"):
            shape = (self.indexer_page_num, 0, 1, self.indexer_page_stride_size)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.pin_memory = False
        self.index_k_with_scale_buffer = torch.empty(
            shape, dtype=self.indexer_dtype, device=self.device
        )
        self.packed_device_index_buffers = []
        self.index_k_data_refs = []
        self.index_k_device_ptrs = torch.empty(
            0, dtype=torch.uint64, device=self.device_pool.device
        )
        self.index_k_data_ptrs = self.index_k_device_ptrs

    def _init_write_back_staging_buffers(self):
        if self.layer_num:
            return super()._init_write_back_staging_buffers()
        self.staging_buffer = None

    def get_hybrid_pool_buffer(self):
        return super().get_hybrid_pool_buffer() if self.layer_num else []

    def _index_k_transfer_pool(self, device_pool, is_draft):
        if not is_draft:
            return self.device_pool
        pool = self._draft_transfer_pools.get(id(device_pool), device_pool)
        if not pool.index_k_with_scale_buffer[0].numel():
            return None
        return pool

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend,
        *, is_draft=False,
    ):
        pool = self._index_k_transfer_pool(device_pool, is_draft)
        if pool is not None and self.layer_num:
            return super().load_to_device_per_layer(
                pool, host_indices, device_indices, layer_id, io_backend,
                is_draft=is_draft,
            )

    def _backup_from_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend,
        *, is_draft=False,
    ):
        pool = self._index_k_transfer_pool(device_pool, is_draft)
        if pool is not None:
            return super()._backup_from_device_per_layer(
                pool, host_indices, device_indices, layer_id, io_backend,
                is_draft=is_draft,
            )

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        if self.layer_num:
            return super().backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend,
            )


class _GLM53IndexerHostPoolBuilder(DSAIndexerHostPoolBuilder):
    def build(self, *, decl, anchor_host, allocator_type, packed_draft_device_pools):
        return _GLM53IndexerPoolHost(
            decl=decl,
            anchor_host=anchor_host,
            allocator_type=allocator_type,
            packed_draft_device_pools=packed_draft_device_pools,
        )


def glm53_index_k_host_decl(decl):
    pool = decl.device_pool
    storage_info = decl.storage_info
    if pool.index_k_with_scale_buffer is None:
        storage_info = replace(
            storage_info,
            bytes_per_token_per_layer=pool.index_head_dim * torch.bfloat16.itemsize,
        )
    return replace(
        decl,
        storage_info=storage_info,
        host_pool_builder=_GLM53IndexerHostPoolBuilder(),
    )
