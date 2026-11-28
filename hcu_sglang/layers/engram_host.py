# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import logging
import os
from typing import Optional

import torch
from torch import nn

from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs
from sglang.srt.layers.engram import EngramEmbedding, FP8_BLOCK_SIZE, _HostTable, engram_gather
from sglang.srt.runtime_context import get_parallel

from .engram_hsa import HcuAttachedAllocation, _HcuSharedAllocation

logger = logging.getLogger(__name__)


class _HostTableGroup:
    """CPU collectives for TP ranks sharing one host-memory allocation."""

    def __init__(self, ranks):
        self.ranks = ranks
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(torch.distributed.get_rank())
        # Disjoint TP groups may initialize different layers independently.
        # Only members participate; no world-wide group creation is required.
        self.cpu_group = torch.distributed.new_group(
            ranks=ranks, backend="gloo", use_local_synchronization=True
        )

    def barrier(self):
        torch.distributed.barrier(group=self.cpu_group)

    def all_gather_object(self, obj):
        objects = [None] * self.world_size
        torch.distributed.all_gather_object(objects, obj, group=self.cpu_group)
        return objects

    def broadcast_object(self, obj=None, src=0):
        objects = [obj]
        torch.distributed.broadcast_object_list(
            objects, src=self.ranks[src], group=self.cpu_group
        )
        return objects[0]


@functools.cache
def _get_host_table_group(tp_group):
    """Split a TP group by actual shared-memory access, once per TP group."""
    if tp_group.world_size == 1:
        return tp_group
    remaining = set(range(tp_group.world_size))
    local_ranks = None
    while remaining:
        source = min(remaining)
        same_node = in_the_same_node_as(tp_group.cpu_group, source_rank=source)
        members = [i for i, same in enumerate(same_node) if same]
        if source not in members:
            raise RuntimeError("Cannot determine the Engram host-memory group")
        remaining.difference_update(members)
        if tp_group.rank_in_group in members:
            local_ranks = [tp_group.ranks[i] for i in members]
    if local_ranks is None:
        raise RuntimeError("Current rank has no Engram host-memory group")
    if len(local_ranks) == tp_group.world_size:
        return tp_group
    return _HostTableGroup(local_ranks)


class _SharedTable:
    _weight_cache_skip = True
    layout = "shared"

    def __init__(self, allocation, group):
        self.allocation = allocation
        self.bytes = allocation.bytes
        self.device_ptr = allocation.device_ptr
        self.group = group
        self.dirty = False

    def finish_load(self, label):
        if self.dirty:
            self.group.barrier()
            self.dirty = False
            logger.info("Engram %s HSA IPC table ready (%d bytes, %d ranks)",
                        label, self.allocation.nbytes, self.group.world_size)

    def export_ipc(self):
        a = self.allocation
        return {"handle": bytes(a.handle), "span": a.span, "nbytes": a.nbytes}


class HcuEngramEmbedding(EngramEmbedding):
    _weight_cache_skip_attrs = ("host_table", "_host_table_ptrs", "_pending_host_table")

    def __init__(self, num_embeddings: int, dim: int, layer_id: int):
        layout = envs.SGLANG_HCU_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get()
        requested = (envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get()
                     or envs.SGLANG_HCU_ENABLE_DSV41_ENGRAM_HOST_TABLE.get())
        use_host = requested and (layout != "shared" or os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") == "0")
        self._pending_host_table = None
        if not use_host and not envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            super().__init__(num_embeddings, dim, layer_id)
            return
        nn.Module.__init__(self)
        self.dim = dim
        parallel = get_parallel()
        self.tp_size = parallel.tp_size
        self.row_start = num_embeddings * parallel.tp_rank // self.tp_size
        self.rows = num_embeddings * (parallel.tp_rank + 1) // self.tp_size - self.row_start
        self.host_table = None
        n = num_embeddings if use_host and layout == "shared" else self.rows
        w_bytes, s_bytes = n * dim, n * (dim // FP8_BLOCK_SIZE)
        if use_host and layout not in ("shared", "per_rank"):
            raise ValueError(f"Unsupported Engram host table layout {layout!r}")
        if use_host and torch.empty(0).device.type == "meta":
            self._pending_host_table = (n, w_bytes, s_bytes)
        elif use_host:
            nbytes = max(1, w_bytes + s_bytes)
            if layout == "shared":
                group = _get_host_table_group(parallel.tp_group)
                allocation = _HcuSharedAllocation(nbytes, group, group.rank_in_group)
                self.host_table = _SharedTable(allocation, group)
            else:
                self.host_table = _HostTable(layout, nbytes, f"sglang_engram_{layer_id}", parallel.tp_group)
        if self.host_table is None:
            self.weight = nn.Parameter(torch.empty(n, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
            self.scale = nn.Parameter(torch.empty(n, dim // FP8_BLOCK_SIZE, dtype=torch.float8_e8m0fnu), requires_grad=False)
        else:
            self._bind_host_table(n, w_bytes, s_bytes)
        self.weight.weight_loader = self._load_rows
        self.scale.weight_loader = self._load_rows

    def _bind_host_table(self, n: int, w_bytes: int, s_bytes: int):
        dim = self.dim
        raw = self.host_table.bytes[: w_bytes + s_bytes]
        weight = raw[:w_bytes].view(torch.float8_e4m3fn).view(n, dim)
        scale = raw[w_bytes:].view(torch.float8_e8m0fnu).view(n, dim // FP8_BLOCK_SIZE)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        device_ptr = self.host_table.device_ptr
        self._host_table_ptrs = (device_ptr, device_ptr + w_bytes)

    def export_ipc_host_state(self):
        if self.host_table is None:
            return None
        if not self._shared:
            raise RuntimeError("Engram weight-cache IPC requires shared HSA host tables")
        return self.host_table.export_ipc()

    def attach_ipc_host_state(self, desc: Optional[dict]) -> None:
        """Weight-cache engine: map the table the daemon loaded."""
        if self._pending_host_table is None:
            if desc is not None:
                raise RuntimeError(
                    "engram: the weight cache daemon holds a host table but this "
                    "engine was built without one; their launch environments differ"
                )
            return
        if desc is None:
            raise RuntimeError(
                "engram: this engine expects a shared host table but the weight "
                "cache daemon did not export one; their launch environments differ"
            )
        n, w_bytes, s_bytes = self._pending_host_table
        if desc["nbytes"] != max(1, w_bytes + s_bytes):
            raise RuntimeError(
                f"engram host table size mismatch: daemon {desc['nbytes']} bytes, "
                f"engine {max(1, w_bytes + s_bytes)} bytes"
            )
        self.host_table = _SharedTable(
            HcuAttachedAllocation(desc["handle"], desc["span"], desc["nbytes"]), None
        )
        self._bind_host_table(n, w_bytes, s_bytes)
        self.weight.weight_loader = self._load_rows
        self.scale.weight_loader = self._load_rows
        self._pending_host_table = None

    def _load_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if not self._shared:
            return super()._load_rows(param, loaded_weight)
        group = self.host_table.group
        if group is None:
            raise RuntimeError("An attached Engram table is read-only")
        n = param.shape[0]
        rows = slice(n * group.rank_in_group // group.world_size,
                     n * (group.rank_in_group + 1) // group.world_size)
        param.data[rows].copy_(loaded_weight[rows])
        self.host_table.dirty = True

    def forward(self, indices, forward_batch=None, *, cp_all_tokens=False):
        if not self._shared:
            return super().forward(indices, forward_batch, cp_all_tokens=cp_all_tokens)
        out = self._empty(indices)
        if indices.numel():
            engram_gather(*self._table_ptrs(), indices.reshape(-1), out.view(-1, self.dim),
                          self.dim, FP8_BLOCK_SIZE, row_hi=self.weight.shape[0])
        return out
