# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    is_allocation_symmetric,
)
from sglang.srt.runtime_context import get_parallel

from hcu_sglang.layers.attention.current_dsa.packed_kv_store import (
    store_interleaved_packed_kv,
)


def gather_packed_kv_cache_and_store(strategy, x, forward_batch, cache):
    metadata = forward_batch.attn_cp_metadata
    if metadata is None or getattr(metadata, "gather_index", None) is not None:
        raise ValueError("Packed KV exchange requires canonical interleave metadata")
    total_tokens = int(metadata.total_seq_lens)
    logical = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    actual = metadata.per_rank_actual_token
    cp_size, cp_rank = strategy.cp_size, strategy.cp_rank
    if (
        cp_size != 8
        or not 0 <= cp_rank < cp_size
        or len(logical) != cp_size
        or len(actual) != cp_size
        or total_tokens <= 0
        or any(int(n) <= 0 for n in logical)
        or any(int(p) < int(n) for p, n in zip(actual, logical))
        or list(logical) != [len(range(rank, total_tokens, cp_size)) for rank in range(cp_size)]
    ):
        raise ValueError("Packed KV exchange received unsupported CP lengths")
    local_logical_len = int(logical[cp_rank])
    physical_rank_len = max(actual)
    loc = forward_batch.out_cache_loc
    if (
        x.dtype != torch.uint8
        or x.ndim != 2
        or x.shape[1] != 656
        or x.shape[0] < local_logical_len
        or cache.dtype != torch.uint8
        or cache.ndim != 2
        or cache.shape[1] != 656
        or not cache.is_contiguous()
        or cache.data_ptr() % 16 != 0
        or loc.dtype != torch.int64
        or loc.ndim != 1
        or not loc.is_contiguous()
        or loc.numel() != total_tokens
        or not x.is_cuda
        or x.device != cache.device
        or x.device != loc.device
    ):
        raise ValueError("Packed KV exchange received incompatible tensor layouts")
    padded_x = x.new_zeros((physical_rank_len, 656))
    padded_x[:local_logical_len] = x[:local_logical_len]
    with use_symmetric_memory(
        get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        gathered = x.new_empty((cp_size * physical_rank_len, 656))
    attn_cp_all_gather_into_tensor(gathered, padded_x.contiguous())
    store_interleaved_packed_kv(gathered, loc, cache, cp_size, physical_rank_len)
