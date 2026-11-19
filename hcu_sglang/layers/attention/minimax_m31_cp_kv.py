import logging
import os
from dataclasses import dataclass
from functools import cache

import torch

LOG = logging.getLogger(__name__)
_HEADS = (4, 4, 1)
_HEAD_DIM = 128
_FP8 = torch.float8_e4m3fn


@dataclass
class _GatherWorkspace:
    send: torch.Tensor
    raw: torch.Tensor
    reorder: torch.Tensor


def _rank_major_reorder(meta, counts, max_rows, device):
    rank_rows = torch.arange(len(counts) * max_rows, device=device)
    if sum(counts) != rank_rows.numel():
        rank_rows = torch.cat(
            [rank_rows[i * max_rows : i * max_rows + rows] for i, rows in enumerate(counts)]
        )
    chunks = rank_rows.split(meta.reverse_split_len)
    return torch.cat([chunks[i] for i in meta.cp_reverse_index])


def _workspace(meta, counts, device, stream, dtype, width, allocation_context):
    workspaces = getattr(meta, "_minimax_cp_kv_workspaces", None)
    if workspaces is None:
        workspaces = meta._minimax_cp_kv_workspaces = {}
    key = (device, stream, dtype, width)
    workspace = workspaces.get(key)
    if workspace is None:
        max_rows = max(counts)
        send = torch.empty((max_rows, width), dtype=dtype, device=device)
        with allocation_context():
            raw = torch.empty((len(counts) * max_rows, width), dtype=dtype, device=device)
        reorder = _rank_major_reorder(meta, counts, max_rows, device)
        workspace = workspaces[key] = _GatherWorkspace(send, raw, reorder)
    return workspace


def _gather_payload(group, workspace, inputs, local_rows, pack, scales):
    send = workspace.send
    if local_rows:
        flat = tuple(x[:local_rows].flatten(1) for x in inputs)
        if pack is None:
            torch.cat(flat, dim=1, out=send[:local_rows])
        else:
            flat = tuple(
                x if x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
                else x.clone(memory_format=torch.contiguous_format)
                for x in flat
            )
            packet = pack(*flat, *scales)
            if local_rows == send.shape[0]:
                send = packet
            else:
                send[:local_rows].copy_(packet)
    if local_rows < send.shape[0]:
        send[local_rows:].zero_()
    group.all_gather_into_tensor(workspace.raw, send)


@cache
def _lightop_packet_ops():
    try:
        from lightop import op
        from lightop.minimax_cp_cache_scatter import minimax_cp_cache_scatter
        from lightop.minimax_cp_fp8_pack import minimax_cp_fp8_pack
    except (ImportError, OSError):
        return None
    if callable(getattr(op, "minimax_cp_fp8_pack", None)) and callable(
        getattr(op, "minimax_cp_cache_scatter", None)
    ):
        return minimax_cp_fp8_pack, minimax_cp_cache_scatter
    return None


@cache
def _announce(index_dtype, packet_dtype):
    if index_dtype == _FP8 and packet_dtype == torch.uint8:
        LOG.info("MiniMax CP KV write uses packed FP8 AllGather")
    elif index_dtype == _FP8:
        LOG.info("MiniMax CP KV write uses BF16 AllGather and fused Triton FP8 scatter")
    else:
        LOG.info("MiniMax CP KV write uses LightOp FP8 K/V + BF16 index K packets")


def try_store_cp_kv(pool, layer, batch, strategy, k, v, index_k):
    if os.environ.get("SGLANG_MINIMAX_CP_FP8_KV_EXCHANGE", "1") != "1":
        return None
    main = pool.main_pool
    index = pool.index_k_pool
    inputs = (k, v, index_k)
    slots = batch.out_cache_loc
    meta = batch.attn_cp_metadata
    counts = meta.per_rank_logical_token or meta.per_rank_actual_token
    if (
        main is None
        or index is None
        or pool._enable_hisparse
        or main.is_quantized_kv_cache
        or main.dtype != _FP8
        or index.dtype not in (_FP8, torch.bfloat16)
        or not k.is_cuda
        or any(
            x is None or x.dtype != torch.bfloat16 or x.device != k.device
            or x.shape[1:] != (heads, _HEAD_DIM)
            for x, heads in zip(inputs, _HEADS)
        )
        or slots.device != k.device
        or slots.dtype != torch.int64
        or slots.ndim != 1
        or not slots.is_contiguous()
        or slots.numel() == 0
        or slots.numel() != sum(counts)
        or torch.cuda.is_current_stream_capturing()
    ):
        return None

    scales = (layer.k_scale_float, layer.v_scale_float)
    if index.dtype == _FP8:
        if any(s not in (None, 1.0) for s in (*scales, layer.idx_k_scale_float)):
            return None
        from ...kernels.minimax_cp_cache import scatter_fp8_kv

        pack, scatter = None, scatter_fp8_kv
        packet_dtype, width = torch.bfloat16, 1152
        if os.environ.get("HCU_MINIMAX_CP_PACK_FP8") == "1":
            from hcu_adapter.kernels.cp_fp8_packet_triton import pack_unit_scale_fp8_kv
            from ...kernels.minimax_cp_cache import scatter_packed_fp8_kv

            pack, scatter = pack_unit_scale_fp8_kv, scatter_packed_fp8_kv
            packet_dtype = torch.uint8
    else:
        if not all(s is None or (type(s) in (int, float) and 1e-37 < s < 1e37) for s in scales):
            return None
        ops = _lightop_packet_ops()
        if ops is None:
            return None
        pack, scatter = ops
        packet_dtype, width = torch.uint8, 1280

    caches = (*pool.get_kv_buffer(layer.layer_id), pool.get_index_k_buffer(layer.layer_id))
    if any(
        x.device != k.device
        or x.dtype != dtype
        or x.shape != (caches[0].shape[0], heads, _HEAD_DIM)
        or x.stride()[1:] != (_HEAD_DIM, 1)
        or x.stride(0) < heads * _HEAD_DIM
        for x, heads, dtype in zip(caches, _HEADS, (_FP8, _FP8, index.dtype))
    ):
        return None
    if pack is not None:
        byte_caches = tuple(x.flatten(1).view(torch.uint8) for x in caches)
        if any(x.stride(0) % 16 or x.data_ptr() % 16 for x in byte_caches):
            return None

    local_rows = counts[strategy.cp_rank]
    if any(x.shape[0] < local_rows for x in inputs):
        raise ValueError(f"MiniMax CP needs {local_rows} local rows for KV exchange")

    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().attn_cp_group
    workspace = _workspace(
        meta, counts, k.device, torch.cuda.current_stream(k.device).cuda_stream,
        packet_dtype, width,
        lambda: use_symmetric_memory(group, disabled=not is_allocation_symmetric()),
    )
    _gather_payload(group, workspace, inputs, local_rows, pack, scales)
    if pack is None:
        scatter(workspace.raw, workspace.reorder, slots, caches)
    else:
        scatter(workspace.raw, workspace.reorder, slots, *byte_caches)
    _announce(index.dtype, packet_dtype)
    return caches
