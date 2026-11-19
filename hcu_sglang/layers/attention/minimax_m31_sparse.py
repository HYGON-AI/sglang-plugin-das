import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from inspect import signature

import torch

from hcu_sglang.layers.attention import minimax_m31_flashmla as native

LOG = logging.getLogger(__name__)


@dataclass
class _PrefillMetadata:
    prefix_lens: torch.Tensor
    page_table: torch.Tensor | None = None
    index_metadata: tuple[torch.Tensor, torch.Tensor] | None = None


_cp_prefill_metadata = ContextVar("minimax_m31_cp_prefill_metadata", default=None)


@contextmanager
def cp_prefill_metadata(meta, segment, lengths, q_lengths):
    if (
        os.getenv("SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA", "1") != "1"
        or (lengths.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        yield lengths - q_lengths
        return

    by_segment = getattr(meta, "_minimax_cp_prefill_metadata", None)
    if by_segment is None:
        by_segment = meta._minimax_cp_prefill_metadata = {}
    stream = torch.cuda.current_stream(lengths.device).cuda_stream if lengths.is_cuda else None
    key = (lengths.device, stream, segment)
    plan = by_segment.get(key)
    if plan is None:
        plan = by_segment[key] = _PrefillMetadata(lengths - q_lengths)

    token = _cp_prefill_metadata.set(plan)
    try:
        yield plan.prefix_lens
    finally:
        _cp_prefill_metadata.reset(token)


@cache
def _signature(original):
    return signature(original)


def _args(original, args, kwargs):
    bound = _signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


@cache
def _announce(phase):
    LOG.info("MiniMax M3.1 native gfx938 FlashMLA %s active", phase)


def _supported(p, decode=False, hisparse=False):
    return (
        p["disable_index_value"]
        and p["score_type"] == "max"
        and p["sink"] is None
        and p["idx_sink"] is None
        and p["block_size_k"] == p["page_size"] == 128
        and p["topk"] == 16
        and p["q"].is_cuda
        and p["q"].shape[0] > 0
        and p["q"].shape[1:] == (64, 128)
        and p["k_cache"].shape[-2:] == p["v_cache"].shape[-2:] == (4, 128)
        and p["idx_q"].shape[1:] == (4, 128)
        and p["idx_k_cache"].shape[-2:] == (1, 128)
        and p["k_cache"].dtype == p["v_cache"].dtype
        and p["k_cache"].dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and p["idx_k_cache"].dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and p.get("loc_mapping") is None
        and p.get("dense_main_attn_fn") is None
        and (hisparse or p.get("hisparse_swap_in_fn") is None)
        and p.get("indexer_cp") is None
        and (decode or p["seqlens_cpu"] is not None)
        and native.flash_mla_gfx938_available()
    )


def _query(q, k, scale):
    if q.dtype == k.dtype:
        return q
    if q.dtype == torch.bfloat16 and k.dtype == torch.float8_e4m3fn:
        from sglang.srt.layers.attention.minimax_sparse_backend import _quant_q_fp8

        return _quant_q_fp8(q, scale)
    raise ValueError(f"MiniMax FlashMLA cannot pair query {q.dtype} with cache {k.dtype}")


def _page_table(p, slots, lengths, extent):
    num_pages = (extent + 127) // 128
    if num_pages <= 0:
        raise ValueError("MiniMax FlashMLA needs a positive KV extent")
    page_table = torch.empty((slots.numel(), num_pages), device=slots.device, dtype=torch.int32)
    native.update_flash_mla_page_table(page_table, p["req_to_token"], slots, lengths, 128)
    return page_table


def _prefill(original, *args, **kwargs):
    p = _args(original, args, kwargs)
    if not _supported(p):
        return original(*args, **kwargs)

    _announce("prefill")
    cu, lengths, prefix = p["cu_seqlens"], p["seq_lens"], p["prefix_lens"]
    metadata = _cp_prefill_metadata.get()
    if metadata is None:
        metadata = _PrefillMetadata(prefix)
    if metadata.page_table is None:
        metadata.page_table = _page_table(p, p["slot_ids"], lengths, p["max_seqlen_k"])
    table = metadata.page_table
    topk = p["cached_topk_idx"]
    if topk is None:
        idx_k = p["idx_k_cache"].view(-1, 1, 128)
        idx_q = _query(p["idx_q"], idx_k, p["idx_q_scale"])
        if metadata.index_metadata is None:
            k_end = native.build_flash_mla_sparse_prefill_k_end(
                cu, prefix, lengths - prefix, idx_q.shape[0], idx_q.device
            )
            metadata.index_metadata = k_end, table.clamp_min(0).reshape(-1).contiguous()
        k_end, safe_pages = metadata.index_metadata
        topk = native.flash_mla_sparse_prefill_indexer(
            idx_q, idx_k, table, cu, prefix, p["seqlens_cpu"], 128, 16,
            p["init_blocks"], p["local_blocks"],
            k_end=k_end, safe_pages=safe_pages,
            index_scales=(p["idx_q_scale"], p["idx_k_scale"]),
        )
    k, v = p["k_cache"].view(-1, 4, 128), p["v_cache"].view(-1, 4, 128)
    q = _query(p["q"], k, p["q_scale"])
    result = native.flash_mla_sparse_prefill_main(
        q, k, v, topk, table, cu, lengths, prefix, 128,
        sm_scale=p["sm_scale"], q_scale=p["q_scale"], k_scale=p["k_scale"], v_scale=p["v_scale"],
    )
    return (None, result, topk) if p["return_topk_idx"] else (None, result)


@dataclass
class _DecodeMetadata:
    req_to_token: torch.Tensor
    slot_ids: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen: int
    group: int
    slots: torch.Tensor
    table: torch.Tensor
    k_end: torch.Tensor


_decode_metadata = ContextVar("minimax_m31_decode_metadata", default=None)


def _model_forward(original, model, input_ids, positions, forward_batch, *args, **kwargs):
    mode = forward_batch.forward_mode
    if (
        not (mode.is_decode() or mode.is_target_verify())
        or forward_batch.can_run_tbo
        or os.getenv("SGLANG_MINIMAX_CACHE_DECODE_INDEX_METADATA", "1") != "1"
    ):
        return original(model, input_ids, positions, forward_batch, *args, **kwargs)
    token = _decode_metadata.set([])
    try:
        return original(model, input_ids, positions, forward_batch, *args, **kwargs)
    finally:
        _decode_metadata.reset(token)


def _decode_plan(p, *, reuse):
    cached = _decode_metadata.get() if reuse else None
    req_to_token = p["req_to_token"]
    slot_ids = p["slot_ids"]
    seq_lens = p["seq_lens"]
    extent, group = p["max_seqlen"], p["packed_queries"]
    if cached:
        plan = cached[0]
        if (
            plan.req_to_token is req_to_token
            and plan.slot_ids is slot_ids
            and plan.seq_lens is seq_lens
            and plan.max_seqlen == extent
            and plan.group == group
        ):
            return plan

    slots = slot_ids.reshape(-1, group)[:, 0].contiguous()
    lengths = seq_lens.reshape(-1, group)[:, -1].contiguous()
    table = _page_table(p, slots, lengths, extent)
    plan = _DecodeMetadata(
        req_to_token, slot_ids, seq_lens, extent, group, slots, table,
        seq_lens.to(torch.int32).contiguous(),
    )
    if cached is not None:
        cached[:] = [plan]
    return plan


def _decode(original, *args, **kwargs):
    from hcu_sglang.models.minimax_m31_hisparse import verify_context

    context = verify_context.get()
    p = _args(original, args, kwargs)
    if not _supported(p, decode=True, hisparse=context is not None):
        if context is not None:
            raise ValueError("MiniMax HiSparse requires the native FlashMLA MSA128 contract")
        return original(*args, **kwargs)

    _announce("decode")
    group = p["packed_queries"]
    plan = _decode_plan(p, reuse=context is None)
    slots, table = plan.slots, plan.table
    topk = p["cached_topk_idx"]
    if topk is None:
        if not hasattr(table, "_live_stage1"):
            native.pack_live_stage1_pages(table, plan.k_end.reshape(-1, group)[:, -1], 128)
        idx_k = p["idx_k_cache"].view(-1, 1, 128)
        idx_q = _query(p["idx_q"], idx_k, p["idx_q_scale"])
        topk = native.flash_mla_sparse_decode_indexer(
            idx_q, idx_k, table, plan.k_end, 128, 16, p["init_blocks"], p["local_blocks"],
            verify_group_size=group, is_verify=group > 1,
            index_scales=(p["idx_q_scale"], p["idx_k_scale"]),
        )
        if p["topk_out"] is not None:
            p["topk_out"].copy_(topk)
    k, v = p["k_cache"].view(-1, 4, 128), p["v_cache"].view(-1, 4, 128)
    q = _query(p["q"], k, p["q_scale"])
    main_args = dict(
        q=q, k_cache=k, v_cache=v, topk_idx=topk, page_table=table,
        seq_lens=p["seq_lens"], block_size_k=128, sched_meta=native.new_flash_mla_decode_metadata(),
        sm_scale=p["sm_scale"], q_scale=p["q_scale"], k_scale=p["k_scale"], v_scale=p["v_scale"],
        verify_group_size=group,
    )
    if context is None:
        return None, native.flash_mla_sparse_decode_main(**main_args)

    verify, layer_id = context
    if group != verify.width:
        raise ValueError("MiniMax HiSparse requires DSpark verify8")

    def hot_indices(selected, lengths):
        return verify.swap(slots, layer_id, selected, lengths)

    if verify.query_chunk_size < group:
        from hcu_sglang.layers.attention.minimax_hisparse import flash_mla_chunked_verify

        return None, flash_mla_chunked_verify(
            query_chunk_size=verify.query_chunk_size,
            hot_indices_fn=hot_indices,
            **main_args,
        )
    return None, native.flash_mla_sparse_decode_main(
        **main_args, hot_indices=hot_indices(topk, p["seq_lens"])
    )


def _index_dtype(original, *, fp8_attn_gemm, kv_cache_dtype, model_dtype):
    if (
        os.getenv("SGLANG_MINIMAX_INDEX_FP8") == "dot"
        and kv_cache_dtype == torch.float8_e4m3fn
        and torch.version.hip
    ):
        return torch.float8_e4m3fn
    return original(
        fp8_attn_gemm=fp8_attn_gemm, kv_cache_dtype=kv_cache_dtype, model_dtype=model_dtype
    )


def _can_fuse_kv_store(original, pool, index_pool, cache_k, cache_idx_k, unit_scales):
    # The pinned JIT store supports raw copies on HIP but explicitly rejects
    # BF16-to-FP8. Keep the same scaled-store fallback as the old baseline.
    if torch.version.hip and (
        cache_k.dtype != pool.main_pool.dtype or cache_idx_k.dtype != index_pool.dtype
    ):
        return False
    return original(pool, index_pool, cache_k, cache_idx_k, unit_scales)


def _restore_hip_verify(_result, backend, runner):
    # The adapter deliberately exposes CUDA semantics to SGLang. This
    # hardware-dependent flag selects the existing linear HIP verify metadata
    # and grouped sparse decode path, including in-graph metadata rebuilds.
    if not torch.version.hip or backend.is_hip:
        return
    from sglang.srt.layers.attention.minimax_sparse_backend import get_spec

    spec = get_spec()
    if spec.speculative_algorithm != "DSPARK":
        return
    if spec.speculative_eagle_topk != 1:
        raise ValueError("MiniMax HIP DSpark verification requires a linear draft chain")
    backend.is_hip = True


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.minimax_m3.MiniMaxM3Model.forward",
        _model_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.minimax_sparse_backend.MiniMaxSparseAttnBackend.__init__",
        _restore_hip_verify,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse.minimax_sparse_prefill",
        _prefill,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse.minimax_sparse_decode",
        _decode,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MiniMaxSparseKVPool._can_fuse_kv_index_store",
        _can_fuse_kv_store,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.get_minimax_sparse_index_dtype",
        _index_dtype,
        HookType.AROUND,
    )
