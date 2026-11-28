# SPDX-License-Identifier: Apache-2.0
"""HCU Engram dispatch for the official kernel and torch-method seams."""

from functools import cache
from inspect import Parameter, signature

import torch
from sglang.kernels.ops.embeddings.engram_hash import MODE_EXTEND, MODE_VERIFY
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE


@cache
def _provider():
    try:
        from lightop.engram import engram_hash_ids_hcu
    except (ImportError, AttributeError):
        return None, False
    try:
        block = signature(engram_hash_ids_hcu).parameters.get("block")
    except (TypeError, ValueError):
        block = None
    supports_block = block is not None and block.kind in (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    )
    return engram_hash_ids_hcu, supports_block


def _hash(
    input_ids, positions, *, mode, history, token_map, multipliers, primes,
    offsets, pad_id, num_real=None, req_slots=None, block=1, row=None,
    starts=None, image_token_id=None, mm_pad_shift=0, block_t=32,
):
    if not (
        envs.SGLANG_HCU_OPT_ENGRAM_HASH_IDS.get()
        and input_ids.is_cuda
        and mode in (MODE_EXTEND, MODE_VERIFY)
        and multipliers.shape == (2, 4)
        and primes.shape == (2, 3, 8)
        and offsets.shape == (2, 24)
        and input_ids.ndim == 1
        and positions.shape == input_ids.shape
        and history.ndim == 2
        and history.shape[1] == 3
    ):
        return None
    # CP hash inputs stay global. The model slices the resulting rows by rank.
    num_real = input_ids.numel() if num_real is None else num_real
    if num_real > input_ids.numel():
        raise ValueError("Engram CP hashing requires all tokens before CP sharding")
    op, supports_block = _provider()
    if op is None or (mode == MODE_VERIFY and not supports_block):
        return None
    tables = (token_map, multipliers, primes, offsets)
    if mode == MODE_EXTEND:
        if row is None or starts is None or row.shape != (num_real,):
            return None
        tables += (row, starts)
    if any(t.dtype != torch.int64 for t in tables):
        return None
    ids = (input_ids, positions, history)
    if any(t.dtype not in (torch.int32, torch.int64) for t in ids):
        return None
    if any(t.device != input_ids.device or not t.is_contiguous() for t in (*tables, *ids)):
        return None
    # LightOp expects per-request history, even when the source is a slot table.
    if req_slots is not None:
        history = history.index_select(0, req_slots)
    args = (
        input_ids, positions, history, row, starts, token_map, multipliers,
        primes, offsets, num_real, pad_id, image_token_id, mm_pad_shift,
    )
    return op(*args, block=block) if mode == MODE_VERIFY else op(*args)


def around_hash_ids(original_fn, input_ids, positions, **kwargs):
    result = _hash(input_ids, positions, **kwargs)
    if result is None:
        return original_fn(input_ids, positions, **kwargs)
    hashes, tokens = result
    # The kernel seam returns int32 predecessors; LightOp and the torch method use int64.
    return hashes, tokens.to(torch.int32)


def around_torch_hash_ids(
    original_fn, owner, input_ids, positions, kmode, history, num_real, block, row, starts,
):
    result = _hash(
        input_ids, positions, mode=kmode, history=history, num_real=num_real,
        block=block, row=row, starts=starts, token_map=owner.token_map,
        multipliers=owner.multipliers, primes=owner.primes, offsets=owner.offsets,
        pad_id=owner.pad_id, image_token_id=owner.image_token_id, mm_pad_shift=MM_PAD_SHIFT_VALUE,
    )
    if result is None:
        return original_fn(owner, input_ids, positions, kmode, history, num_real, block, row, starts)
    return result


def around_gate(original_fn, x, kv, q_weight, k_weight, eps, clamp_value):
    from sglang.srt.layers import engram

    if (
        not engram._cuda_kernels(x)
        and envs.SGLANG_HCU_OPT_ENGRAM_GATE.get()
        and x.is_cuda
        and x.ndim == 3
        and x.dtype == kv.dtype
        and all(
            t.device == x.device and t.is_contiguous() and t.dtype in (torch.bfloat16, torch.float32)
            for t in (x, kv, q_weight, k_weight)
        )
    ):
        return engram.fused_engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)
    return original_fn(x, kv, q_weight, k_weight, eps, clamp_value)
