# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_prefill(
    Q, KV, IDS, LENGTHS, SINK, OUT, MAXIMUM, LSE,
    Q_STRIDE_0: tl.constexpr, Q_STRIDE_1: tl.constexpr,
    KV_STRIDE_0: tl.constexpr, IDX_STRIDE_0: tl.constexpr,
    HEADS: tl.constexpr, TOPK: tl.constexpr, KV_ROWS: tl.constexpr,
    SCALE: tl.constexpr, HAS_LENGTHS: tl.constexpr, HAS_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, D)
    query = tl.load(Q + row * Q_STRIDE_0 + heads[:, None] * Q_STRIDE_1 + dims[None, :], heads[:, None] < HEADS, other=0)
    length = TOPK
    if HAS_LENGTHS:
        length = tl.load(LENGTHS + row)
    maximum = tl.full((BLOCK_H,), float('-inf'), tl.float32)
    denominator = tl.full((BLOCK_H,), 0.0, tl.float32)
    accumulator = tl.full((BLOCK_H, D), 0.0, tl.float32)
    for block in range(tl.cdiv(TOPK, BLOCK_N)):
        slots = block * BLOCK_N + tl.arange(0, BLOCK_N)
        indices = tl.load(IDS + row * IDX_STRIDE_0 + slots, slots < TOPK, other=-1)
        valid = (slots < TOPK) & (slots < length) & (indices >= 0) & (indices < KV_ROWS)
        keys = tl.load(KV + indices[:, None] * KV_STRIDE_0 + dims[None, :], valid[:, None], other=0)
        scores = tl.dot(query, tl.trans(keys)).to(tl.float32) * (SCALE * 1.4426950408889634)
        scores = tl.where(valid[None, :], scores, float('-inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(new_maximum == float('-inf'), 0.0, new_maximum)
        correction = tl.exp2(maximum - safe_maximum)
        probability = tl.exp2(scores - safe_maximum[:, None])
        accumulator = accumulator * correction[:, None] + tl.dot(probability.to(query.dtype), keys)
        denominator = denominator * correction + tl.sum(probability, 1)
        maximum = new_maximum
    real_lse = maximum * 0.6931471805599453 + tl.log(denominator)
    if HAS_SINK:
        sink = tl.load(SINK + heads, heads < HEADS, other=0) * 1.4426950408889634
        positive_sink = sink == float('inf')
        sink = tl.where(positive_sink, 0.0, sink)
        norm_maximum = tl.maximum(maximum, sink)
        norm_maximum = tl.where(norm_maximum == float('-inf'), 0.0, norm_maximum)
        correction = tl.exp2(maximum - norm_maximum)
        denominator = denominator * correction + tl.exp2(sink - norm_maximum)
        accumulator = accumulator * correction[:, None]
        accumulator = tl.where(positive_sink[:, None], 0.0, accumulator)
    denominator = tl.where(denominator == 0.0, 1.0, denominator)
    value = accumulator / denominator[:, None]
    tl.store(OUT + row * HEADS * D + heads[:, None] * D + dims[None, :], value, heads[:, None] < HEADS)
    tl.store(MAXIMUM + row * HEADS + heads, maximum * 0.6931471805599453, heads < HEADS)
    tl.store(LSE + row * HEADS + heads, real_lse, heads < HEADS)


def sparse_prefill(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None):
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError('Sparse prefill requires three-dimensional query, KV and indices')
    rows, heads, dim = q.shape
    if dim != 512 or d_v != 512 or not 1 <= heads <= 64 or kv.shape[1:] != (1, 512):
        raise ValueError('HCU sparse prefill supports head_dim=value_dim=512, KV heads=1 and query heads<=64')
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or indices.dtype != torch.int32:
        raise TypeError('HCU sparse prefill requires BF16 query/KV and INT32 indices')
    if indices.shape[:2] != (rows, 1) or indices.stride(2) != 1 or q.stride(2) != 1 or kv.stride(2) != 1:
        raise ValueError('Unsupported sparse-prefill shape or inner stride')
    if not q.is_cuda or q.device != kv.device or q.device != indices.device:
        raise ValueError('Sparse-prefill inputs must share a CUDA-facade device')
    if topk_length is not None:
        if topk_length.dtype != torch.int32 or topk_length.numel() != rows or topk_length.device != q.device:
            raise ValueError('Sparse-prefill lengths must be device INT32 values, one per query')
        topk_length = topk_length.reshape(-1).contiguous()
    if attn_sink is not None:
        if attn_sink.dtype != torch.float32 or attn_sink.numel() != heads or attn_sink.device != q.device:
            raise ValueError('Sparse-prefill sink must be device FP32, one per head')
        attn_sink = attn_sink.reshape(-1).contiguous()
    output = torch.empty((rows, heads, 512), dtype=q.dtype, device=q.device)
    maximum = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    lse = torch.empty_like(maximum)
    if rows == 0:
        return output, maximum, lse
    block_h = 64 if heads == 64 else 16
    num_warps = 8 if heads == 64 else 4
    block_n = 32 if heads == 64 else 64
    _sparse_prefill[(rows, triton.cdiv(heads, block_h))](
        q, kv, indices, q if topk_length is None else topk_length,
        q if attn_sink is None else attn_sink, output, maximum, lse,
        q.stride(0), q.stride(1), kv.stride(0), indices.stride(0),
        heads, indices.shape[-1], kv.shape[0], float(sm_scale),
        topk_length is not None, attn_sink is not None,
        BLOCK_H=block_h, BLOCK_N=block_n, D=512, num_warps=num_warps, num_stages=1,
    )
    return output, maximum, lse
