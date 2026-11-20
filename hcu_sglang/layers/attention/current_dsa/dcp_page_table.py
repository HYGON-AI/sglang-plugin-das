# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import accumulate
from typing import List, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode as _decode,
    transform_index_page_table_prefill as _prefill,
)


def _allocate_prefill_result(
    topk_indices: torch.Tensor, real_num_tokens: int, output_num_tokens: Optional[int]
) -> torch.Tensor:
    topk_num_tokens = topk_indices.shape[0]
    if output_num_tokens is None:
        output_num_tokens = topk_num_tokens
    assert real_num_tokens <= topk_num_tokens, (
        f"sum(extend_lens_cpu) ({real_num_tokens}) exceeds topk_indices rows ({topk_num_tokens})"
    )
    assert topk_num_tokens <= output_num_tokens, (
        f"topk_indices rows ({topk_num_tokens}) exceeds output_num_tokens ({output_num_tokens})"
    )
    result = torch.empty(
        (output_num_tokens, topk_indices.shape[1]),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if real_num_tokens < output_num_tokens:
        result[real_num_tokens:].fill_(-1)
    return result


@triton.jit
def transform_index_page_table_decode_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_size: tl.constexpr,
    page_table_row_stride: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
):
    TOPK: tl.constexpr = 2048
    req_id = tl.program_id(0)
    page_table_ptr = page_table_ptr + req_id * page_table_row_stride
    topk_indices_ptr = topk_indices_ptr + req_id * TOPK
    result_ptr = result_ptr + req_id * TOPK
    offset = tl.arange(0, TOPK)
    loaded_topk_indices = tl.load(topk_indices_ptr + offset)
    mask = loaded_topk_indices >= 0
    loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
    if dcp_size > 1:
        mask = mask & (loaded_kv_indices % dcp_size == dcp_rank)
        loaded_kv_indices = loaded_kv_indices // dcp_size
    tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
    tl.store(result_ptr + offset, -1, mask=~mask)


@triton.jit
def transform_index_page_table_prefill_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    cu_seqlens_q_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_table_stride_0: tl.constexpr,
    page_table_stride_1: tl.constexpr,
    topk_indices_stride_0: tl.constexpr,
    topk_indices_stride_1: tl.constexpr,
    result_stride_0: tl.constexpr,
    result_stride_1: tl.constexpr,
    PAGE_TABLE_IS_EXPANDED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
):
    request_id = tl.program_id(0)
    query_offsets = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    topk_offsets = tl.program_id(2) * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    query_start = tl.load(cu_seqlens_q_ptr + request_id)
    query_end = tl.load(cu_seqlens_q_ptr + request_id + 1)
    if query_start + tl.program_id(1) * BLOCK_Q >= query_end:
        return
    token_indices = query_start + query_offsets
    mask = (token_indices[:, None] < query_end) & (topk_offsets[None, :] < TOPK)
    loaded_topk_indices = tl.load(
        topk_indices_ptr
        + token_indices[:, None] * topk_indices_stride_0
        + topk_offsets[None, :] * topk_indices_stride_1,
        mask=mask,
        other=-1,
    )
    valid_topk_mask = mask & (loaded_topk_indices >= 0)
    if PAGE_TABLE_IS_EXPANDED:
        page_table_rows = token_indices
    else:
        page_table_rows = token_indices * 0 + request_id
    loaded_kv_indices = tl.load(
        page_table_ptr
        + page_table_rows[:, None] * page_table_stride_0
        + loaded_topk_indices * page_table_stride_1,
        mask=valid_topk_mask,
        other=-1,
    )
    if dcp_size > 1:
        owned_mask = valid_topk_mask & (loaded_kv_indices % dcp_size == dcp_rank)
        loaded_kv_indices = tl.where(owned_mask, loaded_kv_indices // dcp_size, -1)
    tl.store(
        result_ptr
        + token_indices[:, None] * result_stride_0
        + topk_offsets[None, :] * result_stride_1,
        loaded_kv_indices,
        mask=mask,
    )


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
    dcp_size: int = 1,
    dcp_rank: int = 0,
) -> torch.Tensor:
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048
    qo_len = topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
        page_table,
        topk_indices,
        result,
        page_size,
        page_table_row_stride=page_table.stride(0),
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
    output_num_tokens: Optional[int] = None,
    page_table_is_expanded: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    dcp_size: int = 1,
    dcp_rank: int = 0,
) -> torch.Tensor:
    assert page_size == 1
    assert topk_indices.shape[1] == 2048
    real_num_tokens = sum(extend_lens_cpu)
    result = _allocate_prefill_result(topk_indices, real_num_tokens, output_num_tokens)
    if real_num_tokens == 0:
        return result
    max_extend_len = max(extend_lens_cpu)
    block_q = 1 if max_extend_len == 1 else 2 if max_extend_len == 2 else 4
    block_topk = 256
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor(
            [0, *accumulate(extend_lens_cpu)],
            dtype=torch.int32,
            device=topk_indices.device,
        )
    grid = (
        cu_seqlens_q.shape[0] - 1,
        triton.cdiv(max_extend_len, block_q),
        triton.cdiv(topk_indices.shape[1], block_topk),
    )
    transform_index_page_table_prefill_kernel[grid](
        page_table,
        topk_indices,
        cu_seqlens_q,
        result,
        page_table.stride(0),
        page_table.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        result.stride(0),
        result.stride(1),
        PAGE_TABLE_IS_EXPANDED=page_table_is_expanded,
        TOPK=topk_indices.shape[1],
        BLOCK_Q=block_q,
        BLOCK_TOPK=block_topk,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        num_warps=4,
    )
    return result


def transform_index_page_table_decode(*, dcp_size=1, dcp_rank=0, **kwargs):
    if dcp_size == 1:
        return _decode(**kwargs)
    return transform_index_page_table_decode_fast(
        dcp_size=dcp_size, dcp_rank=dcp_rank, **kwargs
    )


def transform_index_page_table_prefill(*, dcp_size=1, dcp_rank=0, **kwargs):
    if dcp_size == 1:
        return _prefill(**kwargs)
    return transform_index_page_table_prefill_fast(
        dcp_size=dcp_size, dcp_rank=dcp_rank, **kwargs
    )
