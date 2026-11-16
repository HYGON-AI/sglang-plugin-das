from __future__ import annotations
from functools import cache


from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_JIT_SOURCE = Path(__file__).parents[2] / "jit_overlay" / "csrc"


def load_jit(*args, cuda_files, **kwargs):
    from sglang.kernels.jit.utils import load_jit as upstream_load_jit

    return upstream_load_jit(
        *args,
        cuda_files=[str(_JIT_SOURCE / name) for name in cuda_files],
        **kwargs,
    )


@cache
def _jit_copy_planned_module(
    block_size: int,
    is_mla: bool,
    is_dsv4_layout: bool,
    skip_io: bool,
) -> Module:
    template_args = make_cpp_args(block_size, is_mla, is_dsv4_layout, skip_io)
    return load_jit(
        "minimax_sparse_copy_planned",
        block_size,
        is_mla,
        is_dsv4_layout,
        skip_io,
        cuda_files=["kvcacheio/minimax_hisparse.cuh"],
        cuda_wrappers=[
            (
                "copy_cache_planned",
                f"copy_cache_planned<{template_args}>",
            )
        ],
    )


@cache_once
def _jit_commit_mha_all_layers() -> Module:
    return load_jit(
        "hisparse_commit_mha",
        256,
        cuda_files=["kvcacheio/minimax_hisparse.cuh"],
        cuda_wrappers=[("commit", "commit_mha_all_layers<256>")],
    )


def commit_mha_all_layers(
    scratch: torch.Tensor,
    host_locs: torch.Tensor,
    accepted: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    host_k: torch.Tensor,
    host_v: torch.Tensor,
    item_size_bytes: int,
) -> None:
    """Back up accepted verify tokens into contiguous layer-first host K/V."""
    _jit_commit_mha_all_layers().commit(
        scratch,
        host_locs,
        accepted,
        source_k,
        source_v,
        host_k,
        host_v,
        item_size_bytes,
    )


@cache_once
def _jit_swap_mha_pages_tiled(
    block_size: int, blocks_per_request: int, heads_per_request: int
) -> Module:
    return load_jit(
        "hisparse_swap_mha_pages",
        block_size,
        blocks_per_request,
        heads_per_request,
        cuda_files=["kvcacheio/minimax_hisparse.cuh"],
        cuda_wrappers=[
            ("swap", f"swap_mha_pages_tiled<{block_size}, {blocks_per_request}, {heads_per_request}>")
        ],
    )


def swap_mha_pages_tiled(
    src_locs,
    dst_locs,
    counts,
    real,
    host_k,
    host_v,
    device_k,
    device_v,
    block_size: int,
    blocks_per_request: int,
    heads_per_request: int = 1,
) -> None:
    """Swap MiniMax 64 KiB K pages and matching V pages from mapped host KV."""
    _jit_swap_mha_pages_tiled(block_size, blocks_per_request, heads_per_request).swap(
        src_locs, dst_locs, counts, real, host_k, host_v, device_k, device_v
    )


def copy_cache_planned_mha(
    src_locs: torch.Tensor,
    dst_locs: torch.Tensor,
    counts: torch.Tensor,
    num_real_reqs: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    item_size_bytes: int,
) -> None:
    """Copy packed MHA plans, in page or token units, between mapped pools."""
    module = _jit_copy_planned_module(256, False, False, False)
    module.copy_cache_planned(
        src_locs,
        dst_locs,
        counts,
        num_real_reqs,
        src_k,
        src_v,
        dst_k,
        dst_v,
        128,
        item_size_bytes,
    )
