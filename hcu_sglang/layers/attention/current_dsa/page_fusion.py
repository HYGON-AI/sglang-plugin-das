"""Guarded full-width graph-page gather and sixfold repeat for HCU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_repeat(
    src,
    indices,
    dst,
    SRC_STRIDE: tl.constexpr,
    SRC_ROWS: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(indices + row)
    index = tl.where(index < 0, index + SRC_ROWS, index)
    value = tl.load(src + index * SRC_STRIDE + col, mask=col < WIDTH, other=0)
    for repeat in range(6):
        tl.store(dst + (row * 6 + repeat) * DST_STRIDE + col, value, mask=col < WIDTH)


def try_gather_repeat(src, indices, dst, bs, repeats, width):
    """Use only graph-pool indices known in range by the caller's allocator.

    Invalid index error behavior is not emulated; the callsite must establish
    req_pool_indices are valid graph-pool indices, including padded row zero.
    """
    if not (
        type(bs) is int
        and 1 <= bs <= 5
        and repeats == 6
        and width == 1048651
        and isinstance(src, torch.Tensor)
        and isinstance(indices, torch.Tensor)
        and isinstance(dst, torch.Tensor)
        and src.is_cuda
        and torch.version.hip is not None
        and indices.device == src.device
        and dst.device == src.device
        and src.dtype == dst.dtype == torch.int32
        and indices.dtype == torch.int64
        and src.ndim == 2
        and src.shape[1] == width
        and src.is_contiguous()
        and indices.shape == (bs,)
        and indices.is_contiguous()
        and dst.shape == (bs * repeats, width)
        and dst.is_contiguous()
        and src.untyped_storage().data_ptr() != dst.untyped_storage().data_ptr()
        and indices.untyped_storage().data_ptr() != dst.untyped_storage().data_ptr()
    ):
        return False
    _gather_repeat[(bs, triton.cdiv(width, 1024))](
        src,
        indices,
        dst,
        src.stride(0),
        src.shape[0],
        dst.stride(0),
        width,
        1024,
        num_warps=4,
    )
    return True
