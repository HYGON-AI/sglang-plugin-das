# SPDX-License-Identifier: Apache-2.0
"""Native BF16-rounded SwiGLU fused with per-row INT8 quantization."""

import threading

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from hcu_sglang.multimodal_gen import minimax_h3_quant_env as envs
from hcu_sglang.multimodal_gen.minimax_h3_lightop_gemm import (
    lightop_gemm_enabled,
    try_lightop_gemm,
)
from hcu_sglang.multimodal_gen.minimax_h3_quant import (
    MAX_ROW_QUANT_LINEAR_OFFSET,
    MiniMaxH3Int8LinearMethod,
    _row_quant_block_size,
)


# Enumerate every BF16 bit pattern, not a sampled approximation of SiLU.
_BF16_PATTERN_COUNT = 1 << (torch.finfo(torch.bfloat16).bits)
_tables = {}
_tables_lock = threading.Lock()


def _silu_table(device):
    """Cache one 128-KiB table per device, with cross-stream readiness.

    Initialize during eager warmup. A cold graph capture takes the native
    path rather than publishing a table whose initialization needs replay.
    """
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        with _tables_lock:
            entry = _tables.get(device.index)
            if entry is None:
                if torch.cuda.is_current_stream_capturing():
                    return None
                patterns = torch.arange(
                    _BF16_PATTERN_COUNT, device=device, dtype=torch.int32
                ).to(torch.int16)
                from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _silu_mul

                gates = patterns.view(torch.bfloat16).view(-1, 8)
                packed = torch.cat((gates, torch.ones_like(gates)), dim=-1)
                table = _silu_mul(packed, reuse_input=False).reshape(-1)
                ready = torch.cuda.Event()
                ready.record(stream)
                entry = (table, ready, stream.cuda_stream)
                _tables[device.index] = entry
        table, ready, producer = entry
        if stream.cuda_stream != producer:
            stream.wait_event(ready)
        return table


def _swiglu_quant_block_size(x):
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[0] < 1
        or x.shape[1] < 2
        or x.shape[1] % 2
        or x.numel() - 1 > MAX_ROW_QUANT_LINEAR_OFFSET
    ):
        return None
    # Reuse the existing row reduction and padded-output indexing limits.
    return _row_quant_block_size(x[:, : x.shape[1] // 2], "int8")


@triton.jit
def _swiglu_quant_rows(
    X, TABLE, Q, SX, M: tl.constexpr, K: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = (row < M) & (col < K)
    gate = tl.load(X + row * (2 * K) + col, mask, 0)
    up = tl.load(X + row * (2 * K) + K + col, mask, 0).to(tl.float32)
    bits = gate.to(tl.uint16, bitcast=True).to(tl.int32)
    activation = tl.load(TABLE + bits).to(tl.float32)
    # The lookup retains SiLU's BF16 rounding; retain product rounding too,
    # before computing absmax and the existing ties-to-even quantization.
    product = (activation * up).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(product), 0), 1.0e-12)
    sx = amax * (1.0 / 127.0)
    quant = libdevice.nearbyint(tl.div_rn(product, sx))
    quant = tl.minimum(tl.maximum(quant, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + col, quant, col < K)
    tl.store(SX + row, sx)


def quant_swiglu_rows(x):
    """Return padded INT8 rows/scales, or None for unsupported/cold capture.

    Shares the native SiLU table's device, stream and graph-capture lifetime.
    Padding and launch parameters match quant_rows(), independently of H3 sizes.
    """
    block_size = _swiglu_quant_block_size(x)
    if block_size is None:
        return None
    with torch.cuda.device(x.device):
        table = _silu_table(x.device)
        if table is None:
            return None
        m, width = x.shape
        k = width // 2
        # Same _int_mm padding contract as quant_rows(), including small M.
        padded_m = max(32, triton.cdiv(m, 16) * 16)
        q = torch.empty((padded_m, k), device=x.device, dtype=torch.int8)
        sx = torch.empty((padded_m, 1), device=x.device, dtype=torch.float32)
        _swiglu_quant_rows[(padded_m,)](
            x,
            table,
            q,
            sx,
            m,
            k,
            block_size,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return q, sx


def _eligible_fc2(layer, x):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3MLP

    if (
        type(layer) is not MiniMaxH3MLP
        or layer.reuse_fc1_activation
        or not isinstance(x, torch.Tensor)
        or not x.is_cuda
        or x.ndim != 2
        or x.dtype != torch.bfloat16
        or not envs.SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_LIGHTOP_GEMM_WHOLE
    ):
        return None

    fc2 = layer.fc2
    if (
        type(fc2) is not RowParallelLinear
        or fc2.tp_size != 1
        or fc2.tp_rank != 0
        or not fc2.input_is_parallel
        or fc2.bias is not None
        or fc2._forward_pre_hooks
        or fc2._forward_hooks
        or type(fc2.quant_method) is not MiniMaxH3Int8LinearMethod
        or not fc2.quant_method.fused
        or fc2.weight.device != x.device
        or not lightop_gemm_enabled(fc2.weight)
    ):
        return None
    return fc2


def around_mlp_forward(original, layer, x):
    fc2 = _eligible_fc2(layer, x)
    if fc2 is None:
        return original(layer, x)

    hidden, _ = layer.fc1(x)
    if hidden.ndim == 2 and hidden.shape[1] == 2 * fc2.weight.shape[1]:
        quantized = quant_swiglu_rows(hidden)
        if quantized is not None:
            q, sx = quantized
            result = try_lightop_gemm(q, fc2.weight, sx, fc2.weight_scale_inv)
            if result is not None:
                return result[: hidden.shape[0]]

    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _silu_mul

    activation = _silu_mul(hidden, reuse_input=layer.reuse_fc1_activation)
    result, _ = fc2(activation)
    return result
