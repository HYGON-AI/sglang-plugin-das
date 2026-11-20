# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import lru_cache

import torch
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.runtime_context import get_forward

from hcu_sglang.utils.detection import get_gcn_arch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _is_gfx938(device_index):
    return get_gcn_arch(device_index).split(":", 1)[0] == "gfx938"


def _supports_fusion(layer, hidden_states):
    proj = layer.o_proj
    scheme = getattr(proj, "scheme", None)
    return (
        layer.attn_output_gate
        and type(scheme) is CompressedTensorsW8A8Fp8
        and getattr(scheme, "supports_fp8_prequantized_input", False)
        and getattr(getattr(proj, "weight", None), "dtype", None) == torch.float8_e4m3fn
        and proj.input_is_parallel
        and hidden_states.numel() > 0
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and _is_gfx938(hidden_states.device.index)
    )


def _compatible_layout(attn_output, gate, width):
    return (
        attn_output.ndim == 2
        and attn_output.shape[0] > 0
        and attn_output.shape[1] == width
        and attn_output.dtype == torch.bfloat16
        and attn_output.is_contiguous()
        and gate.dtype == attn_output.dtype
        and gate.device == attn_output.device
        and gate.stride(-1) == 1
        and (
            gate.shape == attn_output.shape
            or (
                gate.ndim == 3
                and gate.shape[0] == attn_output.shape[0]
                and gate.shape[1] * gate.shape[2] == width
            )
        )
    )


@lru_cache(maxsize=1)
def _kernel():
    try:
        from lightop.activation import fuse_sigmoid_mul_fp8_quant
    except ImportError:
        from hcu_sglang.layers.attention.sigmoid_mul_fp8 import (
            fuse_sigmoid_mul_fp8_quant,
        )
    return fuse_sigmoid_mul_fp8_quant


@lru_cache(maxsize=None)
def _log_dispatch(mode, provider):
    logger.info("Qwen attention sigmoid mul + FP8 fused before o_proj (%s, %s)", mode, provider)


def around_self_attention(original, self, positions, hidden_states, forward_batch):
    if (
        torch.compiler.is_compiling()
        or not _supports_fusion(self, hidden_states)
        or get_forward().sp_active
    ):
        return original(self, positions, hidden_states, forward_batch)

    from sglang.srt.models import qwen3_5 as native

    q, k, v, gate = self._prepare_qkv_gate(
        positions=positions, hidden_states=hidden_states, forward_batch=forward_batch
    )
    attn_output = self.attn(q, k, v, forward_batch)
    proj = self.o_proj
    if native._fp8_tuple_input(proj, attn_output.shape[0]):
        return proj(native.fused_sigmoid_mul(attn_output, gate, quant=True))[0]
    if _compatible_layout(attn_output, gate, proj.input_size_per_partition):
        kernel = _kernel()
        quantized, scale = kernel(attn_output, gate)
        _log_dispatch(forward_batch.forward_mode.name, kernel.__module__)
        return proj((quantized, scale, attn_output.dtype))[0]

    return proj(native.fused_sigmoid_mul(attn_output, gate, inplace=True))[0]
