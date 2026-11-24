# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache

import torch
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.utils import get_bool_env_var

from hcu_sglang.utils.detection import get_gcn_arch, is_hcu

logger = logging.getLogger(__name__)
enabled = get_bool_env_var("SGLANG_HCU_QWEN_RMS_FP8_FUSION") and is_hcu()


@dataclass
class _DecoderInputs:
    consumers: dict
    pending: dict = field(default_factory=dict)


_inputs = ContextVar("qwen_rms_fp8_inputs", default=None)


@lru_cache(maxsize=1)
def _kernel():
    try:
        from lightop import gemma_rms_norm_per_token_fp8_quant, op
    except ImportError:
        return None
    if not hasattr(op, "gemma_rms_norm_fp8_quant"):
        return None
    return gemma_rms_norm_per_token_fp8_quant


@lru_cache(maxsize=None)
def _is_gfx938(device_index):
    return get_gcn_arch(device_index).split(":", 1)[0] == "gfx938"


def _accepts_fusion(layer):
    scheme = getattr(layer, "scheme", None)
    weight = getattr(layer, "weight", None)
    return (
        type(scheme) is CompressedTensorsW8A8Fp8
        and getattr(scheme, "supports_fp8_prequantized_input", False)
        and getattr(weight, "dtype", None) == torch.float8_e4m3fn
    )


def around_decoder_forward(original, self, *args, **kwargs):
    if (
        torch.compiler.is_compiling()
        or get_parallel().tp_size != 1
        or get_forward().sp_active
        or _kernel() is None
    ):
        return original(self, *args, **kwargs)

    gdn = getattr(self, "linear_attn", None)
    attn = self.qkv_proj if gdn is None else gdn.in_proj_qkvz
    ffn = getattr(self.mlp, "gate_up_proj", None)
    consumers = {}
    if _accepts_fusion(attn) and getattr(gdn, "_fused_in_proj_weight", None) is None:
        consumers[id(self.input_layernorm)] = (
            id(attn),
            "qkv_proj" if gdn is None else "in_proj_qkvz",
        )
    if _accepts_fusion(ffn):
        consumers[id(self.post_attention_layernorm)] = (id(ffn), "gate_up_proj")
    if not consumers:
        return original(self, *args, **kwargs)

    token = _inputs.set(_DecoderInputs(consumers))
    try:
        return original(self, *args, **kwargs)
    finally:
        _inputs.reset(token)


def _supported_layout(x, weight, residual):
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(residual, torch.Tensor)
        and x.is_cuda
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == 5120
        and x.dtype == torch.bfloat16
        and residual.shape == x.shape
        and weight.shape == (5120,)
        and all(
            t.dtype == x.dtype
            and t.device == x.device
            and t.is_contiguous()
            and t.data_ptr() % 32 == 0
            for t in (x, residual, weight)
        )
        and _is_gfx938(x.device.index)
    ):
        return False
    return abs(x.data_ptr() - residual.data_ptr()) >= x.numel() * x.element_size()


def try_fused_norm(norm, x, residual):
    scope = _inputs.get()
    if scope is None:
        return None
    consumer = scope.consumers.get(id(norm))
    if consumer is None:
        return None

    weight = norm.weight.data
    if not _supported_layout(x, weight, residual):
        return None

    kernel = _kernel()
    q, scale = kernel(
        x,
        weight,
        norm.variance_epsilon,
        fp8type=0,
        residual=residual,
        update_input=True,
    )
    projection, name = consumer
    scope.pending[projection] = (x, (q, scale, x.dtype), name)
    return x, residual


@lru_cache(maxsize=None)
def _log_consumer(name):
    logger.info("Qwen LightOp Add + Gemma RMSNorm + FP8 consumed by %s", name)


def take_prequantized_input(layer, x):
    scope = _inputs.get()
    if scope is None:
        return None
    pending = scope.pending.pop(id(layer), None)
    if pending is None:
        return None
    normalized, quant, name = pending
    if normalized is not x:
        return None
    _log_consumer(name)
    return quant
