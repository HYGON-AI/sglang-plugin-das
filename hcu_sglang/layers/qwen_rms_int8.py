import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache

import torch
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from sglang.srt.utils import get_bool_env_var

from hcu_sglang.layers.fused_rms_quant import supports_fused_rms_quant_input
from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)
enabled = get_bool_env_var("SGLANG_HCU_QWEN_RMS_INT8_FUSION") and is_hcu()


@dataclass
class _DecoderInputs:
    consumers: dict
    pending: dict = field(default_factory=dict)


_inputs = ContextVar("qwen_rms_int8_inputs", default=None)


def _accepts_fusion(layer):
    scheme = getattr(layer, "scheme", None)
    return (
        type(scheme) is CompressedTensorsW8A8Int8
        and not scheme.is_static_input_scheme
        and scheme.input_symmetric
    )


def around_decoder_forward(original, self, *args, **kwargs):
    if torch.compiler.is_compiling():
        return original(self, *args, **kwargs)
    gdn = getattr(self, "linear_attn", None)
    attn = self.qkv_proj if gdn is None else gdn.in_proj_qkvz
    ffn = getattr(self.mlp, "gate_up_proj", None)
    consumers = {
        id(norm): (id(proj), name)
        for norm, proj, name in (
            (self.input_layernorm, attn, "qkv_proj" if gdn is None else "in_proj_qkvz"),
            (self.post_attention_layernorm, ffn, "gate_up_proj"),
        )
        if _accepts_fusion(proj)
    }
    if not consumers:
        return original(self, *args, **kwargs)
    token = _inputs.set(_DecoderInputs(consumers))
    try:
        return original(self, *args, **kwargs)
    finally:
        _inputs.reset(token)


def try_fused_norm(norm, x, residual):
    scope = _inputs.get()
    if scope is None:
        return None
    consumer = scope.consumers.get(id(norm))
    if consumer is None:
        return None
    weight = norm.gemma_weight
    if not supports_fused_rms_quant_input(x, weight, residual):
        return None

    from lightop import rms_norm_dynamic_per_token_quant_sglang

    out = x if residual is not None else torch.empty_like(x)
    quant = rms_norm_dynamic_per_token_quant_sglang(
        input=x,
        weight=weight,
        epsilon=norm.variance_epsilon,
        residual=residual,
        update_input=residual is not None,
        norm_output=out if residual is None else None,
    )
    projection, name = consumer
    scope.pending[projection] = (out, quant, name)
    return (out, residual) if residual is not None else out


@lru_cache(maxsize=None)
def _log_consumer(name):
    logger.info("Qwen LightOP RMSNorm + dynamic INT8 quant consumed by %s", name)


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
