# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import lru_cache

import torch
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from sglang.srt.runtime_context import get_forward
from sglang.srt.utils import get_bool_env_var

from hcu_sglang.layers.attention.sigmoid_mul_int8 import sigmoid_mul_per_token_int8_quant
from hcu_sglang.layers.quantization.compressed_int8 import PrequantizedInt8Input
from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)
enabled = get_bool_env_var("SGLANG_HCU_QWEN_ATTN_SIGMOID_INT8_FUSION") and is_hcu()


@lru_cache(maxsize=None)
def _log_dispatch(mode):
    logger.info("Qwen full attention sigmoid mul + INT8 fused before o_proj (%s)", mode)


def around_self_attention(original, self, positions, hidden_states, forward_batch):
    proj = self.o_proj
    scheme = getattr(proj, "scheme", None)
    if (
        torch.compiler.is_compiling()
        or not self.attn_output_gate
        or type(scheme) is not CompressedTensorsW8A8Int8
        or scheme.is_static_input_scheme
        or not scheme.input_symmetric
        or not proj.input_is_parallel
        or get_forward().sp_active
    ):
        return original(self, positions, hidden_states, forward_batch)

    q, k, v, gate = self._prepare_qkv_gate(
        positions=positions, hidden_states=hidden_states, forward_batch=forward_batch
    )
    attn = self.attn(q, k, v, forward_batch)
    if (
        attn.ndim == 2
        and attn.shape[0]
        and attn.shape[1] == proj.input_size_per_partition
        and attn.is_cuda
        and attn.dtype == torch.bfloat16
        and attn.is_contiguous()
        and gate.dtype == attn.dtype
        and gate.device == attn.device
        and gate.stride(-1) == 1
        and (
            gate.shape == attn.shape
            or (
                gate.ndim == 3
                and gate.shape[0] == attn.shape[0]
                and gate.shape[1] * gate.shape[2] == attn.shape[1]
            )
        )
    ):
        quantized, scale = sigmoid_mul_per_token_int8_quant(attn, gate)
        _log_dispatch(forward_batch.forward_mode.name)
        return proj(PrequantizedInt8Input(quantized, scale, attn.dtype))[0]

    return proj(fused_sigmoid_mul(attn, gate, inplace=True))[0]
