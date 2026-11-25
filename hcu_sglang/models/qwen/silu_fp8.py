# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import lru_cache

import torch
from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_exec, get_forward

from hcu_sglang.utils.detection import get_gcn_arch, is_hcu

logger = logging.getLogger(__name__)
_TARGET = "sglang.srt.models.qwen2_moe.Qwen2MoeMLP.forward"


def register():
    if envs.SGLANG_USE_FUSED_SILU_MUL_FP8_QUANT.get() and is_hcu():
        HookRegistry.register(_TARGET, around_forward, HookType.AROUND)


@lru_cache(maxsize=None)
def _is_gfx938(device_index):
    return get_gcn_arch(device_index).split(":", 1)[0] == "gfx938"


def around_forward(original, self, x):
    proj = self.down_proj
    supports = getattr(proj, "supports_fused_silu_mul_fp8_quant_input", None)
    if (
        torch.compiler.is_compiling()
        or self._enable_silu_fp4_quant_fusion
        or supports is None
        or not supports()
        or getattr(getattr(proj, "weight", None), "dtype", None) != torch.float8_e4m3fn
        or not proj.input_is_parallel
        or proj.input_size_per_partition % 32
        or not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or not x.shape[0]
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not _is_gfx938(x.device.index)
        or get_forward().sp_active
        or get_exec().deterministic.rl_on_policy_target is not None
    ):
        return original(self, x)

    gate_up, _ = self.gate_up_proj(x)
    if (
        isinstance(gate_up, torch.Tensor)
        and gate_up.shape == (x.shape[0], 2 * proj.input_size_per_partition)
        and gate_up.device == x.device
        and gate_up.dtype == torch.bfloat16
        and gate_up.is_contiguous()
        and gate_up.storage_offset() % 32 == 0
    ):
        output, _ = proj(gate_up, use_fused_silu_mul_fp8_quant=True)
        _log_dispatch(proj.input_size_per_partition)
        return output

    output, _ = proj(self.act_fn(gate_up))
    return output


@lru_cache(maxsize=None)
def _log_dispatch(width):
    logger.info("Qwen MLP LightOp SiLU mul + FP8 fused before down_proj (width=%d)", width)
