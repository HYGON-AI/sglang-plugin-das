"""HCU compatibility for the diffusion QK-Norm + RoPE JIT kernel."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import get_gcn_arch, is_hcu


_H3_FUSED_PROBE = ContextVar("hcu_minimax_h3_fused_qknorm_rope_probe", default=False)
_H3_4HEADS_ACTIVE = ContextVar("hcu_minimax_h3_qknorm_rope_4heads", default=False)
_H3_4HEADS_ENV = "SGLANG_ENABLE_DIFFUSION_MINIMAX_H3_DIT_QKNORM_ROPE_4HEADS"


def around_can_use_fused_inplace_qknorm_rope(
    original: Callable[..., bool], *args: Any, **kwargs: Any
) -> bool:
    if is_hcu() and not _H3_FUSED_PROBE.get():
        return False
    return original(*args, **kwargs)


def around_minimax_h3_attention_init(original, attention, *args, **kwargs):
    if not is_hcu():
        return original(attention, *args, **kwargs)

    arch = get_gcn_arch(torch.cuda.current_device()).partition(":")[0]
    allow_fusion = arch == "gfx936"
    token = _H3_FUSED_PROBE.set(allow_fusion)
    try:
        result = original(attention, *args, **kwargs)
    finally:
        _H3_FUSED_PROBE.reset(token)
    if not allow_fusion:
        attention._use_fused_qknorm_rope = False
    return result


def around_minimax_h3_attend(original, attention, *args, **kwargs):
    from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var

    if (
        not attention._use_fused_qknorm_rope
        or not get_bool_env_var(_H3_4HEADS_ENV, "false")
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return original(attention, *args, **kwargs)

    token = _H3_4HEADS_ACTIVE.set(True)
    try:
        return original(attention, *args, **kwargs)
    finally:
        _H3_4HEADS_ACTIVE.reset(token)


def around_fused_inplace_qknorm_rope(original, *args, **kwargs):
    if _H3_4HEADS_ACTIVE.get():
        from hcu_sglang.multimodal_gen.minimax_h3_dit_qknorm_rope import (
            try_fused_inplace_qknorm_rope_4heads,
        )

        if try_fused_inplace_qknorm_rope_4heads(*args, **kwargs):
            return None
    return original(*args, **kwargs)


def register_hooks(registry=None) -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if registry is not None:
        HookRegistry = registry

    HookRegistry.register(
        "sglang.kernels.ops.diffusion.rope.qknorm_rope_jit."
        "can_use_fused_inplace_qknorm_rope",
        around_can_use_fused_inplace_qknorm_rope,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
        "MiniMaxH3Attention.__init__",
        around_minimax_h3_attention_init,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
        "MiniMaxH3Attention._attend",
        around_minimax_h3_attend,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.diffusion.rope.qknorm_rope_jit."
        "fused_inplace_qknorm_rope",
        around_fused_inplace_qknorm_rope,
        HookType.AROUND,
    )
