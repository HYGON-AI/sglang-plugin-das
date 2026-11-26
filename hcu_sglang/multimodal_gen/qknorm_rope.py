"""HCU compatibility for the diffusion QK-Norm + RoPE JIT kernel."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import get_gcn_arch, is_hcu


_H3_FUSED_PROBE = ContextVar("hcu_minimax_h3_fused_qknorm_rope_probe", default=False)


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
