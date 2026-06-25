"""FlashAttention hooks for DCU fused RMSNorm + RoPE KV-store paths."""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_FORWARD_EXTEND_TARGET = (
    "sglang.srt.layers.attention.flashattention_backend."
    "FlashAttentionBackend.forward_extend"
)
_FORWARD_DECODE_TARGET = (
    "sglang.srt.layers.attention.flashattention_backend."
    "FlashAttentionBackend.forward_decode"
)


def register() -> None:
    hooks = (
        (_FORWARD_EXTEND_TARGET, around_forward_extend),
        (_FORWARD_DECODE_TARGET, around_forward_decode),
    )
    for target, hook in hooks:
        logger.info("[sglang_dcu_plugin] register AROUND hook: %s", target)
        HookRegistry.register(target, hook, HookType.AROUND)


def _should_skip_kv_store(args, kwargs) -> bool:
    if not get_bool_env_var("SGLANG_USE_FUSED_RMS_ROTARY"):
        return False

    # Method args:
    #   self, q, k, v, layer, forward_batch, save_kv_cache=True,
    #   q_rope=None, k_rope=None, sinks=None
    k = args[2] if len(args) > 2 else kwargs.get("k")
    if k is None:
        return False

    k_rope = args[8] if len(args) > 8 else kwargs.get("k_rope")
    return k_rope is None


def _call_without_kv_store(original_fn, *args, **kwargs):
    if not _should_skip_kv_store(args, kwargs):
        return original_fn(*args, **kwargs)

    if len(args) > 6:
        args = list(args)
        args[6] = False
        return original_fn(*args, **kwargs)

    kwargs = dict(kwargs)
    kwargs["save_kv_cache"] = False
    return original_fn(*args, **kwargs)


def around_forward_extend(original_fn, *args, **kwargs):
    return _call_without_kv_store(original_fn, *args, **kwargs)


def around_forward_decode(original_fn, *args, **kwargs):
    return _call_without_kv_store(original_fn, *args, **kwargs)
