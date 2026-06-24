"""Hooks for layers/moe."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sglang_dcu_plugin.layers.moe.topk import replacement_fused_topk


logger = logging.getLogger(__name__)


def register() -> None:
    target = "sglang.srt.layers.moe.topk.fused_topk"
    logger.info("[sglang_dcu_plugin] register REPLACE hook: %s", target)
    HookRegistry.register(target, replacement_fused_topk, HookType.REPLACE)
