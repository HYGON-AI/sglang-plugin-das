"""Hooks for layers/norm_activation."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from sglang_dcu_plugin.layers.attention.fla.layernorm_gated import (
    replacement_layer_norm_fwd,
)


logger = logging.getLogger(__name__)


def register() -> None:
    target = "sglang.srt.layers.attention.fla.layernorm_gated._layer_norm_fwd"
    logger.info("[sglang_dcu_plugin] register REPLACE hook: %s", target)
    HookRegistry.register(target, replacement_layer_norm_fwd, HookType.REPLACE)
