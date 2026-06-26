"""SGLang DCU general plugin - model and layer adaptations."""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point for sglang.srt.plugins (called by load_plugins)."""
    from sglang_dcu_plugin.layers.attention import flashattention_backend
    from sglang_dcu_plugin.layers.attention.fla import (
        layernorm_gated,
        linear_attention,
    )
    from sglang_dcu_plugin.layers.attention.linear import gdn_backend
    from sglang_dcu_plugin.layers.moe import topk
    from sglang_dcu_plugin.models import bailing_moe
    from sglang_dcu_plugin.models.qwen import qwen3, qwen3_5, qwen3_moe

    flashattention_backend.register()
    gdn_backend.register()
    linear_attention.register()
    layernorm_gated.register()
    topk.register()
    bailing_moe.register()
    qwen3.register()
    qwen3_moe.register()
    qwen3_5.register()
    logger.info("[sglang_dcu_plugin] optimization hooks registered")
