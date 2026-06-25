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
    from sglang_dcu_plugin.models.qwen import qwen3_5

    flashattention_backend.register()
    gdn_backend.register()
    linear_attention.register()
    layernorm_gated.register()
    topk.register()
    qwen3_5.register()
    logger.info("[sglang_dcu_plugin] optimization hooks registered")
