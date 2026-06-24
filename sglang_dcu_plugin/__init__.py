"""SGLang DCU general plugin — model & layer adaptations only."""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point for sglang.srt.plugins (called by load_plugins)."""
    from sglang_dcu_plugin.layers import register_hooks as register_layer_hooks
    from sglang_dcu_plugin.models import register_hooks as register_model_hooks

    register_layer_hooks()
    register_model_hooks()
    logger.info("[sglang_dcu_plugin] All hooks registered")
