"""SGLang DCU general plugin — model & layer adaptations only."""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point for sglang.srt.plugins (called by load_plugins)."""
    from sglang_dcu_plugin.hooks import register_all_hooks

    register_all_hooks()
    logger.info("[sglang_dcu_plugin] All hooks registered")
