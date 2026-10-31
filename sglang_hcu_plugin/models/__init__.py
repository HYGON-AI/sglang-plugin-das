"""HCU model adaptations and hook registration."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_hcu_plugin.models import deepseek, qwen, utils

    deepseek.register_hooks()
    qwen.register_hooks()
    utils.register_hooks()
    logger.info("[sglang_hcu_plugin] model hooks registered")
