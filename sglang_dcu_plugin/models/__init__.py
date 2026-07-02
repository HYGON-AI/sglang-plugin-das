"""DCU model adaptations and hook registration."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_dcu_plugin.models import deepseek, qwen, utils

    deepseek.register_hooks()
    qwen.register_hooks()
    utils.register_hooks()
    logger.info("[sglang_dcu_plugin] model hooks registered")
