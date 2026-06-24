"""DCU DeepSeek model adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_dcu_plugin.models.deepseek import (
        deepseek_v2,
        deepseek_v4,
        weight_loader,
    )

    deepseek_v2.register_hooks()
    deepseek_v4.register_hooks()
    weight_loader.register_hooks()
    logger.debug("[sglang_dcu_plugin] deepseek hooks registered")
