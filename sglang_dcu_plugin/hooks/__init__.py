"""Central hook registration."""

import logging

logger = logging.getLogger(__name__)


def register_all_hooks() -> None:
    from sglang_dcu_plugin.hooks.layers import (
        attention,
        dp_attention,
        moe,
        norm_activation,
        quantization,
    )
    from sglang_dcu_plugin.hooks.models import deepseek, qwen, utils as model_utils

    attention.register()
    moe.register()
    quantization.register()
    norm_activation.register()
    dp_attention.register()
    deepseek.register()
    qwen.register()
    model_utils.register()
    logger.info("[sglang_dcu_plugin] hooks: layers + models registered")
