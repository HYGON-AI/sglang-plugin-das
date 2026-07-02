"""DCU layer adaptations and hook registration."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_dcu_plugin.layers import (
        attention,
        dp_attention,
        moe,
        norm_activation,
        quantization,
    )

    attention.register_hooks()
    moe.register_hooks()
    quantization.register_hooks()
    norm_activation.register_hooks()
    dp_attention.register_hooks()
    logger.info("[sglang_dcu_plugin] layer hooks registered")
