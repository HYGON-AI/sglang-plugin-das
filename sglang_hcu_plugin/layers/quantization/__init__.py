"""HCU quantization adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_hcu_plugin.layers.quantization import fp8, w8a8_fp8

    fp8.register_hooks()
    w8a8_fp8.register_hooks()
    logger.debug("[sglang_hcu_plugin] quantization hooks registered")
