"""HCU MoE adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_hcu_plugin.layers.moe import lightop_runner

    lightop_runner.register_hooks()
    logger.debug("[sglang_hcu_plugin] moe hooks registered")
