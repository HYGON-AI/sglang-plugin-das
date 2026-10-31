"""HCU attention adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_hcu_plugin.layers.attention import (
        hcu_mla_backend,
        flashattention_patch,
        nsa_patch,
    )

    flashattention_patch.register_hooks()
    nsa_patch.register_hooks()
    hcu_mla_backend.register_hooks()
    logger.debug("[sglang_hcu_plugin] attention hooks registered")
