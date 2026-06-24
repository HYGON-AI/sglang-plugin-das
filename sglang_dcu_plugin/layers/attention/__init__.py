"""DCU attention adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_dcu_plugin.layers.attention import (
        dcu_mla_backend,
        flashattention_patch,
        nsa_patch,
    )

    flashattention_patch.register_hooks()
    nsa_patch.register_hooks()
    dcu_mla_backend.register_hooks()
    logger.debug("[sglang_dcu_plugin] attention hooks registered")
