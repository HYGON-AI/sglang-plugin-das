"""HCU attention adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.layers.attention import (
        hcu_mla_backend,
        current_dsa,
        flashattention_patch,
        nsa_patch,
    )

    flashattention_patch.register_hooks()
    nsa_patch.register_hooks()
    hcu_mla_backend.register_hooks()
    current_dsa.register_hooks()
    logger.debug("[hcu_sglang] attention hooks registered")
