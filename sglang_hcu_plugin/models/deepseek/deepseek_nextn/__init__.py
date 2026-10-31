"""HCU deepseek_nextn plugin package."""

from __future__ import annotations

import logging

from . import hooks as nextn_hooks

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    """Register hooks for deepseek_nextn HCU fork delta."""
    n = nextn_hooks.register_hooks()

    logger.info(
        "[sglang_hcu_plugin] Registered %d deepseek_nextn hooks",
        n,
    )
