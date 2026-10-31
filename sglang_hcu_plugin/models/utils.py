"""HCU model utility adaptations."""

from __future__ import annotations

import logging

from sglang_hcu_plugin.models.deepseek.hook_utils import register_function_replaces

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from sglang_hcu_plugin.models.model_utils import methods

    n = register_function_replaces(
        "sglang.srt.models.utils",
        ("enable_fused_set_kv_buffer",),
        methods.utils,
    )
    logger.info("[sglang_hcu_plugin] Registered %d models.utils hooks", n)
