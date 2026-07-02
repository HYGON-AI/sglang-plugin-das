"""DCU DeepSeek V2 plugin package."""

from __future__ import annotations

import logging

from . import hooks as v2_hooks
from . import module_funcs
from sglang_dcu_plugin.models.deepseek.hook_utils import (
    ensure_post_apply_module_patch,
    register_function_replaces,
)

logger = logging.getLogger(__name__)

_BASE_MODULE = "sglang.srt.models.deepseek_v2"


def register_hooks() -> None:
    """Register method-level REPLACE hooks for DCU DeepSeek V2 fork delta."""
    n = 0
    n += register_function_replaces(
        _BASE_MODULE,
        ("ds_bmm_wrapper", "add_forward_absorb_core_attention_backend"),
        module_funcs,
    )
    n += v2_hooks.register_hooks()

    ensure_post_apply_module_patch()

    logger.info(
        "[sglang_dcu_plugin] Registered %d precise DeepSeek V2 hooks",
        n,
    )
