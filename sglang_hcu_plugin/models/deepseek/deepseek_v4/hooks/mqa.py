"""MQALayer method hooks."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_v4 import methods
from sglang_hcu_plugin.models.deepseek.hook_utils import (
    register_method_replaces_from_module,
)

_BASE = "sglang.srt.models.deepseek_v4"


def register_hooks() -> int:
    return register_method_replaces_from_module(
        _BASE,
        "MQALayer",
        ("__init__", "_compute_q_b", "_forward_prepare", "forward"),
        methods.mqa,
    )
