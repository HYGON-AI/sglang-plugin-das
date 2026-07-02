"""DeepseekV2DecoderLayer method hooks."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v2 import methods
from sglang_dcu_plugin.models.deepseek.hook_utils import (
    register_method_replaces_from_module,
)

_BASE = "sglang.srt.models.deepseek_v2"


def register_hooks() -> int:
    return register_method_replaces_from_module(
        _BASE,
        "DeepseekV2DecoderLayer",
        ("__init__", "forward"),
        methods.decoder,
    )
