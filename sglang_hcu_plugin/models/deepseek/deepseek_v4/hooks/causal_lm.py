"""DeepseekV4ForCausalLM method hooks."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_v4 import methods
from sglang_hcu_plugin.models.deepseek.hook_utils import (
    register_method_replaces_from_module,
)

_BASE = "sglang.srt.models.deepseek_v4"


def register_hooks() -> int:
    return register_method_replaces_from_module(
        _BASE,
        "DeepseekV4ForCausalLM",
        ("__init__", "forward", "load_weights", "post_load_weights"),
        methods.causal_lm,
    )
