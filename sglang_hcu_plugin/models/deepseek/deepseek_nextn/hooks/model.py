"""DeepseekModelNextN hooks."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_nextn import methods
from sglang_hcu_plugin.models.deepseek.hook_utils import (
    register_method_replaces_from_module,
)

_BASE = "sglang.srt.models.deepseek_nextn"
_SRC = methods.model

_METHODS = (
    "__init__",
    "forward",
)


def register_hooks() -> int:
    return register_method_replaces_from_module(
        _BASE,
        "DeepseekModelNextN",
        _METHODS,
        _SRC,
    )
