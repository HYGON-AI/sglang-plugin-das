"""DeepseekV2WeightLoaderMixin hooks."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_common import methods
from sglang_dcu_plugin.models.deepseek.hook_utils import (
    register_method_replaces_from_module,
)

_BASE = "sglang.srt.models.deepseek_common.deepseek_weight_loader"
_SRC = methods.weight_loader

_METHODS = (
    "do_load_weights",
    "post_load_weights",
)


def register_hooks() -> int:
    return register_method_replaces_from_module(
        _BASE,
        "DeepseekV2WeightLoaderMixin",
        _METHODS,
        _SRC,
    )
