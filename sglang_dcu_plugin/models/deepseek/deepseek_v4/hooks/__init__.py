"""DeepSeek V4 method-level plugin hooks."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v4.hooks import (
    causal_lm,
    decoder,
    model,
    mqa,
)


def register_hooks() -> int:
    n = 0
    n += mqa.register_hooks()
    n += decoder.register_hooks()
    n += model.register_hooks()
    n += causal_lm.register_hooks()
    return n
