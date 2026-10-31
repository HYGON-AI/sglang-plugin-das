"""deepseek_nextn hook registration."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_nextn.hooks import model


def register_hooks() -> int:
    return model.register_hooks()
