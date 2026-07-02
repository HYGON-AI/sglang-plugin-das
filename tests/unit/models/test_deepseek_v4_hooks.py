"""Unit tests for DeepSeek V4 plugin hooks."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sglang.srt.plugins.hook_registry import HookRegistry

METHODS_DIR = (
    Path(__file__).resolve().parents[2]
    / "sglang_dcu_plugin/models/deepseek/deepseek_v4/methods"
)


@pytest.fixture(autouse=True)
def reset_hook_registry():
    HookRegistry.reset()
    yield
    HookRegistry.reset()


def test_deepseek_v4_methods_extracted():
    expected = {
        "mqa.py": {"__init__", "_compute_q_b", "_forward_prepare", "forward"},
        "decoder.py": {"hc_pre", "hc_post", "forward"},
        "model.py": {"__init__", "forward"},
        "causal_lm.py": {"__init__", "forward", "load_weights", "post_load_weights"},
    }
    for filename, names in expected.items():
        tree = ast.parse((METHODS_DIR / filename).read_text(encoding="utf-8"))
        found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert names <= found, f"{filename} missing {names - found}"


def test_deepseek_v4_hook_registration():
    from sglang_dcu_plugin.models.deepseek import deepseek_v4

    deepseek_v4.register_hooks()

    targets = set(HookRegistry._hooks)
    assert "sglang.srt.models.deepseek_v4.MQALayer.forward" in targets
    assert "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_pre" in targets
    assert "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights" in targets
    assert "sglang.srt.models.deepseek_v4.MQALayer" not in targets
    assert len(HookRegistry._hooks) == 13
