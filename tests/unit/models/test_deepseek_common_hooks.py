"""Unit tests for deepseek_common / deepseek_nextn / models.utils plugin hooks."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sglang.srt.plugins.hook_registry import HookRegistry

PLUGIN = Path(__file__).resolve().parents[2] / "sglang_hcu_plugin/models"


@pytest.fixture(autouse=True)
def reset_hook_registry():
    HookRegistry.reset()
    yield
    HookRegistry.reset()


def test_deepseek_common_methods_extracted():
    methods_dir = PLUGIN / "deepseek/deepseek_common/methods"
    expected = {
        "weight_loader.py": {"do_load_weights", "post_load_weights"},
        "attention_backend_handler.py": {"handle_attention_hcu_mla"},
    }
    for filename, names in expected.items():
        tree = ast.parse((methods_dir / filename).read_text(encoding="utf-8"))
        found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert names <= found, f"{filename} missing {names - found}"


def test_deepseek_nextn_methods_extracted():
    tree = ast.parse(
        (PLUGIN / "deepseek/deepseek_nextn/methods/model.py").read_text(
            encoding="utf-8"
        )
    )
    found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"__init__", "forward"} <= found


def test_models_utils_method_extracted():
    tree = ast.parse(
        (PLUGIN / "model_utils/methods/utils.py").read_text(encoding="utf-8")
    )
    found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "enable_fused_set_kv_buffer" in found


def test_deepseek_common_hook_registration():
    from sglang_hcu_plugin.models.deepseek import deepseek_common, deepseek_nextn

    deepseek_common.register_hooks()
    deepseek_nextn.register_hooks()
    targets = set(HookRegistry._hooks)
    assert (
        "sglang.srt.models.deepseek_common.deepseek_weight_loader"
        ".DeepseekV2WeightLoaderMixin.do_load_weights"
    ) in targets
    assert (
        "sglang.srt.models.deepseek_common.deepseek_weight_loader"
        ".DeepseekV2WeightLoaderMixin.post_load_weights"
    ) in targets
    assert "sglang.srt.models.deepseek_nextn.DeepseekModelNextN.__init__" in targets
    assert "sglang.srt.models.deepseek_nextn.DeepseekModelNextN.forward" in targets


def test_models_utils_hook_registration():
    from sglang_hcu_plugin.models import utils as model_utils

    model_utils.register_hooks()
    assert "sglang.srt.models.utils.enable_fused_set_kv_buffer" in HookRegistry._hooks
