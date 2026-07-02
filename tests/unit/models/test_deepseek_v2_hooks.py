"""Unit tests for precise DeepSeek V2 plugin hooks."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

METHODS_DIR = (
    Path(__file__).resolve().parents[2]
    / "sglang_dcu_plugin/models/deepseek/deepseek_v2/methods"
)


@pytest.fixture(autouse=True)
def reset_hook_registry():
    HookRegistry.reset()
    yield
    HookRegistry.reset()


def test_methods_package_has_expected_functions():
    expected = {
        "mlp.py": {"forward"},
        "moe.py": {
            "__init__",
            "forward",
            "forward_normal",
            "forward_deepep",
            "forward_normal_dual_stream",
            "_forward_shared_experts",
        },
        "decoder.py": {"__init__", "forward"},
        "attention.py": {
            "__init__",
            "prepare_qkv_latent",
            "forward_absorb_prepare",
            "forward_absorb_core",
            "_set_mla_kv_buffer",
            "_get_mla_kv_buffer",
            "_get_mla_kv_buffer_from_fp8",
            "_concat_and_cast_mha_k",
            "_chunked_prefix_attn_mha",
            "forward_normal_chunked_kv_prepare",
            "forward_normal_chunked_kv_core",
            "forward_normal_one_shot_prepare",
            "forward_normal_one_shot_core",
            "forward_absorb_fused_mla_rope_prepare",
            "forward_absorb_fused_mla_rope_core",
            "forward_absorb_fused_mla_rope_cpu_prepare",
            "forward_absorb_fused_mla_rope_cpu_core",
        },
    }
    for filename, names in expected.items():
        tree = ast.parse((METHODS_DIR / filename).read_text(encoding="utf-8"))
        found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert names <= found, f"{filename} missing {names - found}"


def test_precise_hooks_not_whole_file_replace():
    from sglang_dcu_plugin.models.deepseek import deepseek_v2

    deepseek_v2.register_hooks()

    targets = set(HookRegistry._hooks)
    assert "sglang.srt.models.deepseek_v2.DeepseekV2MLP.forward" in targets
    assert "sglang.srt.models.deepseek_v2.DeepseekV2MoE.forward" in targets
    assert "sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA" not in targets
    assert (
        "sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA.forward_prepare"
        not in targets
    )
    assert (
        "sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA.prepare_qkv_latent"
        in targets
    )
    assert (
        "sglang.srt.models.deepseek_common.attention_forward_methods"
        ".forward_mla.DeepseekMLAForwardMixin.forward_absorb_prepare"
        in targets
    )
    assert "sglang.srt.models.deepseek_v2.MoEGate" not in targets


def test_hook_count():
    from sglang_dcu_plugin.models.deepseek import deepseek_v2

    deepseek_v2.register_hooks()
    assert len(HookRegistry._hooks) == 28


def test_no_deepseek_v2_impl_module():
    spec = importlib.util.find_spec("sglang_dcu_plugin.models.deepseek.deepseek_v2_impl")
    assert spec is None
