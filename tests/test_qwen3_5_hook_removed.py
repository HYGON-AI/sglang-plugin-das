# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention"


def _constant_str(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Best-effort evaluation of a string expression, following module aliases."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_str(node.left, aliases)
        right = _constant_str(node.right, aliases)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                resolved = _constant_str(value.value, aliases)
                if resolved is None:
                    return None
                parts.append(resolved)
            else:
                return None
        return "".join(parts)
    return None


def _module_string_aliases(tree: ast.Module) -> dict[str, str]:
    """Collect module-level string assignments such as ``_TARGET = "..."``."""
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value_node = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value_node = node.value
            targets = [node.target]
        else:
            continue
        value = _constant_str(value_node, aliases)
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                aliases[target.id] = value
    return aliases


def _registered_targets(source: str) -> list[str]:
    """Resolve the first argument of every ``register(...)`` call in ``source``.

    Following module-level aliases is what catches the old passthrough
    registration, which passed ``_TARGET`` instead of a literal string.
    """
    tree = ast.parse(source)
    aliases = _module_string_aliases(tree)
    targets = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(
            function, "id", None
        )
        if name != "register":
            continue
        value = _constant_str(node.args[0], aliases)
        if value is not None:
            targets.append(value)
    return targets


class Qwen35HookRemovedTests(unittest.TestCase):
    def test_plugin_entry_does_not_register_the_passthrough_hook(self):
        text = (ROOT / "hcu_sglang/__init__.py").read_text()
        # The MTP quant head shares the qwen3_5 module prefix; the passthrough
        # hook is identified by its target, not by the prefix.
        self.assertNotIn(TARGET, text)

    def test_only_the_fused_rms_rope_module_registers_the_target(self):
        # The passthrough registration stayed retired, but the target itself is
        # hooked again: #94/#98/#104/#106 route the fused RMSNorm + RoPE
        # KV-store wrappers through it from models/qwen/qwen3_5.py (restored
        # together with the contract entry in b8d8db0). No other module may
        # register it, or two AROUND wrappers stack on the same method.
        registrants = [
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "hcu_sglang").rglob("*.py"))
            if TARGET in _registered_targets(path.read_text())
        ]
        self.assertEqual(registrants, [str(Path("hcu_sglang/models/qwen/qwen3_5.py"))])

    def test_detector_flags_module_level_alias_registration(self):
        # Regression for the old qwen3_5 module: the target lived in a
        # module-level alias, so a literal-only scan missed it.
        old_source = '''
_TARGET = "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention"
_MODULE = "sglang.srt.models.qwen3_5"
_GDN_INIT_TARGET = _MODULE + ".Qwen3_5GatedDeltaNet.__init__"


def register() -> None:
    HookRegistry.register(_TARGET, replacement_self_attention, HookType.AROUND)
    HookRegistry.register(_GDN_INIT_TARGET, around_gdn_init, HookType.AROUND)
'''
        registered = _registered_targets(old_source)
        self.assertIn(TARGET, registered)
        self.assertIn(
            "sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet.__init__", registered
        )

    def test_declared_target_contract_lists_the_registered_target(self):
        import json

        declared = json.loads((ROOT / "tests/contracts/hcu_source_alignment_targets.json").read_text())[
            "expected_hook_targets"
        ]
        self.assertIn(TARGET, declared)


if __name__ == "__main__":
    unittest.main()
