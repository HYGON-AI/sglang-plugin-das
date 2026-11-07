# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def register_target_reprs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "HookRegistry"
            and node.args
        ):
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant):
            out.append(str(value.value))
        else:
            text = ast.unparse(value)
            if text.startswith(("f'", 'f"')):
                text = text[2:-1]
            out.append(text)
    return out


class W8A8RunnerDispatchTests(unittest.TestCase):
    def test_w8a8_arms_no_longer_register_w8a8_create_moe_runner(self):
        for name in ("w8a8_aiter", "w8a8_lightop"):
            path = ROOT / f"hcu_sglang/layers/quantization/{name}.py"
            targets = register_target_reprs(path)
            hits = [
                t
                for t in targets
                if t.endswith(".create_moe_runner")
                and ("{module}" in t or "{target}" in t)
            ]
            self.assertEqual(
                hits,
                [],
                f"{name} must not register w8a8 create_moe_runner (got {hits})",
            )

    def test_compressed_create_moe_runner_stays_with_lightop(self):
        path = ROOT / "hcu_sglang/layers/quantization/w8a8_lightop.py"
        targets = register_target_reprs(path)
        hits = [t for t in targets if t.endswith(".create_moe_runner")]
        self.assertEqual(hits, ["{compressed_target}.create_moe_runner"])
        self.assertIn("CompressedTensorsW8A8Fp8MoE", path.read_text())

    def test_dispatcher_registers_both_w8a8_targets_once(self):
        source = (
            ROOT / "hcu_sglang/layers/quantization/w8a8_runner_dispatch.py"
        ).read_text()
        self.assertIn('"W8A8Int8MoEMethod"', source)
        self.assertIn('"W8A8FP8MoEMethod"', source)
        targets = [
            t
            for t in register_target_reprs(
                ROOT / "hcu_sglang/layers/quantization/w8a8_runner_dispatch.py"
            )
            if t.endswith(".create_moe_runner")
        ]
        self.assertEqual(len(targets), 1)
        self.assertIn("{module}.{class_name}", targets[0])

    def test_lightop_arm_wins_when_requested(self):
        from hcu_sglang.layers.quantization import (
            w8a8_aiter,
            w8a8_lightop,
            w8a8_runner_dispatch,
        )

        original = object()
        with (
            patch.object(w8a8_lightop, "_requested_lightop", return_value=True),
            patch.object(w8a8_lightop, "around_create_moe_runner") as lightop_arm,
            patch.object(w8a8_aiter, "around_create_moe_runner") as aiter_arm,
        ):
            w8a8_runner_dispatch.around_create_moe_runner(
                original, "quant", "layer", "config"
            )
        lightop_arm.assert_called_once_with(original, "quant", "layer", "config")
        aiter_arm.assert_not_called()

    def test_aiter_arm_runs_when_lightop_not_requested(self):
        from hcu_sglang.layers.quantization import (
            w8a8_aiter,
            w8a8_lightop,
            w8a8_runner_dispatch,
        )

        original = object()
        with (
            patch.object(w8a8_lightop, "_requested_lightop", return_value=False),
            patch.object(w8a8_lightop, "around_create_moe_runner") as lightop_arm,
            patch.object(w8a8_aiter, "around_create_moe_runner") as aiter_arm,
        ):
            w8a8_runner_dispatch.around_create_moe_runner(
                original, "quant", "layer", "config"
            )
        aiter_arm.assert_called_once_with(original, "quant", "layer", "config")
        lightop_arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
