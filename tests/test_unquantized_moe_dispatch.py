# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TARGET = "sglang.srt.layers.quantization.unquant.UnquantizedFusedMoEMethod.forward_cuda"


def register_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and node.args
        ):
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant):
            out.append(str(value.value))
        else:
            text = ast.unparse(value)
            out.append(text[2:-1] if text.startswith(("f'", 'f"')) else text)
    return out


class UnquantizedMoeDispatchTests(unittest.TestCase):
    def test_forward_cuda_registered_only_by_dispatcher(self):
        hits = []
        for name in ("unquantized_aiter", "unquantized_marlin", "unquantized_moe_dispatch"):
            path = ROOT / f"hcu_sglang/layers/quantization/{name}.py"
            if TARGET in register_targets(path):
                hits.append(name)
        self.assertEqual(hits, ["unquantized_moe_dispatch"])

    def test_marlin_arm_wins_when_enabled(self):
        from hcu_sglang.layers.quantization import (
            unquantized_aiter,
            unquantized_marlin,
            unquantized_moe_dispatch,
        )

        original = object()
        with (
            patch.object(unquantized_marlin, "_enabled", return_value=True),
            patch.object(unquantized_marlin, "around_forward_cuda") as marlin_arm,
            patch.object(unquantized_aiter, "around_forward_cuda") as aiter_arm,
        ):
            unquantized_moe_dispatch.around_forward_cuda(
                original, "method", "layer", "dispatch"
            )
        marlin_arm.assert_called_once_with(original, "method", "layer", "dispatch")
        aiter_arm.assert_not_called()

    def test_aiter_arm_runs_when_marlin_disabled(self):
        from hcu_sglang.layers.quantization import (
            unquantized_aiter,
            unquantized_marlin,
            unquantized_moe_dispatch,
        )

        original = object()
        with (
            patch.object(unquantized_marlin, "_enabled", return_value=False),
            patch.object(unquantized_marlin, "around_forward_cuda") as marlin_arm,
            patch.object(unquantized_aiter, "around_forward_cuda") as aiter_arm,
        ):
            unquantized_moe_dispatch.around_forward_cuda(
                original, "method", "layer", "dispatch"
            )
        aiter_arm.assert_called_once_with(original, "method", "layer", "dispatch")
        marlin_arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
