# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
TARGET = "sglang.srt.arg_groups.parallel_hook.handle_eplb_and_dispatch"


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


class EplbHookDispatchTests(unittest.TestCase):
    def test_handle_target_registered_only_by_dispatcher(self):
        hits = []
        for name in ("integration", "placement", "hook_dispatch"):
            path = ROOT / f"hcu_sglang/eplb/{name}.py"
            if TARGET in register_targets(path):
                hits.append(name)
        self.assertEqual(hits, ["hook_dispatch"])

    def test_chain_runs_placement_outer_and_integration_inner(self):
        from hcu_sglang.eplb import hook_dispatch

        calls = []
        original = Mock(side_effect=lambda args: calls.append("original"))
        integration_arm = Mock(
            side_effect=lambda orig, args: (calls.append("integration"), orig(args))[1]
        )
        placement_arm = Mock(
            side_effect=lambda orig, args: (calls.append("placement"), orig(args))[1]
        )
        fake_integration = types.ModuleType("hcu_sglang.eplb.integration")
        fake_integration.around_dispatch_validation = integration_arm
        fake_placement = types.ModuleType("hcu_sglang.eplb.placement")
        fake_placement.around_validation = placement_arm
        with patch.dict(
            sys.modules,
            {
                "hcu_sglang.eplb.integration": fake_integration,
                "hcu_sglang.eplb.placement": fake_placement,
            },
        ):
            hook_dispatch.around_handle_eplb_and_dispatch(original, "args")
        # Registration order used to make placement the outer hook.
        self.assertEqual(calls, ["placement", "integration", "original"])
        self.assertEqual(placement_arm.call_args.args[0].__name__, "chained_original")


if __name__ == "__main__":
    unittest.main()
