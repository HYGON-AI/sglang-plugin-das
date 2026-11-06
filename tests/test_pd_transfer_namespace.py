# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PDTransferNamespaceTests(unittest.TestCase):
    def test_prefill_and_decode_use_owned_transfer_configuration(self):
        for name in ("prefill", "decode"):
            with self.subTest(name=name):
                tree = ast.parse((ROOT / "hcu_sglang/disagg" / (name + ".py")).read_text())
                imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and any(alias.name == "configure_v4_transfer" for alias in node.names)]
                self.assertEqual(len(imports), 1)
                self.assertEqual(imports[0].module, "hcu_sglang.mem_cache.cp_cache_layer_split.transfer")
                pool_imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and any(alias.name == "DeepSeekV4TokenToKVPool" for alias in node.names)]
                self.assertEqual([node.module for node in pool_imports], ["hcu_sglang.mem_cache.hcu_v4_pool"])


if __name__ == "__main__":
    unittest.main()
