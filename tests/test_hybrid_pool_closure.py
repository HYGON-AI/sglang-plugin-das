# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hybrid_pool_closure", ROOT / "tools/check_hybrid_pool_closure.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
INPUT = ROOT.parent / "input-code"


class HybridPoolClosureTests(unittest.TestCase):
    def sources(self):
        official = (INPUT / "sglang" / CHECK.REL).read_text()
        das = (INPUT / "sglang-das" / CHECK.REL).read_text()
        return official, das

    def test_every_helper_of_the_extracted_definitions_is_ast_equal(self):
        report = CHECK.closure_report(*self.sources())
        self.assertEqual(report["extracted"], list(CHECK.EXTRACTED))
        self.assertTrue(report["all_equal"])
        self.assertGreaterEqual(report["helper_count"], 15)
        names = {row["name"] for row in report["helpers"]}
        for required in ("build_pool_entry", "_split_hicache_size", "_get_allocator_type",
                         "build_kv_only_stack", "build_host_pool_group",
                         "build_minimax_sparse_hicache_stack", "StackBuildResult"):
            self.assertIn(required, names)

    def test_persisted_evidence_matches_a_fresh_run(self):
        path = ROOT / "artifacts/kv-cpu-20261105/hybrid-pool-closure.json"
        if not path.is_file():
            self.skipTest("closure evidence has not been generated")
        stored = json.loads(path.read_text())
        fresh = CHECK.closure_report(*self.sources())
        self.assertEqual(stored["helper_count"], fresh["helper_count"])
        self.assertEqual(stored["helpers"], fresh["helpers"])

    def test_negative_case_detects_a_mutated_helper(self):
        official, das = self.sources()
        mutated = official.replace(
            "def _get_allocator_type() -> str:",
            "def _get_allocator_type() -> str:\n    return 'mutated'",
            1,
        )
        self.assertNotEqual(mutated, official)
        report = CHECK.closure_report(mutated, das)
        self.assertFalse(report["all_equal"])
        failing = {row["name"] for row in report["helpers"] if not row["equal"]}
        self.assertIn("_get_allocator_type", failing)

    def test_negative_case_detects_a_missing_helper(self):
        official, das = self.sources()
        tree = ast.parse(official)
        tree.body = [node for node in tree.body
                     if not (isinstance(node, ast.FunctionDef) and node.name == "build_pool_entry")]
        report = CHECK.closure_report(ast.unparse(tree), das)
        self.assertFalse(report["all_equal"])
        failing = {row["name"] for row in report["helpers"] if not row["equal"]}
        self.assertIn("build_pool_entry", failing)


if __name__ == "__main__":
    unittest.main()
