from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import textwrap
import unittest

from hcu_sglang.mem_cache import cpu_cache

ROOT = Path(__file__).resolve().parents[1]


class CpuCacheSourcesTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "tests/contracts/cpu_cache_sources.json").read_text())
        self.source = (ROOT / "hcu_sglang/mem_cache/cpu_cache_methods.py").read_text()
        self.tree = ast.parse(self.source)

    def test_extracted_definitions_retain_source_bytes(self):
        self.assertEqual(len(self.manifest["definitions"]), 28)
        for row in self.manifest["definitions"]:
            with self.subTest(symbol=row["owned_symbol"]):
                cls, name = row["owned_symbol"].split(".")
                owner = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
                node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                body = textwrap.dedent("".join(self.source.splitlines(True)[start - 1:node.end_lineno]))
                self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), row["source_definition_sha256"])
                self.assertFalse(any(isinstance(n, ast.Name) and n.id == "super" for n in ast.walk(node)))

    def test_all_changed_methods_are_registered_or_installed_helpers(self):
        tree = ast.parse((ROOT / "hcu_sglang/mem_cache/cpu_cache.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_GROUPS" for t in n.targets))
        groups = ast.literal_eval(node.value)
        expected = {(r["donor"], n) for r in self.manifest["groups"] for n in r["methods"] + r["helpers"]}
        actual = {(donor, n) for _, _, donor, methods, helpers in groups for n in methods + helpers}
        self.assertEqual(actual, expected)
        declared = json.loads((ROOT / "tests/contracts/hcu_source_alignment_targets.json").read_text())["expected_hook_targets"]
        self.assertIn("sglang.srt.mem_cache.unified_cache.storage_attachment.StorageAttachment.__init__", declared)
        for module, cls, donor, methods, helpers in groups:
            parent = module + ("." + cls if cls else "")
            for name in methods:
                self.assertIn(parent + "." + name, declared)
            if helpers and cls and "__init__" not in methods:
                self.assertIn(parent + ".__init__", declared)
            if helpers and not cls:
                self.assertIn(module, cpu_cache._MODULE_HELPERS)
                donor_name, names = cpu_cache._MODULE_HELPERS[module]
                self.assertEqual(donor_name, donor)
                self.assertEqual(set(names), set(helpers))

    def test_reachable_from_existing_allocation_entry(self):
        tree = ast.parse((ROOT / "hcu_sglang/mem_cache/allocation.py").read_text())
        register = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register_hooks")
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "cpu_cache" and n.func.attr == "register_hooks" for n in ast.walk(register)))
        compile(self.source, "cpu_cache_methods.py", "exec")


if __name__ == "__main__":
    unittest.main()
