from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeOwnedSourceTests(unittest.TestCase):
    def test_only_declared_deltas_change_fixed_source_bytes(self):
        manifest = json.loads(
            (ROOT / "tests/contracts/runtime_owned_sources.json").read_text()
        )
        self.assertTrue(manifest["files"])
        self.assertEqual(len(manifest["files"]), len({r["target"] for r in manifest["files"]}))
        for row in manifest["files"]:
            with self.subTest(target=row["target"]):
                path = ROOT / row["target"]
                data = path.read_bytes()
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), row["target_sha256"]
                )
                text = data.decode()
                suffix = row.get("adapter_suffix", "")
                if suffix:
                    self.assertTrue(text.endswith(suffix))
                    text = text[: -len(suffix)]
                for replacement in row["reverse_replacements"]:
                    owned, source = replacement[:2]
                    count = 1 if len(replacement) > 2 and replacement[2] == "once" else -1
                    self.assertIn(owned, text)
                    text = text.replace(owned, source, count)
                self.assertEqual(
                    hashlib.sha256(text.encode()).hexdigest(),
                    row["source_sha256"],
                )

    def test_extracted_and_adapter_provenance_is_pinned(self):
        manifest = json.loads((ROOT / "tests/contracts/runtime_owned_sources.json").read_text())
        targets = set()
        for section in ("files", "extracted_files", "adapter_files"):
            for row in manifest.get(section, []):
                self.assertNotIn(row["target"], targets)
                targets.add(row["target"])
                self.assertEqual(hashlib.sha256((ROOT / row["target"]).read_bytes()).hexdigest(), row["target_sha256"])
                if section == "files":
                    continue
                self.assertTrue(row["sources"])
                for source in row["sources"]:
                    self.assertRegex(source["source_sha256"], r"^[0-9a-f]{64}$")
                if section == "adapter_files":
                    self.assertIsInstance(row["provenance"], str)
                    self.assertTrue(row["provenance"].strip())
                else:
                    contract = json.loads((ROOT / row["definition_contract"]).read_text())
                    if "definitions" in contract:
                        declared = {(r["source"], r["source_sha256"]) for r in contract["definitions"]}
                        self.assertEqual(declared, {(r["source"], r["source_sha256"]) for r in row["sources"]})
                    else:
                        selected = [r for r in contract["files"] if r["owned"] == row["target"]]
                        self.assertTrue(selected)
                        self.assertTrue(all(r["definitions"] for r in selected))
                        self.assertEqual({r["source"] for r in selected}, {r["source"] for r in row["sources"]})
        expected = {"hcu_sglang/mem_cache/cpu_cache.py", "hcu_sglang/mem_cache/cpu_cache_methods.py", "hcu_sglang/models/registry.py"}
        expected.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "hcu_sglang/models").glob("*/entries/*.py"))
        self.assertTrue(expected.issubset(targets), expected - targets)

    def test_owned_modules_do_not_retain_family_source_imports(self):
        forbidden = (
            "sglang.kernels.ops.muse_spark",
            "sglang.kernels.spec",
            "sglang.srt.layers.muse_spark",
            "sglang.srt.configs.muse_spark",
            "sglang.srt.models.muse_spark_omni",
            "sglang.srt.layers.layer_boundary.hcu_legacy",
            "sglang.srt.mem_cache.layer_split",
            "sglang.srt.layers.utils.glm5_next_cp",
            "sglang.srt.layers.communicator_glm5_next_cp",
            "sglang.srt.layers.communicator_glm5_next_mhc",
        )
        manifest = json.loads(
            (ROOT / "tests/contracts/runtime_owned_sources.json").read_text()
        )
        for row in manifest["files"]:
            with self.subTest(target=row["target"]):
                tree = ast.parse((ROOT / row["target"]).read_text())
                imports = [
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                ] + [
                    alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names
                ]
                self.assertFalse(
                    [
                        name
                        for name in imports
                        if any(
                            name == prefix or name.startswith(prefix + ".")
                            for prefix in forbidden
                        )
                    ], imports
                )

    def test_new_modules_compile_without_changing_existing_config_module(self):
        manifest = json.loads(
            (ROOT / "tests/contracts/runtime_owned_sources.json").read_text()
        )
        for row in manifest["files"]:
            source = (ROOT / row["target"]).read_text()
            compile(source, row["target"], "exec")
        self.assertTrue((ROOT / "hcu_sglang/configs.py").is_file())
        self.assertTrue((ROOT / "hcu_sglang/configs_muse_spark.py").is_file())


if __name__ == "__main__":
    unittest.main()
