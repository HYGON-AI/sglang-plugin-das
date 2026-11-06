from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("l0_gate_headers", ROOT / "ci/l0_gate.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class MigrationHeaderTests(unittest.TestCase):
    def check(self, text, *, section="files", source_lines=(), omit=(), status=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contracts = root / "tests/contracts"
            contracts.mkdir(parents=True)
            (root / "migrated.py").write_text(text)
            declared = re.search(r"SPDX-|copyright|licensed under|Apache License", "\n".join(source_lines), re.I)
            source = {
                "source": "python/sglang/source.py",
                "source_sha256": "a" * 64,
                "source_header_lines": list(source_lines),
                "source_declaration": status or ("保留来源声明" if declared else "来源无声明"),
            }
            for key in omit:
                del source[key]
            row = {"target": "migrated.py"}
            if section == "files":
                row.update(source)
            else:
                row["sources"] = [source]
            data = {section: [row]}
            (contracts / "runtime_owned_sources.json").write_text(json.dumps(data))
            (contracts / "owned_model_sources.json").write_text(json.dumps({"files": []}))
            return GATE.migration_header_errors(root)

    def test_preserved_spdx_and_attribution_pass(self):
        header = "# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: 2026 SGLang contributors\n"
        self.assertEqual(self.check(header + "value = 1\n", section="extracted_files", source_lines=header.splitlines()), [])

    def test_source_without_declarations_does_not_require_fabricated_header(self):
        self.assertEqual(self.check("value = 1\n"), [])

    def test_source_with_only_license_or_copyright_passes_unchanged(self):
        for line in ("# SPDX-License-Identifier: Apache-2.0", "# Copyright original author"):
            with self.subTest(line=line):
                self.assertEqual(self.check(line + "\nvalue = 1\n", source_lines=[line]), [])

    def test_deleted_source_header_metadata_is_rejected(self):
        errors = self.check("value = 1\n", omit=["source_header_lines"])
        self.assertEqual(len(errors), 1)
        self.assertIn("missing original source header metadata", errors[0])

    def test_missing_source_path_or_sha_is_rejected(self):
        for key, message in (("source", "source path"), ("source_sha256", "source SHA256")):
            with self.subTest(key=key):
                errors = self.check("value = 1\n", omit=[key])
                self.assertEqual(len(errors), 1)
                self.assertIn(message, errors[0])

    def test_unregistered_declaration_status_is_rejected(self):
        errors = self.check("value = 1\n", omit=["source_declaration"])
        self.assertEqual(len(errors), 1)
        self.assertIn("source declaration status", errors[0])

    def test_source_license_cannot_be_labeled_absent(self):
        line = "# SPDX-License-Identifier: Apache-2.0"
        errors = self.check(line + "\nvalue = 1\n", source_lines=[line], status="来源无声明")
        self.assertEqual(len(errors), 1)
        self.assertIn("source declaration status", errors[0])

    def test_missing_original_license_is_rejected(self):
        errors = self.check("# Copyright original author\nvalue = 1\n", source_lines=["# SPDX-License-Identifier: Apache-2.0"])
        self.assertEqual(len(errors), 1)
        self.assertIn("missing original source header line", errors[0])

    def test_body_text_cannot_spoof_header(self):
        lines = ["# SPDX-License-Identifier: Apache-2.0", "# Copyright SGLang Team"]
        errors = self.check("value = 1\n" + "\n".join(lines), source_lines=lines)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("missing original source header line" in e for e in errors))

    def test_added_generic_header_cannot_replace_source_attribution(self):
        errors = self.check("# SPDX-License-Identifier: Apache-2.0\n# Copyright Hygon\nvalue = 1\n", source_lines=["# Copyright original author"])
        self.assertEqual(len(errors), 1)
        self.assertIn("missing original source header line", errors[0])

    def test_empty_target_does_not_bypass_original_declaration(self):
        self.assertTrue(self.check("", source_lines=["# Copyright original author"]))

    def test_empty_package_marker_has_no_migrated_code(self):
        self.assertEqual(self.check(""), [])

    def test_original_adapter_tracks_source_without_copying_its_header(self):
        self.assertEqual(self.check("value = 1\n", section="adapter_files", source_lines=["# Copyright original author"]), [])

    def test_missing_provenance_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(len(GATE.migration_header_errors(Path(directory))), 2)


if __name__ == "__main__":
    unittest.main()
