# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_closure_builder", ROOT / "tools/build_runtime_closure.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
GATE_SPEC = importlib.util.spec_from_file_location("runtime_closure_gate", ROOT / "ci/l0_gate.py")
GATE = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(GATE)


def triage_row(path="python/sglang/a.py", **updates):
    row = {key: "" for key in BUILDER.TRIAGE_FIELDS}
    row.update(path=path, group="kv-cache", current_status="pending", relation="changed",
               category="missing_or_partial_implementation", reason="review pending", verification_lane="CPU-first")
    row.update(updates)
    return row


def implemented(**updates):
    row = {"status": "implemented", "rationale": "reviewed, but GPU remains unverified",
           "implementation": ["hcu_sglang/a.py"], "tests": ["tests/a.py"]}
    row.update(updates)
    return row


class RuntimeClosureTests(unittest.TestCase):
    def test_current_decision_overrides_classification_without_promoting_validation(self):
        source = triage_row(current_status="in_progress")
        row = BUILDER.build_closure([source], {source["path"]: implemented()})[0]
        self.assertEqual(row["execution_status"], "implemented")
        self.assertEqual(row["classification_status_snapshot"], "in_progress")
        self.assertEqual(json.loads(row["implementation"]), ["hcu_sglang/a.py"])
        self.assertEqual(json.loads(row["evidence"]), [])

    def test_missing_decision_is_explicit_and_never_defaults_to_pending(self):
        rows = BUILDER.build_closure([triage_row(current_status="implemented")], {})
        self.assertEqual(rows[0]["execution_status"], "missing_decision")
        self.assertEqual(BUILDER.missing_decision_records(rows), ["python/sglang/a.py"])

    def test_denominator_is_the_union_of_triage_and_decisions(self):
        decisions = {"outside.py": implemented()}
        first = BUILDER.build_closure([triage_row()], decisions)
        decisions["outside.py"]["validation_scope"] = "new evidence boundary"
        second = BUILDER.build_closure([triage_row()], decisions)
        self.assertEqual([row["path"] for row in first], ["outside.py", "python/sglang/a.py"])
        self.assertTrue(all(row["decisions_sha256"] == first[0]["decisions_sha256"] for row in first))
        self.assertNotEqual(first[0]["decisions_sha256"], second[0]["decisions_sha256"])
        outside = next(row for row in first if row["path"] == "outside.py")
        self.assertEqual(outside["group"], "decisions_only")

    def test_render_is_stable_under_input_order_and_round_trips_csv_fields(self):
        rows = [triage_row("b.py"), triage_row("a.py", reason='中文, "quoted"\nsecond line')]
        left = BUILDER.render_closure(BUILDER.build_closure(rows, {"b.py": implemented()}))
        right = BUILDER.render_closure(BUILDER.build_closure(list(reversed(rows)), {"b.py": implemented()}))
        self.assertEqual(left, right)
        restored = list(csv.DictReader(io.StringIO(left)))
        self.assertEqual(restored[0]["classification_reason"], rows[1]["reason"])

    def test_invalid_or_duplicate_rows_and_unsubstantiated_statuses_are_rejected(self):
        invalid = [([triage_row(), triage_row()], {}), ([{"path": "a.py"}], {}),
                   ([], {}), ([triage_row(category="unknown")], {}),
                   ([triage_row(category="not_applicable", reason="")], {}),
                   ([triage_row()], {"python/sglang/a.py": implemented(status="verified")}),
                   ([triage_row()], {"python/sglang/a.py": implemented(rationale="")}),
                   ([triage_row()], {"python/sglang/a.py": implemented(implementation=[])}),
                   ([triage_row()], {"python/sglang/a.py": implemented(status="validated")})]
        for rows, decisions in invalid:
            with self.subTest(rows=rows, decisions=decisions), self.assertRaises(ValueError):
                BUILDER.build_closure(rows, decisions)

    def test_exclusion_reason_and_validation_evidence_remain_explicit(self):
        source = triage_row(category="not_applicable", reason="not reachable on HCU")
        decisions = {source["path"]: {"status": "separate_component", "rationale": "not reachable on HCU"}}
        row = BUILDER.build_closure([source], decisions)[0]
        self.assertEqual(row["execution_status"], "separate_component")
        decisions[source["path"]] = implemented(status="validated", evidence=["artifacts/evidence.json"])
        row = BUILDER.build_closure([source], decisions)[0]
        self.assertEqual(row["execution_status"], "validated")
        self.assertEqual(json.loads(row["evidence"]), ["artifacts/evidence.json"])

    def test_cli_and_l0_reject_changed_decisions_without_rebuild_and_never_write_on_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plans").mkdir()
            (root / "tools").mkdir()
            shutil.copyfile(ROOT / "tools/build_runtime_closure.py", root / "tools/build_runtime_closure.py")
            triage, decisions, output = root / BUILDER.TRIAGE, root / BUILDER.DECISIONS, root / BUILDER.OUTPUT
            with triage.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=BUILDER.TRIAGE_FIELDS)
                writer.writeheader()
                writer.writerow(triage_row())
            decisions.write_text("{}")
            args = ["--triage", str(triage), "--decisions", str(decisions), "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(BUILDER.main(args + ["--check"]), 1)
                self.assertFalse(output.exists())
                self.assertEqual(BUILDER.main(args), 1)
                self.assertTrue(any("no decision record" in error for error in GATE.runtime_closure_errors(root)))
                previous = output.read_bytes()
                decisions.write_text(json.dumps({"python/sglang/a.py": implemented()}))
                self.assertEqual(BUILDER.main(args + ["--check"]), 1)
                self.assertEqual(output.read_bytes(), previous)
                self.assertTrue(any("runtime closure is stale" in error for error in GATE.runtime_closure_errors(root)))
                self.assertEqual(BUILDER.main(args), 0)
                self.assertEqual(GATE.runtime_closure_errors(root), [])
                self.assertEqual(BUILDER.main(args + ["--output", str(decisions)]), 1)
                self.assertEqual(json.loads(decisions.read_text())["python/sglang/a.py"]["status"], "implemented")

    def test_repository_output_contains_every_triage_file_once_and_is_current(self):
        with (ROOT / BUILDER.TRIAGE).open(newline="") as stream:
            triage = list(csv.DictReader(stream))
        decisions = json.loads((ROOT / BUILDER.DECISIONS).read_text())
        expected = BUILDER.render_closure(BUILDER.build_closure(triage, decisions)).encode()
        self.assertEqual((ROOT / BUILDER.OUTPUT).read_bytes(), expected)
        self.assertEqual(len(triage), 1464)
        self.assertEqual(len({row["path"] for row in triage}), len(triage))
        rows = list(csv.DictReader(io.StringIO(expected.decode())))
        self.assertEqual(len(rows), len({row["path"] for row in triage} | set(decisions)))


if __name__ == "__main__":
    unittest.main()
