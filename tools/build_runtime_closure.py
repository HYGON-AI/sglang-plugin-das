# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TRIAGE = Path("plans/hcu-runtime-triage-20261105.csv")
DECISIONS = Path("plans/sglang-das-decisions.json")
OUTPUT = Path("plans/hcu-runtime-target-closure-20261105.csv")
TRIAGE_FIELDS = (
    "path", "group", "current_status", "relation", "category", "reason",
    "verification_lane", "changed_definitions", "uncovered_definitions",
    "owned_ast_equal_files",
)
FIELDS = (
    "path", "group", "relation", "category", "execution_status",
    "classification_status_snapshot", "classification_reason", "verification_lane",
    "changed_definitions", "uncovered_definitions", "owned_ast_equal_files",
    "rationale", "implementation", "validation_scope", "tests", "evidence",
    "decision_json", "decisions_sha256", "triage_sha256",
)
STATUSES = {"pending", "in_progress", "implemented", "validated", "upstream", "separate_component"}
CATEGORIES = {"existing_needs_validation", "missing_or_partial_implementation", "not_applicable"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_closure(triage, decisions):
    if not isinstance(decisions, dict):
        raise ValueError("decisions must be a path-to-decision object")
    indexed = {}
    for source in triage:
        if not all(isinstance(source.get(key), str) for key in TRIAGE_FIELDS):
            raise ValueError("triage row has missing or invalid fields")
        path = source["path"]
        if not path or path in indexed:
            raise ValueError(f"empty or duplicate triage path: {path}")
        if not source["group"] or not source["relation"] or source["category"] not in CATEGORIES:
            raise ValueError(f"{path}: invalid triage classification")
        if source["category"] == "not_applicable" and not source["reason"].strip():
            raise ValueError(f"{path}: exclusion needs a reason")
        indexed[path] = source
    if not indexed:
        raise ValueError("triage must contain at least one file")
    decisions_sha = digest(decisions)
    triage_sha = digest([indexed[path] for path in sorted(indexed)])
    unknown = {}
    for path in decisions:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("decisions has an empty or non-string path")
        if path not in indexed:
            unknown[path] = {key: "" for key in TRIAGE_FIELDS}
            unknown[path].update(
                path=path,
                group="decisions_only",
                current_status="unknown",
                relation="unknown",
                category="unclassified",
                reason="decision recorded without a triage row",
            )
    indexed.update(unknown)
    rows = []
    for path, source in sorted(indexed.items()):
        decision = decisions.get(path, {})
        if not isinstance(decision, dict):
            raise ValueError(f"{path}: decision must be an object")
        status = decision.get("status", "missing_decision")
        if status not in STATUSES and status != "missing_decision":
            raise ValueError(f"{path}: invalid status {status}")
        if status == "missing_decision" and decision:
            raise ValueError(f"{path}: missing_decision row carries a decision")
        if status != "missing_decision":
            if status != "pending" and not decision.get("rationale"):
                raise ValueError(f"{path}: disposition needs a rationale")
            if status in {"implemented", "validated"} and not decision.get("implementation"):
                raise ValueError(f"{path}: implementation paths are required")
            if status == "validated" and not decision.get("evidence"):
                raise ValueError(f"{path}: validation evidence is required")
        rows.append({
            "path": path, "group": source["group"], "relation": source["relation"],
            "category": source["category"], "execution_status": status,
            "classification_status_snapshot": source["current_status"],
            "classification_reason": source["reason"],
            **{key: source[key] for key in (
                "verification_lane", "changed_definitions", "uncovered_definitions", "owned_ast_equal_files"
            )},
            "rationale": decision.get("rationale", ""),
            "implementation": canonical_json(decision.get("implementation", [])),
            "validation_scope": decision.get("validation_scope", ""),
            "tests": canonical_json(decision.get("tests", [])),
            "evidence": canonical_json(decision.get("evidence", [])),
            "decision_json": canonical_json(decision),
            "decisions_sha256": decisions_sha, "triage_sha256": triage_sha,
        })
    return rows


def missing_decision_records(rows):
    return sorted(row["path"] for row in rows if row["execution_status"] == "missing_decision")


def render_closure(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build one runtime closure row per triage file; implementation is not runtime validation.")
    parser.add_argument("--triage", type=Path, default=ROOT / TRIAGE)
    parser.add_argument("--decisions", type=Path, default=ROOT / DECISIONS)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {args.triage.resolve(), args.decisions.resolve()}:
            raise ValueError("output must not overwrite an input")
        with args.triage.open(encoding="utf-8", newline="") as stream:
            triage = list(csv.DictReader(stream))
        rows = build_closure(triage, json.loads(args.decisions.read_text(encoding="utf-8")))
        content = render_closure(rows).encode("utf-8")
        if args.check:
            if not args.output.is_file() or args.output.read_bytes() != content:
                print(f"runtime closure is stale or missing: {args.output}; run tools/build_runtime_closure.py", file=sys.stderr)
                return 1
        else:
            args.output.write_bytes(content)
        summary = {
            "files": len(rows),
            "execution_status": dict(sorted(Counter(row["execution_status"] for row in rows).items())),
            "missing_decision": len(missing_decision_records(rows)),
        }
        print(json.dumps(summary, ensure_ascii=False))
        if summary["missing_decision"]:
            print(
                f"runtime closure: {summary['missing_decision']} triage files have no decision record; "
                "add a disposition instead of relying on a default status",
                file=sys.stderr,
            )
            return 1
        return 0
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        print(f"runtime closure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
