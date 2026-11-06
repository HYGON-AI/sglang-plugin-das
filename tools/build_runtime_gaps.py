# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TRIAGE = Path("plans/hcu-runtime-triage-20261105.csv")
OWNED = Path("tests/contracts/runtime_owned_sources.json")
HOOKS = Path("artifacts/kv-cpu-20261105/hook-targets.json")
NON_COVERAGE_RELATIONS = {"changed", "fork_only"}


def module_of(path):
    if not path.startswith("python/"):
        return None
    stem = path[len("python/"):]
    if not stem.endswith(".py"):
        return None
    stem = stem[:-3]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def owned_sources(payload):
    sources = set()
    for entry in payload.get("files", []):
        sources.add(entry["source"])
    for group in ("extracted_files", "adapter_files"):
        for entry in payload.get(group, []):
            for source in entry.get("sources", []):
                sources.add(source["source"])
    return sources


def build(triage_rows, owned, hook_payload, groups=None):
    targets = hook_payload.get("targets", {})
    unresolved = hook_payload.get("unresolved_detail", [])
    if hook_payload.get("unresolved", 0) != len(unresolved):
        raise ValueError(
            "unresolved count %d does not match recorded detail %d"
            % (hook_payload.get("unresolved", 0), len(unresolved))
        )
    buckets = collections.defaultdict(list)
    undetermined = []
    for row in triage_rows:
        if row["relation"] not in NON_COVERAGE_RELATIONS:
            continue
        if groups is not None and row["group"] not in groups:
            continue
        path = row["path"]
        module = module_of(path)
        if module is None:
            undetermined.append(
                {
                    "path": path,
                    "reason": "path_outside_python_tree",
                    "detail": "无法从路径推出模块名，不能判定是否有 owned 覆盖或注册目标",
                }
            )
            continue
        covering = sorted({owner for target, owner in targets.items() if target == module or target.startswith(module + ".")})
        record = {
            "path": path,
            "group": row["group"],
            "relation": row["relation"],
            "category": row["category"],
            "changed_definitions": row["changed_definitions"],
            "uncovered_definitions": row["uncovered_definitions"],
        }
        if path in owned:
            buckets["covered_owned_file"].append(dict(record, owner_scope="runtime_owned_sources"))
        elif covering:
            buckets["covered_hook_target"].append(dict(record, hook_owners=covering[:5]))
        else:
            buckets["gap_no_owned_no_target"].append(record)
    return buckets, undetermined


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classify every changed/fork_only triage file into owned coverage, hook-target coverage, real gap, or undetermined."
    )
    parser.add_argument("--triage", type=Path, default=TRIAGE)
    parser.add_argument("--owned", type=Path, default=OWNED)
    parser.add_argument("--hooks", type=Path, default=HOOKS)
    parser.add_argument("--groups", default="", help="comma separated triage group filter")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)

    triage_rows = list(csv.DictReader((ROOT / args.triage).open()))
    owned = owned_sources(json.loads((ROOT / args.owned).read_text()))
    hook_payload = json.loads((ROOT / args.hooks).read_text())
    groups = {item for item in args.groups.split(",") if item} or None
    buckets, undetermined = build(triage_rows, owned, hook_payload, groups)

    summary = {
        "schema_version": 1,
        "semantics": (
            "changed/fork_only 文件按覆盖来源分桶：owned 全量副本或抽取来源、公开 Hook 目标、"
            "两者皆无（真缺口）、无法判定（不并入任何既有桶）。"
        ),
        "triage_rows": len(triage_rows),
        "counts": {name: len(items) for name, items in sorted(buckets.items())},
        "undetermined": len(undetermined),
        "unresolved_hook_targets": hook_payload.get("unresolved", 0),
        "buckets": dict(buckets),
        "undetermined_detail": undetermined,
    }
    if args.json:
        (ROOT / args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if args.csv:
        fields = ("path", "group", "relation", "category", "changed_definitions", "uncovered_definitions")
        buffer = []
        buffer.append(",".join(fields))
        for record in buckets["gap_no_owned_no_target"]:
            buffer.append(",".join(str(record[field]) for field in fields))
        (ROOT / args.csv).write_text("\n".join(buffer) + "\n")
    print(json.dumps({"counts": summary["counts"], "undetermined": summary["undetermined"]}, ensure_ascii=False))
    if undetermined:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
