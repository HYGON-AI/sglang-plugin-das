# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DAS = Path("../input-code/sglang-das/python")
OFFICIAL = Path("../input-code/sglang/python")
RUNTIME = Path("artifacts/kv-cpu-20261105/runtime-targets.json")
STATIC = Path("artifacts/kv-cpu-20261105/hook-targets.json")


def module_of(path):
    stem = path[len("python/"):-3]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def top_definitions(tree):
    found = collections.OrderedDict()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = {"kind": "function", "text": ast.unparse(node)}
        elif isinstance(node, ast.ClassDef):
            methods = collections.OrderedDict()
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[sub.name] = ast.unparse(sub)
            found[node.name] = {"kind": "class", "text": ast.unparse(node), "methods": methods}
    return found


def changed_definitions(das_tree, official_tree):
    mine = top_definitions(das_tree)
    theirs = top_definitions(official_tree)
    changed = []
    for name, node in mine.items():
        other = theirs.get(name)
        if other is None:
            changed.append({"name": name, "kind": node["kind"], "change": "added"})
            continue
        if other["kind"] != node["kind"]:
            changed.append({"name": name, "kind": node["kind"], "change": "kind_changed"})
            continue
        if node["kind"] == "function":
            if node["text"] != other["text"]:
                changed.append({"name": name, "kind": "function", "change": "modified"})
            continue
        if node["text"] == other["text"]:
            continue
        added = [m for m in node["methods"] if m not in other["methods"]]
        removed = [m for m in other["methods"] if m not in node["methods"]]
        modified = [m for m in node["methods"] if m in other["methods"] and node["methods"][m] != other["methods"][m]]
        if not added and not removed and not modified:
            changed.append({"name": name, "kind": "class", "change": "modified", "methods": ["<class body outside methods>"]})
            continue
        for method in added:
            changed.append({"name": name, "kind": "method", "change": "added", "method": method})
        for method in modified:
            changed.append({"name": name, "kind": "method", "change": "modified", "method": method})
        for method in removed:
            changed.append({"name": name, "kind": "method_removed", "change": "removed_in_das", "method": method})
    return changed


def candidates(defn, module):
    name = defn["name"]
    if defn["kind"] == "method":
        base = "%s.%s.%s" % (module, name, defn["method"])
    else:
        base = "%s.%s" % (module, name)
    return [base]


def covered(base, targets):
    hits = []
    for target in targets:
        if target == base or target.startswith(base + "."):
            hits.append(target)
    return hits


def importers_of(module, index):
    return sorted(name for name in index.get(module, []))


def build_index(das_root):
    index = collections.defaultdict(set)
    files = sorted(das_root.rglob("*.py"))
    if len(files) < 1000:
        raise SystemExit("DAS tree looks empty (%d files under %s)" % (len(files), das_root))
    for path in files:
        rel = path.relative_to(das_root).as_posix()
        rel_module = rel[:-3].replace("/", ".")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="ignore"))
        except SyntaxError as exc:
            raise SystemExit("unparsable DAS file %s: %s" % (rel, exc))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                index[name].add(rel_module)
    return index, len(files)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rejudge gap files against the runtime hook target set with definition-level matching."
    )
    parser.add_argument("--paths", type=Path, required=True, help="newline separated python/... source paths")
    parser.add_argument("--das", type=Path, default=DAS)
    parser.add_argument("--official", type=Path, default=OFFICIAL)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--static", type=Path, default=STATIC)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    das_root = (ROOT / args.das).resolve()
    official_root = (ROOT / args.official).resolve()
    runtime = set(json.loads((ROOT / args.runtime).read_text())["targets"])
    static = set(json.loads((ROOT / args.static).read_text())["targets"])
    if not runtime:
        raise SystemExit("runtime target set is empty; refusing to report coverage")
    index, scanned = build_index(das_root)

    paths = [line.strip() for line in (ROOT / args.paths).read_text().splitlines() if line.strip()]
    if not paths:
        raise SystemExit("gap path list is empty")
    report = {"schema_version": 1, "scanned_das_files": scanned, "runtime_targets": len(runtime), "files": []}
    totals = collections.Counter()
    for rel_path in paths:
        das_file = das_root / rel_path[len("python/"):]
        official_file = official_root / rel_path[len("python/"):]
        if not das_file.exists():
            raise SystemExit("missing DAS file: %s" % das_file)
        if official_file.exists():
            official_tree = ast.parse(official_file.read_text(encoding="utf-8-sig"))
        else:
            official_tree = ast.Module(body=[], type_ignores=[])
        module = module_of(rel_path)
        das_tree = ast.parse(das_file.read_text(encoding="utf-8-sig"))
        defs = changed_definitions(das_tree, official_tree)
        entries = []
        for defn in defs:
            bases = candidates(defn, module)
            hits = []
            for base in bases:
                hits.extend(covered(base, runtime))
            entries.append(
                dict(
                    defn,
                    runtime_targets=sorted(set(hits)),
                    static_targets=sorted({t for base in bases for t in covered(base, static)}),
                )
            )
        uncovered = [e for e in entries if not e["runtime_targets"]]
        importers = importers_of(module, index)
        if uncovered and importers:
            reach = "imported_but_uncovered"
        elif uncovered and not importers:
            reach = "no_static_importer"
        elif entries:
            reach = "runtime_covered"
        else:
            reach = "no_definition_delta"
        totals[reach] += 1
        totals["definitions"] += len(entries)
        totals["definitions_uncovered"] += len(uncovered)
        report["files"].append(
            {
                "path": rel_path,
                "official_present": official_file.exists(),
                "module": module,
                "reachability": reach,
                "importers": importers,
                "changed_definitions": len(entries),
                "uncovered_definitions": len(uncovered),
                "definitions": entries,
            }
        )
    report["counts"] = dict(totals)
    (ROOT / args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
