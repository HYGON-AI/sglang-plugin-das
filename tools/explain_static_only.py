# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_hook_targets as scan

ROOT = Path(__file__).resolve().parents[1]


def registration_sites(tree, path):
    module_assignments = scan._scope_assignments(
        [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef, ast.ClassDef, ast.If, ast.Try))]
    )
    aliases = scan._hook_registry_aliases(tree)
    sites = []

    def record(node, local, guards):
        receiver = scan._expr_text(node.func.value)
        if not scan._is_hook_registry(receiver, aliases):
            return
        value = scan._eval(node.args[0], local, module_assignments)
        if value is None:
            return
        sites.append(
            {
                "target": value,
                "line": node.lineno,
                "guards": [text for kind, text in guards],
            }
        )

    def walk(body, local, guards):
        current = dict(local)
        for node in body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                current[node.targets[0].id] = node.value
            elif isinstance(node, ast.For):
                values = scan._literal_strings(node.iter)
                if values is not None and isinstance(node.target, ast.Name):
                    for value in values:
                        inner = dict(current)
                        inner[node.target.id] = ast.Constant(value=value)
                        walk(node.body, inner, guards + [("for", ast.unparse(node.iter))])
                else:
                    walk(node.body, current, guards + [("for", ast.unparse(node.iter))])
            elif isinstance(node, ast.If):
                walk(node.body, current, guards + [("if", ast.unparse(node.test))])
                walk(node.orelse, current, guards + [("else", ast.unparse(node.test))])
            elif isinstance(node, ast.Try):
                walk(node.body, current, guards + [("try", "try")])
                for handler in node.handlers:
                    walk(handler.body, current, guards + [("except", ast.unparse(handler.type) if handler.type else "bare")])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(node.body, current, guards + [("function", node.name)])
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "register"
                and node.value.args
            ):
                record(node.value, current, guards)

    walk(tree.body, {}, [])
    return sites


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explain why each static_only hook target was not registered at runtime by listing the guards around its registration site.")
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--owned", type=Path, default=Path("artifacts/kv-cpu-20261105/hook-targets.json"))
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    diff = json.loads((ROOT / args.diff).read_text())
    static_only = set(diff["static_only"])
    owners = json.loads((ROOT / args.owned).read_text())["targets"]
    by_owner = {}
    for target in static_only:
        by_owner.setdefault(owners.get(target, ""), []).append(target)

    report = {"schema_version": 1, "static_only": len(static_only), "owners": {}}
    unresolved_sites = []
    for owner, targets in sorted(by_owner.items()):
        if not owner:
            unresolved_sites.extend({"target": t, "owner": "", "reason": "owner_unknown"} for t in targets)
            continue
        path = ROOT / owner
        if not path.is_file():
            unresolved_sites.extend({"target": t, "owner": owner, "reason": "owner_file_missing"} for t in targets)
            continue
        sites = {site["target"]: site for site in registration_sites(ast.parse(path.read_text()), path)}
        rows = []
        for target in sorted(targets):
            site = sites.get(target)
            if site is None:
                rows.append({"target": target, "explained": False, "reason": "no_registration_site_found"})
                continue
            rows.append(
                {
                    "target": target,
                    "line": site["line"],
                    "guards": list(site["guards"]),
                    "explained": bool(site["guards"]),
                }
            )
        report["owners"][owner] = rows

    explained = sum(1 for rows in report["owners"].values() for row in rows if row["explained"])
    missing = sum(1 for rows in report["owners"].values() for row in rows if not row["explained"])
    report["explained"] = explained
    report["unexplained"] = missing
    report["owner_unknown"] = len(unresolved_sites)
    (ROOT / args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"explained": explained, "unexplained": missing, "owner_unknown": len(unresolved_sites)}, ensure_ascii=False))
    return 0 if missing == 0 and not unresolved_sites else 1


if __name__ == "__main__":
    sys.exit(main())
