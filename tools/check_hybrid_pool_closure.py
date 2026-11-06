# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

REL = "python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py"
EXTRACTED = (
    "_MambaStrategy.build",
    "_MambaSwaStrategy.build",
    "_MambaSwaStrategy.build_direct_linker_pool_group",
    "_build_mha_mla_host_pool",
    "build_full_draft_pools",
    "build_hybrid_mamba_stack",
    "build_hybrid_mamba_swa_stack",
    "build_hybrid_swa_group",
    "build_kv_host_pool",
    "build_mamba_host_pool",
    "_mha_host_inputs",
    "attach_hybrid_dsa_pool_to_hiradix_cache",
    "attach_hybrid_minimax_sparse_pool_to_hiradix_cache",
)


def _module_nodes(tree):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _resolve(tree, dotted):
    scope = tree.body
    node = None
    for part in dotted.split("."):
        for candidate in scope:
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and candidate.name == part:
                node = candidate
                break
        else:
            return None
        scope = getattr(node, "body", [])
    return node


def _referenced_module_names(node, module_names):
    used = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in module_names:
            used.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id in module_names:
            used.add(child.value.id)
    return used


def closure_report(official_src, das_src):
    official_tree = ast.parse(official_src)
    das_tree = ast.parse(das_src)
    official_module = _module_nodes(official_tree)
    das_module = _module_nodes(das_tree)
    extracted = {}
    helpers = set()
    for name in EXTRACTED:
        node = _resolve(das_tree, name)
        if node is None:
            raise ValueError(f"extracted definition missing in DAS: {name}")
        extracted[name] = node
        helpers |= _referenced_module_names(node, set(official_module) | set(das_module))
    helpers -= {name.split(".")[-1] for name in EXTRACTED}
    rows = []
    for helper in sorted(helpers):
        das_node = das_module.get(helper)
        official_node = official_module.get(helper)
        if das_node is None or official_node is None:
            rows.append(dict(name=helper, das_present=das_node is not None, official_present=official_node is not None, equal=False, reason="missing on one side"))
            continue
        equal = ast.dump(das_node, include_attributes=False) == ast.dump(official_node, include_attributes=False)
        rows.append(dict(name=helper, das_present=True, official_present=True, equal=equal, kind=type(das_node).__name__))
    return dict(extracted=list(EXTRACTED), helper_count=len(rows), helpers=rows, all_equal=all(row["equal"] for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prove that every helper referenced by the extracted hybrid pool assembler definitions is AST-equal between the fixed DAS and official trees.")
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--das", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = closure_report((args.official / REL).read_text(), (args.das / REL).read_text())
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"helper_count": report["helper_count"], "all_equal": report["all_equal"]}, ensure_ascii=False))
    return 0 if report["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
