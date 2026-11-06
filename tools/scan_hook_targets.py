# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys


def _scope_assignments(body):
    found = {}
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = node.value
    return found


def _eval(node, assignments, module_assignments, depth=0):
    if depth > 8:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        for table in (assignments, module_assignments):
            if node.id in table:
                return _eval(table[node.id], assignments, module_assignments, depth + 1)
        return None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                resolved = _eval(value.value, assignments, module_assignments, depth + 1)
                if resolved is None:
                    return None
                parts.append(resolved)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval(node.left, assignments, module_assignments, depth + 1)
        right = _eval(node.right, assignments, module_assignments, depth + 1)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        base = _eval(node.func.value, assignments, module_assignments, depth + 1)
        if base is None:
            return None
        try:
            args = [_eval(a, assignments, module_assignments, depth + 1) for a in node.args]
            return base.format(*args)
        except Exception:
            return None
    return None


def _literal_strings(node):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            return [e.value for e in node.elts]
    return None


def _hook_registry_aliases(tree):
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("hook_registry"):
            for alias in node.names:
                if alias.name == "HookRegistry":
                    aliases.add(alias.asname or alias.name)
    return aliases


def _is_hook_registry(receiver, aliases):
    return receiver == "HookRegistry" or receiver.endswith(".HookRegistry") or receiver in aliases


def _expr_text(node):
    try:
        return ast.unparse(node)[:300]
    except Exception as exc:
        return "<unparse_failed:%s>" % type(exc).__name__


def _walk_block(body, assignments, module_assignments, path, targets, counter):
    local = dict(assignments)
    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            local[node.targets[0].id] = node.value
        elif isinstance(node, ast.For):
            values = _literal_strings(node.iter)
            if values is not None and isinstance(node.target, ast.Name):
                for value in values:
                    inner = dict(local)
                    inner[node.target.id] = ast.Constant(value=value)
                    _walk_block(node.body, inner, module_assignments, path, targets, counter)
            else:
                _walk_block(node.body, local, module_assignments, path, targets, counter)
        elif isinstance(node, ast.If):
            _walk_block(node.body, local, module_assignments, path, targets, counter)
            _walk_block(node.orelse, local, module_assignments, path, targets, counter)
        elif isinstance(node, ast.Try):
            _walk_block(node.body, local, module_assignments, path, targets, counter)
            for handler in node.handlers:
                _walk_block(handler.body, local, module_assignments, path, targets, counter)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _walk_block(node.body, local, module_assignments, path, targets, counter)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "register"
            and node.value.args
        ):
            call = node.value
            receiver = _expr_text(call.func.value)
            if not _is_hook_registry(receiver, counter["hook_aliases"]):
                counter["non_hook_register"] += 1
                counter["non_hook_register_detail"].append(
                    {"file": str(path), "line": node.lineno, "receiver": receiver}
                )
                continue
            value = _eval(call.args[0], local, module_assignments)
            if value is None:
                counter["unresolved"] += 1
                counter["unresolved_detail"].append(
                    {"file": str(path), "line": node.lineno, "expr": _expr_text(call.args[0])}
                )
                continue
            if value.startswith("sglang."):
                targets.setdefault(value, str(path))
        if isinstance(node, (ast.withitem,)):
            continue


def scan(paths):
    targets = {}
    counter = {
        "unresolved": 0,
        "unresolved_detail": [],
        "non_hook_register": 0,
        "non_hook_register_detail": [],
        "hook_aliases": set(),
    }
    for path in paths:
        tree = ast.parse(path.read_text())
        counter["hook_aliases"] = _hook_registry_aliases(tree)
        module_assignments = _scope_assignments([n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef, ast.ClassDef, ast.If, ast.Try))])
        for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            _walk_block(func.body, {}, module_assignments, path, targets, counter)
    return targets, counter



class _ModuleIndex:
    def __init__(self, root, minimum_files=1000):
        self.root = root
        self.files = {}
        self.modules = set()
        paths = sorted(root.rglob("*.py"))
        if len(paths) < minimum_files:
            raise SystemExit("official tree looks empty (%d files under %s)" % (len(paths), root))
        for path in paths:
            rel = path.relative_to(root).as_posix()
            module = rel[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            self.files[module] = path
            self.modules.add(module)

    def resolve(self, target):
        parts = target.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:cut])
            if prefix in self.files:
                return prefix, parts[cut:]
        return None, parts


def _module_bindings(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="ignore"))
    plain = set()
    classes = {}
    imports = {}

    def collect(statements):
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                plain.add(node.name)
                if isinstance(node, ast.ClassDef):
                    members = set()
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            members.add(sub.name)
                        elif isinstance(sub, ast.Assign):
                            for target in sub.targets:
                                if isinstance(target, ast.Name):
                                    members.add(target.id)
                        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                            members.add(sub.target.id)
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append("<expr>")
                    classes[node.name] = (members, bases)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        plain.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                plain.add(node.target.id)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    plain.add(alias.asname or alias.name)
                    imports[alias.asname or alias.name] = (node.module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    plain.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.If):
                collect(node.body)
                collect(node.orelse)
            elif isinstance(node, ast.Try):
                collect(node.body)
                for handler in node.handlers:
                    collect(handler.body)
                collect(node.orelse)
                collect(node.finalbody)
            elif isinstance(node, (ast.With, ast.For, ast.While)):
                collect(node.body)
                if isinstance(node, ast.For):
                    collect(node.orelse)
                if isinstance(node, ast.While):
                    collect(node.orelse)

    collect(tree.body)
    return plain, classes, imports


def _bindings(index, module, cache):
    if module not in cache:
        cache[module] = _module_bindings(index.files[module])
    return cache[module]


def _class_members(index, module, name, cache, seen=None):
    if seen is None:
        seen = set()
    key = (module, name)
    if key in seen:
        return set(), []
    seen.add(key)
    plain, classes, imports = _bindings(index, module, cache)
    if name not in classes:
        if name in imports:
            target_module, target_name = imports[name]
            if target_module in index.files:
                return _class_members(index, target_module, target_name, cache, seen)
        return set(), ["%s.%s" % (module, name)]
    members, bases = classes[name]
    collected = set(members)
    outside = []
    for base in bases:
        if base == "<expr>":
            outside.append("%s.<expr>" % module)
            continue
        if base in classes:
            more, far = _class_members(index, module, base, cache, seen)
        elif base in imports:
            target_module, target_name = imports[base]
            if target_module in index.files:
                more, far = _class_members(index, target_module, target_name, cache, seen)
            else:
                more, far = set(), ["%s.%s" % (target_module, target_name)]
        else:
            more, far = set(), ["%s.%s" % (module, base)]
        collected |= more
        outside.extend(far)
    return collected, outside


def _installed_by_plugin(owner, attribute, root):
    path = root / owner if not Path(owner).is_absolute() else Path(owner)
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="ignore"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == attribute:
                    return True
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == attribute
        ):
            return True
    return False


def check_resolvable(targets, official_root, cache, minimum_files=1000, owner_root=None):
    index = _ModuleIndex(official_root, minimum_files=minimum_files)
    resolvable = []
    unresolvable = []
    undetermined = []
    for target, owner in sorted(targets.items()):
        module, rest = index.resolve(target)
        if module is None:
            undetermined.append({"target": target, "owner": owner, "reason": "no_module_prefix_in_official_tree"})
            continue
        plain, classes, imports = _bindings(index, module, cache)
        head, tail = rest[0], rest[1:]
        if head not in plain:
            submodule = module + "." + head
            reason = (
                "missing_submodule"
                if len(rest) > 1 and submodule not in index.files
                else "missing_module_attribute"
            )
            record = {"target": target, "owner": owner, "reason": reason, "module": module, "attribute": head}
            if owner_root is not None and _installed_by_plugin(owner, head, owner_root):
                record["reason"] = "installed_by_plugin_before_register"
                undetermined.append(record)
            else:
                unresolvable.append(record)
            continue
        if not tail:
            resolvable.append(target)
            continue
        if head not in classes:
            undetermined.append(
                {"target": target, "owner": owner, "reason": "head_is_imported_or_non_class_symbol", "module": module, "attribute": head}
            )
            continue
        members, outside = _class_members(index, module, head, cache)
        missing = [name for name in tail if name not in members]
        if missing and outside:
            undetermined.append(
                {
                    "target": target,
                    "owner": owner,
                    "reason": "missing_but_opaque_base:%s" % ",".join(sorted(outside)[:3]),
                    "module": module,
                    "attribute": head,
                    "missing": missing,
                }
            )
            continue
        if missing:
            unresolvable.append(
                {"target": target, "owner": owner, "reason": "missing_class_attribute", "module": module, "attribute": head, "missing": missing}
            )
        else:
            resolvable.append(target)
    return resolvable, unresolvable, undetermined


def main(argv=None):
    parser = argparse.ArgumentParser(description="Scan HookRegistry targets including variable, f-string and concatenation composition.")
    parser.add_argument("--root", type=Path, default=Path("hcu_sglang"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-resolvable",
        action="store_true",
        help="static check that every registered target resolves on the official tree; unresolvable targets fail the run",
    )
    parser.add_argument("--official-root", type=Path, default=Path("../input-code/sglang/python"))
    args = parser.parse_args(argv)
    targets, counter = scan(sorted(args.root.rglob("*.py")))
    payload = {
        "count": len(targets),
        "unresolved": counter["unresolved"],
        "unresolved_detail": counter["unresolved_detail"],
        "non_hook_register": counter["non_hook_register"],
        "non_hook_register_detail": counter["non_hook_register_detail"],
        "targets": dict(sorted(targets.items())),
    }
    exit_code = 0
    if args.check_resolvable:
        resolvable, unresolvable, undetermined = check_resolvable(targets, args.official_root, {}, owner_root=args.root.resolve().parent)
        payload["resolvable"] = resolvable
        payload["unresolvable"] = unresolvable
        payload["resolvable_undetermined"] = undetermined
        if unresolvable:
            exit_code = 1
    if args.output:
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "count": len(targets),
                "unresolved": counter["unresolved"],
                "unresolved_recorded": len(counter["unresolved_detail"]),
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
