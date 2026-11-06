#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Offline structural gate: no third-party dependencies or runtime imports."""
from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def migration_header_errors(root: Path) -> list[str]:
    errors = []
    targets = {}
    declaration = re.compile(r"SPDX-License-Identifier:|SPDX-FileCopyrightText:|copyright\b|licensed under|Apache License", re.I)
    for name in ("runtime_owned_sources.json", "owned_model_sources.json"):
        manifest = root / "tests/contracts" / name
        if not manifest.is_file():
            errors.append(f"missing migration provenance contract: {manifest.relative_to(root)}")
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for section in ("files", "extracted_files", "adapter_files"):
                for row in data.get(section, []):
                    target = row.get("target", row.get("owned"))
                    if not target:
                        raise ValueError("migration row has no target")
                    sources = [row] if section == "files" else row.get("sources", [])
                    if not sources:
                        errors.append(f"{target}: migration provenance missing sources")
                    for source in sources:
                        if not isinstance(source.get("source"), str) or not source["source"].strip():
                            errors.append(f"{target}: migration provenance missing source path")
                        if not re.fullmatch(r"[0-9a-f]{64}", source.get("source_sha256", "")):
                            errors.append(f"{target}: migration provenance missing valid source SHA256")
                        lines = source.get("source_header_lines")
                        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                            errors.append(f"{target}: migration provenance missing original source header metadata")
                            continue
                        expected = "保留来源声明" if declaration.search("\n".join(lines)) else "来源无声明"
                        if source.get("source_declaration") != expected:
                            errors.append(f"{target}: missing or inconsistent source declaration status")
                        if section != "adapter_files":
                            targets.setdefault(target, []).extend(lines)
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(f"{manifest.relative_to(root)}: invalid migration provenance: {exc}")
    for target, source_lines in sorted(targets.items()):
        path = root / target
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{target}: cannot read migrated file: {exc}")
            continue
        header = []
        for line in text.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                break
            header.append(line)
        for line in dict.fromkeys(source_lines):
            if line.strip() and line not in header:
                errors.append(f"{target}: missing original source header line: {line}")
    return errors


def runtime_closure_errors(root: Path) -> list[str]:
    command = [sys.executable, str(root / "tools/build_runtime_closure.py"),
               "--triage", str(root / "plans/hcu-runtime-triage-20261105.csv"),
               "--decisions", str(root / "plans/sglang-das-decisions.json"),
               "--output", str(root / "plans/hcu-runtime-target-closure-20261105.csv"), "--check"]
    try:
        result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    except OSError as exc:
        return [f"runtime closure check could not run: {exc}"]
    if result.returncode:
        return [f"runtime closure check failed: {(result.stderr or result.stdout).strip()}"]
    return pending_decision_errors(root)


def pending_decision_errors(root: Path) -> list[str]:
    output = root / "plans/hcu-runtime-target-closure-20261105.csv"
    if not output.is_file():
        return []
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    missing = [row["path"] for row in rows if row.get("execution_status") == "missing_decision"]
    if not missing:
        return []
    return [f"missing decision records: {len(missing)} files have a triage row but no decision entry"]


def main() -> int:
    errors = migration_header_errors(ROOT)
    errors.extend(runtime_closure_errors(ROOT))
    errors.extend(pending_decision_errors(ROOT))
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not any(part in {".git", "artifacts", "__pycache__", ".venv", "build", "dist", ".pytest_cache"}
                         for part in p.relative_to(ROOT).parts)
             and not any(part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)]
    python_files = [p for p in files if p.suffix == ".py"]
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    skills = sorted((ROOT / "skills").glob("*/SKILL.md"))
    for path in skills:
        text = path.read_text(encoding="utf-8")
        match = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
        if not match:
            errors.append(f"{path}: missing frontmatter")
            continue
        fields = dict(re.findall(r"^(name|description):\s*(.+)$", match[1], re.M))
        name = fields.get("name", "")
        if name != path.parent.name or not re.fullmatch(r"[a-z0-9-]{1,63}", name):
            errors.append(f"{path}: invalid or mismatched name")
        if not fields.get("description"):
            errors.append(f"{path}: missing description")
        meta = path.parent / "agents/openai.yaml"
        if not meta.is_file() or "$" + name not in meta.read_text(encoding="utf-8"):
            errors.append(f"{path}: UI metadata/default prompt missing")
    for path in files:
        if path.suffix != ".md":
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            relative = unquote(target.split("#", 1)[0])
            if relative and not (path.parent / relative).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link {target}")
    for forbidden in ("sglang_hcu_plugin", "sgl_kernel", "sitecustomize.py", "usercustomize.py"):
        if (ROOT / forbidden).exists():
            errors.append(f"legacy or global startup artifact outside current scope: {forbidden}")
    runtime = ROOT / "hcu_sglang"
    if runtime.exists():
        for required in ("hcu_sglang/__init__.py", "hcu_sglang/__main__.py", "pyproject.toml"):
            if not (ROOT / required).is_file():
                errors.append("incomplete current runtime layout: " + required)
        for path in runtime.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                modules = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                           [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                if any(name == "sglang_hcu_plugin" or name.startswith("sglang_hcu_plugin.") for name in modules):
                    errors.append(f"legacy namespace import: {path.relative_to(ROOT)}:{node.lineno}")
    elif (ROOT / "pyproject.toml").exists():
        errors.append("packaging exists without the supported hcu_sglang runtime")
    report = {"status": "failed" if errors else "passed", "python_files": len(python_files),
              "skills": len(skills), "errors": errors}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(errors))


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10+ required")
    raise SystemExit(main())
