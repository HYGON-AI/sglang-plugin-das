# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def owner_of(hook):
    module = getattr(hook, "__module__", None)
    record = {"module": module or "", "name": getattr(hook, "__qualname__", ""), "file": ""}
    if module:
        loaded = sys.modules.get(module)
        record["file"] = getattr(loaded, "__file__", "") or ""
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record the HookRegistry targets actually registered by hcu_sglang.register(), with per-target owners.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="also run HookRegistry.apply_hooks() and record applied targets and failures")
    args = parser.parse_args(argv)

    from sglang.srt.plugins.hook_registry import HookRegistry

    recorded = []
    owners = {}
    original = HookRegistry.register.__func__

    def register(cls, target, hook, hook_type=None, **kwargs):
        recorded.append(target)
        owners.setdefault(target, owner_of(hook))
        if hook_type is None:
            return original(cls, target, hook, source=kwargs.get("source"))
        return original(cls, target, hook, hook_type, **kwargs)

    HookRegistry.register = classmethod(register)
    summary = {
        "schema_version": 2,
        "registered": 0,
        "applied": 0,
        "apply_error": None,
        "import_error": None,
        "apply_failures": [],
    }
    try:
        import hcu_sglang

        hcu_sglang.register()
    except Exception as exc:
        summary["import_error"] = f"{type(exc).__name__}: {exc}"
    summary["registered"] = len(recorded)
    summary["targets"] = sorted(set(recorded))
    summary["owners"] = {target: owners[target] for target in summary["targets"]}
    if args.apply and summary["import_error"] is None:
        failures = summary["apply_failures"]
        original_apply_target = HookRegistry._apply_target.__func__

        def apply_target(cls, target, hooks):
            try:
                return original_apply_target(cls, target, hooks)
            except Exception as exc:
                failures.append({"target": target, "error": f"{type(exc).__name__}: {exc}"})
                raise

        HookRegistry._apply_target = classmethod(apply_target)
        try:
            HookRegistry.apply_hooks()
            summary["applied"] = len(HookRegistry._patched)
            summary["applied_targets"] = sorted(HookRegistry._patched)
        except Exception as exc:
            summary["apply_error"] = f"{type(exc).__name__}: {exc}"
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "registered": summary["registered"],
                "applied": summary["applied"],
                "apply_failures": len(summary["apply_failures"]),
                "import_error": summary["import_error"],
                "apply_error": summary["apply_error"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["import_error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
