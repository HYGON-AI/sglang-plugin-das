# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconcile statically scanned hook targets with the targets registered at runtime.")
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--allow-missing", type=Path, help="JSON list of static-only targets with accepted reasons")
    parser.add_argument("--list", action="store_true", help="print every static_only, runtime_only and unexplained target")
    parser.add_argument("--output", type=Path, help="write the full payload including every target list")
    args = parser.parse_args(argv)
    static = set(json.loads(args.static.read_text())["targets"])
    runtime_payload = json.loads(args.runtime.read_text())
    runtime = set(runtime_payload.get("targets") or [])
    static_only = sorted(static - runtime)
    runtime_only = sorted(runtime - static)
    accepted = {}
    if args.allow_missing and args.allow_missing.is_file():
        accepted = {row["target"]: row.get("reason", "") for row in json.loads(args.allow_missing.read_text())}
    unexplained = [target for target in static_only if target not in accepted]
    payload = {
        "static": len(static),
        "runtime": len(runtime),
        "static_only": static_only,
        "runtime_only": runtime_only,
        "accepted": {t: accepted[t] for t in static_only if t in accepted},
        "unexplained_static_only": unexplained,
        "unresolved_static": json.loads(args.static.read_text()).get("unresolved", 0),
        "runtime_only_by_module": dict(
            sorted(
                __import__("collections").Counter(
                    target.rsplit(".", 2)[0] if target.count(".") > 2 else target for target in runtime_only
                ).items()
            )
        ),
    }
    print(json.dumps({k: payload[k] for k in ("static", "runtime", "unresolved_static")} | {
        "static_only": len(static_only), "runtime_only": len(runtime_only), "unexplained": len(unexplained)}, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10))
    if args.list:
        print(
            json.dumps(
                {
                    "static_only": static_only,
                    "runtime_only": runtime_only,
                    "unexplained_static_only": unexplained,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if not unexplained and not runtime_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
