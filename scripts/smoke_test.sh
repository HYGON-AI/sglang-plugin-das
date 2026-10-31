#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SGLANG_PLUGINS=hcu
python3 -m pytest tests/unit -v
