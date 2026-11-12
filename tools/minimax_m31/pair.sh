#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
baseline=chenyue_sgl1026_minimax
plugin=chenyue_sgl1103_plugin_minimax
base="$(dirname -- "$(dirname -- "$root")")/2MiniMax-M3.1-channel-int4-w4a8-int8-w8a8/2eval_baseline"
replay="$root/performance/paired"
mkdir -p "$replay"
exec 9>"$root/runtime/run.lock"
flock -n 9

started=()
cleanup() {
    local rc=$?
    trap - EXIT
    for c in "${started[@]}"; do
        docker stop "$c" >> "$replay/containers.log" 2>&1 || true
    done
    docker inspect --format '{{.Name}} {{.State.Status}} {{.State.FinishedAt}}' "$baseline" "$plugin"         >> "$replay/containers.log" || true
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
for c in "$baseline" "$plugin"; do
    docker inspect --format '{{.Name}} {{.Image}} {{.Config.Image}}' "$c" >> "$replay/containers.log"
    if [[ $(docker inspect --format '{{.State.Running}}' "$c") != true ]]; then
        docker start "$c"
        started+=("$c")
    fi
done
for c in "$baseline" "$plugin"; do
    docker exec "$c" python3 -S -c '
import json, sys, sysconfig
from importlib.metadata import distributions
from pathlib import Path
installed = {d.metadata["Name"].lower(): d.version for d in
             distributions(path=[sysconfig.get_path("platlib")])}
print(json.dumps({
    "python": sys.version.split()[0],
    "dtk": Path("/opt/dtk/.info/version-dev").read_text().strip(),
    "torch": installed.get("torch"),
    "image_sglang_distribution": installed.get("sglang"),
}, indent=2))
' > "$replay/$c-environment.json"
done
git -C "$base/../1eval_baseline1109/source/sglang" rev-parse HEAD > "$replay/baseline-sha.txt"
sha256sum "$base/bench-rsc.sh" "$base/dai-sh/m31_rsc_bench/protocol.json" > "$replay/client-inputs.sha256"

for stage in prefill decode; do
    for pair in 1 2 3; do
        roles=(baseline plugin)
        if (( pair % 2 == 0 )); then roles=(plugin baseline); fi
        for role in "${roles[@]}"; do
            run="$replay/$stage/pair-$pair/$role"
            if [[ -f "$run/status" && $(cat "$run/status") == passed ]]; then continue; fi
            while :; do
                if [[ -e "$run" ]]; then mv "$run" "$run.failed.$(date +%Y%m%d_%H%M%S_%N)"; fi
                echo "$(date -Is) $stage pair=$pair role=$role"
                if bash "$root/scripts/run_case.sh" "$role" "$stage" "$run" bench; then
                    break
                else
                    rc=$?
                    (( rc == 75 )) || exit "$rc"
                fi
            done
        done
    done
done
python3 "$root/scripts/compare.py" "$replay" "$base"
