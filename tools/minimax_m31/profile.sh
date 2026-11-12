#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
baseline=chenyue_sgl1026_minimax
plugin=chenyue_sgl1103_plugin_minimax
stamp=$(date +%Y%m%d_%H%M%S)
exec 9>"$root/runtime/run.lock"
flock -n 9
started=()
cleanup() {
    local rc=$?
    trap - EXIT
    for c in "${started[@]}"; do docker stop "$c" || true; done
    docker inspect --format '{{.Name}} {{.State.Status}} {{.State.FinishedAt}}' "$baseline" "$plugin" || true
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
for c in "$baseline" "$plugin"; do
    if [[ $(docker inspect --format '{{.State.Running}}' "$c") != true ]]; then
        docker start "$c"
        started+=("$c")
    fi
done
cases=("$@")
if (( ${#cases[@]} == 0 )); then cases=(prefill-eager decode-eager decode-full); fi
result=0
for case in "${cases[@]}"; do
    case "$case" in
        prefill-eager|decode-eager|decode-full) ;;
        *) echo "Unknown profile case $case" >&2; exit 2 ;;
    esac
    attempt=0
    while :; do
        attempt=$((attempt + 1))
        run="$root/traces/$case/$stamp-attempt-$attempt"
        if M31_GRAPH_MODE="${case#*-}" bash "$root/scripts/run_case.sh" plugin "${case%-*}" "$run" profile; then
            break
        else
            rc=$?
            if (( rc != 75 )); then
                result=1
                break
            fi
        fi
    done
done
python3 - "$root/traces" "$stamp" <<'PY_INDEX'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
stamp = sys.argv[2]
index = {}
for case in ("prefill-eager", "decode-eager", "decode-full"):
    attempts = []
    for run in sorted((root / case).glob(stamp + "-attempt-*")):
        status = run / "status"
        data = dict(path=str(run.relative_to(root)),
                    status=status.read_text().strip() if status.exists() else "unknown")
        for filename in ("PASS.json", "CAPTURE.json"):
            manifest = run / filename
            if manifest.is_file():
                evidence = json.loads(manifest.read_text())
                data.update(manifest=str(manifest.relative_to(root)), runtime=evidence["runtime"],
                            completed_requests=evidence["completed_requests"],
                            completion_rate=evidence["completion_rate"],
                            verified_ranks=len(evidence["traces"]))
                break
        attempts.append(data)
    index[case] = attempts
tmp = root / ("TRACE_INDEX-" + stamp + ".tmp")
tmp.write_text(json.dumps(index, indent=2) + "\n")
tmp.replace(root / ("TRACE_INDEX-" + stamp + ".json"))
PY_INDEX
exit "$result"
