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
    run="$root/traces/$case/$stamp"
    if ! M31_GRAPH_MODE="${case#*-}" bash "$root/scripts/run_case.sh" plugin "${case%-*}" "$run" profile; then
        result=1
    fi
done
exit "$result"
