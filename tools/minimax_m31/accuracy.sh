#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run=${1:?accuracy run directory}
[[ $run == "$root/accuracy/"* ]]
client=chenyue_sgl1026_minimax
plugin=chenyue_sgl1103_plugin_minimax
workspace=$(dirname -- "$(dirname -- "$root")")
base="$workspace/2MiniMax-M3.1-channel-int4-w4a8-int8-w8a8/3eval_accuracy1110"
inside=/sgl/${root#/public/home/chenyue3/}/${run#"$root/"}
inside_base=/sgl/${base#/public/home/chenyue3/}
started=()
child=
original_site=
status() { printf '%s\n' "$1" > "$run/status.tmp"; mv "$run/status.tmp" "$run/status"; }
cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n $original_site ]]; then
        printf '%s\n' "$original_site" > "$root/runtime/plugin-site.restore"
        mv "$root/runtime/plugin-site.restore" "$root/runtime/plugin-site"
    fi
    for container in "${started[@]}"; do docker stop "$container" || rc=1; done
    docker inspect --format '{{.Name}} {{.State.Status}} {{.State.FinishedAt}}' \
        "$plugin" "$client" > "$run/containers-final.log" || true
    printf '%s\n' "$rc" > "$run/exit-code"
    if [[ $(cat "$run/status") != interrupted && $(cat "$run/status") != complete ]]; then
        status failed
    fi
    exit "$rc"
}
cancel() {
    if [[ -n $child ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    status interrupted
    exit 130
}
trap cleanup EXIT
trap cancel INT TERM

status waiting_for_lock
exec 9>"$root/runtime/run.lock"
flock 9
status waiting_for_gpus
until timeout -k 2 10 hy-smi --showmemuse --showuse > "$run/gpu-util.last" 2>&1 &&
      awk '/HCU memory use|HCU use/ {++n; if ($NF != 0) bad=1} END {exit !(n==16 && !bad)}' "$run/gpu-util.last" &&
      timeout -k 2 10 hy-smi --showpids > "$run/gpu-pids.last" 2>&1 &&
      awk '/HCU Index:[[:space:]]*\[[^]]/ {found=1} END {exit found}' "$run/gpu-pids.last"; do
    sleep 30
done
for container in "$plugin" "$client"; do
    if [[ $(docker inspect --format '{{.State.Running}}' "$container") != true ]]; then
        docker start "$container"
        started+=("$container")
    fi
done
original_site=$(cat "$root/runtime/plugin-site")
bash "$root/scripts/install_plugin.sh"
sha256sum "$workspace/evalscope_more_client.sh" > "$run/client.sha256"
docker exec "$client" env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
    python3 -c 'import importlib.metadata as m; print(m.version("evalscope"))' > "$run/evalscope-version.txt"
docker exec "$client" env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
    python3 -c 'import hashlib,json,sys; from pathlib import Path; print(*(e["path"] for e in json.loads(Path(sys.argv[1]).read_text()) if hashlib.sha256(Path(e["path"]).read_bytes()).hexdigest() != e["sha256"]), sep="\n")' \
    "$inside_base/checks/dataset-manifest.json" > "$run/dataset-hash-mismatch.txt"
if grep -q '[^[:space:]]' "$run/dataset-hash-mismatch.txt"; then exit 1; fi
docker exec "$client" env DRY_RUN=1 HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
    EVAL_OUTPUT_DIR="$inside/dry-run" EVAL_PORT=11010 EVAL_MAX_TOKENS=32768 \
    bash /sgl/9minimax/evalscope_more_client.sh true humaneval,gsm8k,hmmt_nov25 \
    > "$run/client-dry-run.log" 2>&1
result=0
for stage in prefill decode; do
    status "$stage"
    for attempt in 1 2 3; do
        lane="$run/$stage/attempt-$attempt"
        mkdir -p "$lane"
        printf '%s\n' "$lane" > "$run/$stage/current.path"
        graph=eager
        skip=0
        if [[ $stage == decode ]]; then graph=full; skip=1; fi
        M31_GRAPH_MODE="$graph" M31_DIAG_SKIP_STARTUP_WARMUP="$skip" \
            bash "$root/scripts/run_case.sh" plugin "$stage" "$lane" accuracy &
        child=$!
        rc=0
        wait "$child" || rc=$?
        child=
        printf '%s\n' "$rc" > "$lane/exit-code"
        (( rc == 75 )) || break
    done
    if (( rc != 0 )); then result=1; fi
    python3 "$root/scripts/summarize_accuracy.py" "$run"
done
status complete
exit "$result"
