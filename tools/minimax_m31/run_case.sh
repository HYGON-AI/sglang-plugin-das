#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
role=${1:?baseline or plugin}
mode=${2:?prefill or decode}
run=${3:?result directory}
action=${4:-bench}
graph=${M31_GRAPH_MODE:-default}
port=11010
baseline=chenyue_sgl1026_minimax
baseline_dir=/sgl/9minimax/2MiniMax-M3.1-channel-int4-w4a8-int8-w8a8/2eval_baseline
inside=/sgl/${root#/public/home/chenyue3/}
case "$mode" in prefill) short=p ;; decode) short=d ;; *) exit 2 ;; esac
case "$role" in
    baseline) container=$baseline; script="$baseline_dir/$mode/service/0serve_$short.sh" ;;
    plugin) container=chenyue_sgl1103_plugin_minimax; script="$inside/scripts/0serve_$short.sh" ;;
    *) exit 2 ;;
esac
case "$action" in smoke|bench|profile) ;; *) exit 2 ;; esac
[[ "$run" == "$root/"* ]]
inside_run="$inside/${run#"$root/"}"
community="$(dirname -- "$(dirname -- "$root")")/sglang-community"
community_pin=297e54fd105e974cdf401e5b4d20253a22c16f90
mkdir -p "$run"
record_community() {
    local point=$1
    GIT_OPTIONAL_LOCKS=0 git -C "$community" rev-parse HEAD > "$run/community-$point.sha" &&
    GIT_OPTIONAL_LOCKS=0 git -C "$community" status --porcelain=v1 > "$run/community-$point.status" &&
    [[ $(cat "$run/community-$point.sha") == "$community_pin" &&
       ! -s "$run/community-$point.status" ]]
}
status() { printf '%s\n' "$1" > "$run/status.tmp"; mv "$run/status.tmp" "$run/status"; }
stop_owned() {
    local target=$1 kind=$2
    docker exec "$target" bash -c '
        p=$(cat "$1/$2.pid" 2>/dev/null) || exit 0
        [[ $p =~ ^[0-9]+$ && -r /proc/$p/environ ]] || exit 0
        tr "\0" "\n" < /proc/$p/environ | grep -Fxq "M31_RUN_PATH=$1" || exit 0
        kill -TERM -- "-$p" 2>/dev/null || true
        for ((n=0; n<15; n++)); do
            kill -0 -- "-$p" 2>/dev/null || exit 0
            sleep 1
        done
        kill -KILL -- "-$p" 2>/dev/null || true
    ' bash "$inside_run" "$kind"
}
watch_gpus() {
    local foreign
    while sleep 30; do
        hy-smi --showpids > "$run/gpu-pids.last" 2>&1 || continue
        docker top "$container" -eo pid > "$run/container-pids.last" 2>&1 || continue
        foreign=$(comm -23 \
            <(awk '/^PID:/ {p=$2} /HCU Index:[[:space:]]*\[[^]]/ && p {print p}' \
                "$run/gpu-pids.last" | sort -u) \
            <(awk 'NR>1 && $1 ~ /^[0-9]+$/ {print $1}' "$run/container-pids.last" | sort -u))
        if [[ -n $foreign ]]; then
            printf '%s unexpected GPU process(es):
%s
' "$(date -Is)" "$foreign" > "$run/foreign-gpu.log"
            stop_owned "$container" service || true
            if [[ $action != smoke ]]; then stop_owned "$baseline" bench || true; fi
            return
        fi
    done
}
gpu_watch=
cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n $gpu_watch ]]; then
        kill "$gpu_watch" 2>/dev/null || true
        wait "$gpu_watch" 2>/dev/null || true
    fi
    if [[ -s "$run/foreign-gpu.log" ]]; then
        cat "$run/foreign-gpu.log" >&2
        rc=75
    fi
    stop_owned "$container" service || true
    if [[ $action != smoke ]]; then stop_owned "$baseline" bench || true; fi
    if [[ $role == plugin ]] && ! record_community after; then
        echo "Community checkout no longer matches the clean pinned revision" >&2
        rc=1
    fi
    if (( rc != 0 )); then status failed; fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
status waiting_for_gpus
until hy-smi --showmemuse --showuse > "$run/gpu.log" &&
    awk '/HCU memory use|HCU use/ { ++n; if ($NF != 0) bad=1 } END { exit !(n==16 && !bad) }' "$run/gpu.log"; do
    sleep 30
done
if ss -ltnH "sport = :$port" | grep -q .; then
    echo "Port $port is already in use" >&2
    exit 1
fi
status starting
if [[ $role == plugin ]]; then
    cat "$root/runtime/plugin-site" > "$run/plugin-site.txt"
    lightop="$root/runtime/lightop-0e7df9a3e8291f2e0aa7392119d9195cd689c9f9/M31-VERIFIED"
    if [[ -s "$lightop" ]]; then
        cat "$lightop" > "$run/lightop-provider.txt"
    else
        printf '%s\n' image > "$run/lightop-provider.txt"
    fi
    git -C "$root/worktrees/adapter" rev-parse HEAD > "$run/adapter-sha.txt"
    if ! record_community before; then
        echo "Expected clean SGLang community revision $community_pin" >&2
        exit 1
    fi
    cp "$run/community-before.sha" "$run/community-sha.txt"
fi
profile_stage=
if [[ $action == profile ]]; then
    [[ $role == plugin && ( $graph == eager || $graph == full ) ]]
    profile_stage=$mode
fi
docker inspect --format '{{.Image}} {{.State.Status}}' "$container" > "$run/image.txt"
docker exec "$container" env M31_DIAG_NO_OVERLAP="${M31_DIAG_NO_OVERLAP:-0}" \
    M31_PROFILE_STAGE="$profile_stage" M31_GRAPH_MODE="$graph" M31_RUN_PATH="$inside_run"     M31_LOG_FILE="$inside_run/runtime.log"     setsid --wait bash -euc '
        echo "$$" > "$M31_RUN_PATH/service.pid"
        exec bash "$1"
    ' bash "$script" > "$run/service.log" 2>&1 &
service=$!
watch_gpus &
gpu_watch=$!
ready=0
for ((n=0; n<240; n++)); do
    [[ ! -s "$run/foreign-gpu.log" ]] || exit 75
    if ! kill -0 "$service" 2>/dev/null; then
        wait "$service"
        exit 1
    fi
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/server_info" > "$run/server-info.json" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
(( ready == 1 ))
if [[ $action == smoke ]]; then
    status generating
    for attempt in 1 2; do
        curl -fsS --max-time 180 "http://127.0.0.1:$port/generate" -H 'Content-Type: application/json'           -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8,"ignore_eos":true}}'           > "$run/generate-$attempt.json"
    done
    python3 - "$run/generate-1.json" "$run/generate-2.json" <<'PY'
import json, sys
a, b = [json.load(open(p)) for p in sys.argv[1:]]
assert a["text"] and a["text"] == b["text"]
assert all(x["meta_info"]["completion_tokens"] == 8 for x in (a, b))
PY
elif [[ $action == profile ]]; then
    status profiling
    client_rc=0
    timeout 1800 docker exec "$baseline" env M31_RUN_PATH="$inside_run" setsid --wait bash -euc '
        echo "$$" > "$M31_RUN_PATH/bench.pid"
        export PYTHONPATH=/sgl/9minimax/2MiniMax-M3.1-channel-int4-w4a8-int8-w8a8/1eval_baseline1109/source/sglang/python:/usr/local/
        export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        exec python3 "$@"
    ' bash "$inside/scripts/profile/profile_client.py" --mode "$mode" \
        --output "$inside_run/measurements" --trace-dir "$inside_run/traces" \
        > "$run/bench.log" 2>&1 || client_rc=$?
    python3 "$root/scripts/profile/validate.py" "$run" "$mode" "$graph"
    (( client_rc == 0 ))
else
    status benchmarking
    sizes=(1 2 4 8 16 32 40 48 64)
    flags=()
    if [[ $mode == prefill ]]; then flags=(--shared-prefix); fi
    timeout 21600 docker exec "$baseline" env M31_RUN_PATH="$inside_run" setsid --wait bash -euc '
        echo "$$" > "$M31_RUN_PATH/bench.pid"
        exec bash "$@"
    ' bash "$baseline_dir/bench-rsc.sh" --mode "$mode" "${flags[@]}" --bs "${sizes[@]}"         --output "$inside_run/measurements" > "$run/bench.log" 2>&1
    for size in "${sizes[@]}"; do test -s "$run/measurements/bs$size/PASS.json"; done
fi
status passed
