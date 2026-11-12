#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
stamp=$(date +%Y%m%d_%H%M%S)
queue="$root/checks/workflow-$stamp"
mkdir -p "$queue"
printf '%s\n' "$queue" > "$root/runtime/workflow.path"
status() {
    printf '%s\n' "$1" > "$queue/status.tmp"
    mv "$queue/status.tmp" "$queue/status"
}
trap 'rc=$?; if ((rc)); then printf "%s\n" "$rc" > "$queue/exit-code"; fi' EXIT
trap 'exit 130' INT TERM
for stage in prefill decode; do
    attempt=0
    while :; do
        attempt=$((attempt + 1))
        run="$queue/$stage-attempt-$attempt"
        status "$stage smoke attempt $attempt"
        if flock -w 120 "$root/runtime/run.lock" \
            bash "$root/scripts/run_case.sh" plugin "$stage" "$run" smoke; then
            break
        else
            rc=$?
            if (( rc != 75 )); then
                status "$stage smoke failed"
                exit "$rc"
            fi
        fi
    done
done
status profiling
profile_rc=0
bash "$root/scripts/profile.sh" || profile_rc=$?
printf '%s\n' "$profile_rc" > "$queue/profile-exit-code"
status benchmarking
pair_rc=0
bash "$root/scripts/pair.sh" || pair_rc=$?
printf '%s\n' "$pair_rc" > "$queue/pair-exit-code"
if (( profile_rc || pair_rc )); then
    status verification_failed
    exit 1
fi
status passed
