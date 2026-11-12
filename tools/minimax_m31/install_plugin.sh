#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=chenyue_sgl1103_plugin_minimax
inside=/sgl/${root#/public/home/chenyue3/}
plugin="$root/worktrees/plugin"
sha=$(git -C "$plugin" rev-parse --short=12 HEAD)
site="plugin-$sha"
if [[ ! -d "$root/runtime/$site" ]]; then
    git -C "$plugin" archive HEAD | docker exec -i "$container" env HCU_ADAPTER_HOST=none PYTHONDONTWRITEBYTECODE=1 bash -euc '
        stage=$(mktemp -d /tmp/m31-plugin.XXXXXX)
        trap '\''rm -rf -- "$stage"'\'' EXIT
        tar -xf - -C "$stage"
        python3 -m pip install --no-deps --no-build-isolation --target "$1" "$stage"
    ' bash "$inside/runtime/$site"
fi
printf '%s\n' "$site" > "$root/runtime/plugin-site.tmp"
mv -- "$root/runtime/plugin-site.tmp" "$root/runtime/plugin-site"
