#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=chenyue_sgl1103_plugin_minimax
inside=/sgl/${root#/public/home/chenyue3/}
docker exec "$container" env HCU_ADAPTER_HOST=none PYTHONDONTWRITEBYTECODE=1 \
  MAX_JOBS=4 PYTORCH_ROCM_ARCH=gfx938 ROCM_HOME=/opt/dtk bash -euc '
    root=$1
    src=/home/sglang-ws/1020_minimax/all-repo/lightop
    expected=0e7df9a3e8291f2e0aa7392119d9195cd689c9f9
    [[ $(git -c safe.directory="$src" -C "$src" rev-parse HEAD) == "$expected" ]]
    work=$(mktemp -d /tmp/m31-lightop.XXXXXX)
    printf "LightOp source=%s sha=%s build=%s\n" "$src" "$expected" "$work"
    git -c safe.directory="$src" -C "$src" archive "$expected" | tar -xf - -C "$work"
    mkdir -p "$root/runtime/wheels/lightop"
    python3 -m pip wheel --no-deps --no-build-isolation --verbose \
      --wheel-dir "$root/runtime/wheels/lightop" "$work"
    shopt -s nullglob
    wheels=("$root"/runtime/wheels/lightop/lightop-*-cp312-*.whl)
    [[ ${#wheels[@]} == 1 ]]
    sha256sum "${wheels[0]}"
    dest="$root/runtime/lightop-$expected"
    python3 -m pip install --no-deps --no-build-isolation --target "$dest" "${wheels[0]}"
    PYTHONPATH="$dest" python3 -c "
import lightop, torch
from lightop import _router_gemm, minimax_qk_norm_rope, minimax_cp_fp8_pack
from lightop.activation import swiglu_oai_int8_quant_contiguous, swiglu_oai_int8_quant_masked_flat
from lightop.quant.int8 import per_token_quant_int8
assert all(callable(f) for f in (_router_gemm.forward, minimax_qk_norm_rope,
    minimax_cp_fp8_pack, swiglu_oai_int8_quant_contiguous,
    swiglu_oai_int8_quant_masked_flat, per_token_quant_int8))
print(lightop.__file__, lightop.op.__file__, torch.__version__, torch.version.hip)
"
  ' bash "$inside"
