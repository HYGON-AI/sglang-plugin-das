#!/usr/bin/env bash
set -euo pipefail
out=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
container=chenyue_sgl1103_plugin_minimax
inside=/sgl/${out#/public/home/chenyue3/}
docker exec "$container" env HCU_ADAPTER_HOST=none PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=8 FLASH_MLA_OPT=0 bash -euc '
  out=$1
  src=/home/sglang-ws/1020_minimax/all-repo/flashmla
  cutlass_rel=csrc/cutlass/cutlass_3.2.1
  cutlass_src=/home/sglang-ws/1020_minimax/flashmla/$cutlass_rel
  expected=$(git -c safe.directory="$src" -C "$src" rev-parse "HEAD:$cutlass_rel")
  actual=$(git -c safe.directory="$cutlass_src" -C "$cutlass_src" rev-parse HEAD)
  test "$expected" = "$actual"
  test -d "$cutlass_src/include"
  printf "CUTLASS %s\n" "$actual"
  test "$(sha256sum "$src/flash_mla/flash_mla_interface.py" | cut -d" " -f1)" = ead47f6fa659781d945708a787268f7521b2a5489b283aa39bbb16d2c0823f0f
  stage=$(mktemp -d /tmp/m31-flashmla.XXXXXX)
  mkdir -p "$out/runtime/wheels"
  tar --exclude=.git --exclude=build --exclude=dist --exclude=__pycache__ -C "$src" -cf - . | tar -C "$stage" -xf -
  tar --exclude=.git -C "$cutlass_src" -cf - . | tar -C "$stage/$cutlass_rel" -xf -
  git -c safe.directory="$src" -C "$src" rev-parse HEAD
  sha256sum "$src/setup.py" "$src/flash_mla/flash_mla_interface.py"
  mkdir "$stage/compiler"
  cat > "$stage/compiler/aicc" <<\EOF
#!/bin/sh
exec /opt/dtk/bin/aicc -Wno-missing-template-arg-list-after-template-kw "$@"
EOF
  chmod +x "$stage/compiler/aicc"
  export PATH="$stage/compiler:$PATH"
  python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir "$out/runtime/wheels" "$stage"
  wheel=$(find "$out/runtime/wheels" -maxdepth 1 -name "flash_mla-*-cp312-*.whl" -print)
  test "$(printf "%s\n" "$wheel" | wc -l)" -eq 1
  test -f "$wheel"
  python3 -m pip install --no-deps --no-build-isolation --target "$out/runtime/flashmla" "$wheel"
  sha256sum "$wheel"
  PYTHONPATH="$out/runtime/flashmla" python3 -c "from flash_mla.flash_mla_interface import msa128_stage1_page_max, flash_mla_with_kvcache; import flash_mla; print(flash_mla.__file__)"
' bash "$inside"
