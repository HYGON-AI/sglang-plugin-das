#!/usr/bin/env bash
set -euo pipefail
: "${ADAPTER_ROOT:?set ADAPTER_ROOT}"
: "${PLUGIN_ROOT:?set PLUGIN_ROOT}"
: "${PROVIDER_PYTHONPATH:?set PROVIDER_PYTHONPATH}"
: "${MOONCAKE_ROOT:?set MOONCAKE_ROOT}"
: "${PREFILL_HOST:?set PREFILL_HOST}"
: "${DECODE_HOST:?set DECODE_HOST}"
: "${BLAS_TUNING_ROOT:?set BLAS_TUNING_ROOT}"
test -f "${ADAPTER_ROOT}/sitecustomize.py"
test -f "${PLUGIN_ROOT}/hcu_sglang/__init__.py"
test -r "${MOONCAKE_ROOT}/sdk/lib/libibverbs.so.1"
test -r "${MOONCAKE_ROOT}/sdk/lib/libshca-rdmav34.so"
test -d "${MOONCAKE_ROOT}/sdk/include"
test -d "${MOONCAKE_ROOT}/site/mooncake"
export HCU_ADAPTER_HOST=sglang
export SGLANG_PLUGINS=hcu
export HCU_DISABLE_OPS=aiter_ar
export HCU_DEEPEP_BRIDGE=0
export SGLANG_HACK_FLASHMLA_BACKEND=torch
export SGLANG_ENABLE_NAN_LOGITS_CHECK=1
export PYTHONPATH="${ADAPTER_ROOT}:${PLUGIN_ROOT}:${PROVIDER_PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export LD_PRELOAD="${MOONCAKE_ROOT}/sdk/lib/libibverbs.so.1:${MOONCAKE_ROOT}/sdk/lib/libshca-rdmav34.so"
export LD_LIBRARY_PATH="${MOONCAKE_ROOT}/sdk/lib:${MOONCAKE_ROOT}/site/mooncake${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPATH="${MOONCAKE_ROOT}/sdk/include${CPATH:+:${CPATH}}"
