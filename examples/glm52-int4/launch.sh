#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the local GLM5.2 Channel INT4 checkpoint}"
if [[ -n "${NATIVE_ENV_FILE:-}" ]]; then
  source "$NATIVE_ENV_FILE"
fi
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_W4A8_TPMOE_BACKEND=aiter
export SGLANG_OPT_USE_TOPK_V2=0
export HCU_ADAPTER_HOST=sglang
export SGLANG_PLUGINS=hcu
export HCU_SLIMQUANT=0
export SGLANG_ENABLE_CP_V2=1
export SGLANG_HCU_USE_LIGHTOP_ROPE=1
export SGLANG_DSA_FUSE_TOPK=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_HCU_GLM_CT_BF16_ROUTER=1
paths=()
for key in AITER_SITE FLASH_MLA_SITE SGLANG_PYTHON_ROOT ADAPTER_ROOT PLUGIN_ROOT; do
  value="${!key:-}"
  if [[ -n "$value" ]]; then paths+=("$value"); fi
done
if (( ${#paths[@]} )); then
  prefix="$(IFS=:; echo "${paths[*]}")"
  export PYTHONPATH="$prefix${PYTHONPATH:+:$PYTHONPATH}"
fi
command=("${PYTHON_BIN:-python3}" -m hcu_sglang
  --model-path "$MODEL_PATH" --trust-remote-code
  --tp-size 8 --ep-size 8 --moe-dense-tp-size 1
  --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave
  --nsa-prefill-backend flashmla_auto --nsa-decode-backend flashmla_kv
  --dtype bfloat16 --dist-timeout 10000 --watchdog-timeout 3600
  --page-size 64 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 16384
  --cuda-graph-max-bs 16 --max-running-requests 32
  --reasoning-parser glm45 --tool-call-parser glm47
  --speculative-algorithm EAGLE --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --quantization slimquant_w4a8_marlin --mem-fraction-static 0.9
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30000}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "${LOG_DIR:-logs}"
"${command[@]}" 2>&1 | tee "${LOG_DIR:-logs}/glm52-int4-$(date +%Y%m%d-%H%M%S).log"
