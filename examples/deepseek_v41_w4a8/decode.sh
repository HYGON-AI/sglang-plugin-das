#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
: "${DECODE_EXPERT_LOCATION:?set DECODE_EXPERT_LOCATION}"
set -o pipefail
ulimit -c 0

export no_proxy="${no_proxy:-},localhost,127.0.0.1,${PREFILL_HOST},${DECODE_HOST}"
export NO_PROXY="$no_proxy"

export ROCBLAS_TENSILE_LIBPATH=${BLAS_TUNING_ROOT}/rocblas/library
export HIPBLASLT_TENSILE_LIBPATH=${BLAS_TUNING_ROOT}/hipblaslt/library
export HIPBLASLT_TUNING_OVERRIDE_FILE=${BLAS_TUNING_ROOT}/hipblaslt/tuning.txt
export SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG=${BLAS_TUNING_ROOT}/wo_a_algo.json

export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_USE_AITER=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION=1
export SGLANG_DSV41_INT8_SERVICE_OPT=1 
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_HCU_OPT_ENGRAM_GATE=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
export HSA_USERPTR_FOR_PAGED_MEM=0

export NCCL_NET_PLUGIN=shca
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
export SGLANG_HEALTH_CHECK_TIMEOUT=10000
export SGLANG_HCU_FUSED_MHC_BOUNDARY=1

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/models/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8-wo_a-markov_head-lm_head}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}
readonly BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}

the_host=$(hostname)

sglang serve \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --dp 8 \
    --ep 8 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --enable-single-batch-overlap \
    --host "${DECODE_HOST}" \
    --port "$PORT" \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static 0.86 \
    --ep-num-redundant-experts 0 \
    --init-expert-location "${DECODE_EXPERT_LOCATION}" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --moe-runner-backend deep_gemm \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --disable-custom-all-reduce \
    --max-running-requests 64 \
    --cuda-graph-max-bs-decode 64 \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    2>&1 | tee "running_dpsk-v41_PD_D_DP8EP8_${the_host}.log"

# export ROCBLAS_TENSILE_LIBPATH=/home/run/PD/test/blas_tuning_20261106/packages/gemm_gfx936_80cu_20261106/rocblas/library
# export HIPBLASLT_TUNING_OVERRIDE_FILE=/home/run/PD/test/blas_tuning_20261106/packages/gemm_gfx936_80cu_20261106/hipblaslt/tuning.txt

# export ROCBLAS_TENSILE_LIBPATH=/home/run/PD/test/blas_tuning_20261106/packages/gemm_gfx936_80cu_20261106/rocblas/library
# export HIPBLASLT_TUNING_OVERRIDE_FILE=/home/run/PD/test/blas_tuning_20261106/packages/gemm_gfx936_80cu_20261106/hipblaslt/tuning.txt

# export ROCBLAS_TENSILE_LIBPATH=${BLAS_TUNING_ROOT}/rocblas/library
# export HIPBLASLT_TENSILE_LIBPATH=${BLAS_TUNING_ROOT}/hipblaslt/library
# export HIPBLASLT_TUNING_OVERRIDE_FILE=${BLAS_TUNING_ROOT}/hipblaslt/tuning.txt
# export SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG=${BLAS_TUNING_ROOT}/wo_a_algo.json
