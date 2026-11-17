#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
export SGLANG_HACK_FLASHMLA_BACKEND=kernel
: "${PREFILL_EXPERT_LOCATION:?set PREFILL_EXPERT_LOCATION}"
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
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=1
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export LIGHTOP_PREFILL_SPARSE_PREP_FUSION=3
export SGLANG_DSV4_HC_BF16X3_MIN_ROWS=4096
export SGLANG_HCU_MEGA_MOE_RUNTIME=megamoe
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
export SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION=0.05
export SGLANG_KVALLOC_KERNEL=0
export HSA_USERPTR_FOR_PAGED_MEM=0
export SGLANG_HCU_DSV41_BF16_MAIN_PROJ_PREPACK=1

export NCCL_NET_PLUGIN=shca
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1

readonly PORT=${1:-30000}
readonly MODEL_PATH=${2:-/models/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8-wo_a-markov_head-lm_head}
readonly CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
readonly DIST_TIMEOUT=${DIST_TIMEOUT:-1800}

the_host=$(hostname)

sglang serve \
    --disable-custom-all-reduce \
    --model-path "$MODEL_PATH" \
    --tp-size 8 \
    --ep-size 8 \
    --enable-prefill-cp \
    --attn-cp-size 8 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --cp-strategy interleave \
    --enable-cp-cache-layer-split \
    --enable-decoder-swa-bounded-replay \
    --ep-num-redundant-experts 0 \
    --init-expert-location "${PREFILL_EXPERT_LOCATION}" \
    --host "${PREFILL_HOST}" \
    --port "$PORT" \
    --trust-remote-code \
    --cuda-graph-backend-prefill disabled \
    --enable-cache-report \
    --mem-fraction-static 0.85 \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --dist-timeout "$DIST_TIMEOUT" \
    --max-running-requests 32 \
    --deepep-mode normal \
    --moe-a2a-backend megamoe \
    --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size 5 \
    --speculative-moe-a2a-backend deepep \
    --speculative-moe-runner-backend deep_gemm \
    2>&1 | tee "running_dpsk-v41_PD_P_CP8EP8_pc0.93_${the_host}.log"

# /models/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8-wo_a-markov_head-lm_head

# export ROCBLAS_TENSILE_LIBPATH=/home/run/PD/test/blas_tuning_20261106/packages/gemm_gfx936_80cu_20261106/rocblas/library
# export HIPBLASLT_TUNING_OVERRIDE_FILE=/home/run/PD/test/blas_tuning_20261106/packages/gemm_gfx936_80cu_20261106/hipblaslt/tuning.txt

# export ROCBLAS_TENSILE_LIBPATH=/home/run/PD/test/blas_tuning_20261106/packages/gemm_gfx936_80cu_20261106/rocblas/library
# export HIPBLASLT_TUNING_OVERRIDE_FILE=/home/run/PD/test/blas_tuning_20261106/packages/gemm_gfx936_80cu_20261106/hipblaslt/tuning.txt
# export SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG=/home/run/PD/test/shape32368_opt_20261109/wo_a_algo.json

# export ROCBLAS_TENSILE_LIBPATH=${BLAS_TUNING_ROOT}/rocblas/library
# export HIPBLASLT_TENSILE_LIBPATH=${BLAS_TUNING_ROOT}/hipblaslt/library
# export HIPBLASLT_TUNING_OVERRIDE_FILE=${BLAS_TUNING_ROOT}/hipblaslt/tuning.txt
# export SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG=${BLAS_TUNING_ROOT}/wo_a_algo.json

# export SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION=0.05
