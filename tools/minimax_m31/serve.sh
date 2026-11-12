#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
stage=$1
shift
mode=${M31_GRAPH_MODE:-default}
model=${M31_MODEL_PATH:-/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8}
assets=/home/sglang-ws/1020_minimax
community=/sgl/9minimax/sglang-community
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export SGLANG_PROFILE_V2=0
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu_minimax_m31 HCU_SLIMQUANT=0
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
site=$(cat "$root/runtime/plugin-site")
export PYTHONPATH="$root/worktrees/adapter:$root/runtime/$site:$community/python"
if [[ -d "$root/runtime/flashmla/flash_mla" ]]; then
    export PYTHONPATH="$root/runtime/flashmla:$PYTHONPATH"
    export HCU_PLUGIN_MINIMAX_M31_FLASHMLA=1
fi
export SGLANG_USE_INT4_W4A8=1 SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1 SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_W4A8_TPMOE_BACKEND=lightop SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1 AITER_AR_TRANSPORT=ipc HCU_AITER_AR=1
export HCU_DEEPEP_BRIDGE=0 HCU_PLUGIN_MINIMAX_M31_DEEPEP=1 SGLANG_GROUPGEMM=1
export GPU_ARCHS=gfx938 ROCSHMEM_GDR_DISABLE_HDP=1
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export SGLANG_EPLB_AUTO_START_RECORDER=0 SGLANG_USE_DEEPGEMM_MOE=0
export SGLANG_USE_LIGHTOP=1 SGLANG_USE_LIGHTOP_OAI_INT8_QUANT=1
export LIGHTOP_PER_TOKEN_INT8_METHOD=native W8A8_SUPPORT_METHODS=3
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1 SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=1
export SGLANG_MINIMAX_INDEX_FP8=dot SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA=1
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export MSA_STAGE1_KERNEL=v8 FLASH_MLA_MSA128_FP8_SPLIT_POLICY=verify8
export SGLANG_SIMULATE_ACC_LEN=0
opts=(
    --model-path "$model" --quantization slimquant_w4a8_marlin --trust-remote-code
    --random-seed 42 --tp-size 8 --pp-size 1 --ep-size 8 --moe-dense-tp-size 1
    --moe-a2a-backend deepep --moe-runner-backend deep_gemm
    --deepep-dispatcher-output-dtype int8 --disable-shared-experts-fusion
    --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --page-size 128
    --max-running-requests 80 --max-total-tokens 1192960
    --cuda-graph-backend-prefill disabled
    --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
    --host 0.0.0.0 --port "${M31_PORT:-11010}"
)
case "$stage" in
    prefill)
        [[ $mode == default || $mode == eager ]]
        export HCU_PREFILL_CP=1
        export HIPBLASLT_TUNING_OVERRIDE_FILE="$assets/config/w4a8-w8a8/tuning_20261105/blaslt/hipblaslt.prefill.accepted.config"
        export ROCBLAS_TENSILE_LIBPATH="$assets/config/w4a8-w8a8/tuning_20261105/rocblas/prefill/candidate_library_gpu6"
        export SGLANG_HCU_DEEPEP_NO_M_INDICES_SYNC=1
        opts+=(
            --attn-dp-size 1 --attn-cp-size 8 --enable-prefill-cp --cp-strategy zigzag
            --disable-overlap-schedule --attention-backend fa3 --cuda-graph-backend-decode disabled
            --deepep-mode normal
            --deepep-config "$assets/minimax-sh/merged-runtime/prefill/configs/deepep-sms64-only.json"
            --enable-eplb --init-expert-location "$assets/config/eplb/m31_w4a8_cp8_ep8_rsc_bs4_20261109_nmz107.pt"
            --eplb-algorithm deepseek --eplb-rebalance-num-iterations 1000000
            --eplb-rebalance-layers-per-chunk 4 --expert-distribution-recorder-mode stat
            --expert-distribution-recorder-buffer-size 16
            --context-length 1048576 --chunked-prefill-size 65536 --max-prefill-tokens 65536
            --mem-fraction-static 0.90
        )
        ;;
    decode)
        graph=full
        case "$mode" in
            eager) graph=disabled ;;
            default|full) ;;
            *) exit 2 ;;
        esac
        export HCU_PREFILL_CP=0
        export HIPBLASLT_TUNING_OVERRIDE_FILE="$assets/config/w4a8-w8a8/dspark-int8-20261107/hipblaslt.decode.accepted.config"
        export ROCBLAS_TENSILE_LIBPATH="$assets/config/w4a8-w8a8/tuning_20261105/rocblas/decode/candidate_library_gpu6"
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
        export SGLANG_MINIMAX_DSPARK_INT8_KV_ONLY=1 SGLANG_RAGGED_VERIFY_MODE=static
        export SGLANG_DSPARK_FOLDED_PROPOSAL=0 SGLANG_DSPARK_USE_GREEDY_DRAFT=0
        export M31_STAGE1_GRAPH_MAX_KV_LEN=148608
        opts+=(
            --attn-dp-size 8 --enable-dp-lm-head --enable-dp-attention-local-control-broadcast
            --attention-backend triton --prefill-attention-backend triton --decode-attention-backend fa3
            --deepep-mode auto --cuda-graph-backend-decode "$graph" --cuda-graph-bs-decode 1 2 4 5 6 8 9 10
            --speculative-algorithm DSPARK --speculative-draft-model-path "$model/dspark-channel-int8-smooth-20261107"
            --speculative-draft-model-quantization compressed-tensors --speculative-draft-attention-backend triton
            --speculative-dspark-block-size 7 --speculative-num-draft-tokens 8
            --context-length 163840 --chunked-prefill-size 8192 --mem-fraction-static 0.95
        )
        ;;
    *) exit 2 ;;
esac
if [[ ${1:-} == --print-args ]]; then
    printf '%q ' "${opts[@]}"
    printf '\n'
    exit
fi
if [[ ${1:-} == --resolve-config ]]; then
    shift
    exec python3 -c "
import json, sys
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args
load_plugins()
cfg = prepare_server_args(sys.argv[1:])
cfg.resolve_once()
print(json.dumps(cfg.resolved_dict(), default=str, sort_keys=True, indent=2))
" "${opts[@]}" "$@"
fi
exec python3 -m sglang.launch_server "${opts[@]}" "$@"
