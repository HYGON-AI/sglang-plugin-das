#!/usr/bin/env bash
set -euo pipefail
: "${PREFILL_HOST:?set PREFILL_HOST}"
: "${DECODE_HOST:?set DECODE_HOST}"
exec env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    HCU_ADAPTER_HOST=none SGLANG_PLUGINS=__ifb_no_general_plugins__ \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= PYTHONPATH= \
    "${ROUTER_PYTHON:-python3}" -m sglang_router.launch_router \
    --host "${PREFILL_HOST}" --port "${ROUTER_PORT:-30003}" --pd-disaggregation \
    --prefill "http://${PREFILL_HOST}:30000" 8998 --decode "http://${DECODE_HOST}:30000" \
    --policy round_robin --prometheus-port "${ROUTER_METRICS_PORT:-29003}" \
    --request-timeout-secs 120000 --disable-retries
