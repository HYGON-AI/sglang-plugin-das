# Migration Plan

## Phase 1 — layers/attention
## Phase 2 — layers/moe + quantization
## Phase 3 — models/

### deepseek_v2 (P1–P3 done)

See `docs/deepseek_v2_hook_plan.md`.

### deepseek_v4 (started)

- **Upstream baseline**: `hcu-sglang-now/.../models/deepseek_v4.py` reverted to match origin.
- **Plugin**: 对称子包 `deepseek_v2/`、`deepseek_v4/`（见 `docs/deepseek_layout.md`）
- **Depends on v2**: uses patched `DeepseekV2MoE` via `deepseek_v2` module import.

### deepseek_common + deepseek_nextn + models/utils (done)

- **Upstream baseline**: 以下文件已 revert 至 origin：
  - `deepseek_common/{utils,deepseek_weight_loader,attention_backend_handler}.py`
  - `deepseek_common/attention_forward_methods/forward_mla.py`
  - `deepseek_nextn.py`
  - `models/utils.py`
- **Plugin**: `deepseek_common/`、`deepseek_nextn/`、`model_utils/`
- **Note**: `forward_mla.py` 的 skip_topk 逻辑由既有 `deepseek_v2` attention mixin hook 提供

## Phase 4 — remaining layers
