# Hook Targets

Map sglang model/layer is_dcu branches to layers/ or models/ impl modules.
Each impl module exposes register_hooks() alongside its DCU code.

## models/deepseek_v2

| Target | Hook | Source |
|--------|------|--------|
| `DeepseekV2MLP.forward` | REPLACE | `deepseek_v2/methods/mlp.py` |
| `DeepseekV2MoE.*` (6 methods) | REPLACE | `deepseek_v2/methods/moe.py` |
| `DeepseekV2DecoderLayer.__init__/forward` | REPLACE | `deepseek_v2/methods/decoder.py` |
| `DeepseekV2Model/ForCausalLM.forward` | — | 与 upstream 一致，不 hook |
| Attention class + mixin methods | REPLACE | `deepseek_v2/methods/attention.py` |
| `ds_bmm_wrapper`, `add_forward_absorb_*` | REPLACE | `deepseek_v2/module_funcs.py` |
| `_is_dcu`, `_use_fused_*`, … | module setattr | `deepseek_v2/constants.py` |

Full plan: `docs/deepseek_v2_hook_plan.md`

## models/deepseek_v4

| Target | Hook | Source |
|--------|------|--------|
| `MQALayer.*` (4 methods) | REPLACE | `deepseek_v4/methods/mqa.py` |
| `DeepseekV4DecoderLayer.hc_pre/hc_post/forward` | REPLACE | `deepseek_v4/methods/decoder.py` |
| `DeepseekV4Model.__init__/forward` | REPLACE | `deepseek_v4/methods/model.py` |
| `DeepseekV4ForCausalLM.*` (4 methods) | REPLACE | `deepseek_v4/methods/causal_lm.py` |
| `_is_dcu`, `_use_dpskv4_*`, `op`, … | module setattr | `deepseek_v4/constants.py` |

## models/deepseek_common

| Target | Hook | Source |
|--------|------|--------|
| `DeepseekV2WeightLoaderMixin.do_load_weights` | REPLACE | `deepseek_common/methods/weight_loader.py` |
| `DeepseekV2WeightLoaderMixin.post_load_weights` | REPLACE | `deepseek_common/methods/weight_loader.py` |
| `handle_attention_dcu_mla` + `dcu_mla` registry | module patch | `deepseek_common/methods/attention_backend_handler.py` |
| `deepseek_common.utils._is_dcu` | module setattr | `deepseek_common/constants.py` |
| `forward_mla` skip_topk gate | REPLACE (mixin) | `deepseek_v2/methods/attention.py` |

## models/deepseek_nextn

| Target | Hook | Source |
|--------|------|--------|
| `DeepseekModelNextN.__init__/forward` | REPLACE | `deepseek_nextn/methods/model.py` |

## models/utils

| Target | Hook | Source |
|--------|------|--------|
| `enable_fused_set_kv_buffer` | REPLACE | `model_utils/methods/utils.py` |
