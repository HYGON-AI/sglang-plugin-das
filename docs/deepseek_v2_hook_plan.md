# DeepSeek V2 精准 Hook 计划

目标：通过 **method-level REPLACE** + **mixin hook** + **`methods/` 模块** 实现 DCU fork，不修改 upstream 源码。

## 当前结构（P1–P3 完成）

```
sglang_dcu_plugin/models/deepseek/
  hook_utils.py
  deepseek_v2/ / deepseek_v4/   # 子包 __init__.py 含 register_hooks
  deepseek_v2/
    constants.py, module_funcs.py
    methods/{_runtime,mlp,moe,decoder,attention}.py
    hooks/{mlp,moe,decoder,attention}.py
  deepseek_v4/
    constants.py
    methods/{_runtime,mqa,decoder,model,causal_lm}.py
    hooks/{mqa,decoder,model,causal_lm}.py
```

详见 `docs/deepseek_layout.md`。

## 差异概览（origin vs fork）

| 组件 | 变更方法数 | 策略 | 阶段 |
|------|-----------|------|------|
| 模块常量 / 工具函数 | 2 func + 18 attrs | `constants.py` + module patch | **P1 ✅** |
| `DeepseekV2MLP` | 1 (`forward`) | method REPLACE | **P1 ✅** |
| `MoEGate` | 0 | 无需 hook | — |
| `DeepseekV2MoE` | 6 | method REPLACE | **P1 ✅** |
| `DeepseekV2AttentionMLA` | 17 | mixin + class method（见下） | P2 |
| `DeepseekV2DecoderLayer` | 2 | method REPLACE | **P1 ✅** |
| `DeepseekV2Model` | 0 | 与 upstream 一致，不 hook | — |
| `DeepseekV2ForCausalLM` | 0 | 与 upstream 一致，不 hook | — |
| `DeepseekV3/V32ForCausalLM` | 0 | 无需 hook | — |

## 分阶段实施

### P1 — 非 Attention（当前实施）

```
sglang_dcu_plugin/models/deepseek/
  constants.py          # env flags, FORWARD_ABSORB_*, dtype_mapping
  module_funcs.py       # ds_bmm_wrapper, add_forward_absorb_core_attention_backend
  hook_utils.py         # register_method_replaces(), module patch helper
  hooks/
    mlp.py              # DeepseekV2MLP.forward
    moe.py              # DeepseekV2MoE × 6 methods
    decoder.py          # DeepseekV2DecoderLayer.__init__, .forward
    attention.py        # Attention class + mixin methods
  deepseek_v2_impl.py   # 仅作 method body 来源，P2 完成后删除
```

方法体暂从 `deepseek_v2_impl` 引用（`getattr(impl.Cls, "method")`），避免复制 3000 行；P2 起逐段迁入 `hooks/*.py`。

### P2 — Attention 拆 mixin ✅

- **15 mixin method REPLACE**（`forward_mla` / `forward_mha` / `rocm` / `cpu`）
- **2 class method REPLACE**（`DeepseekV2AttentionMLA.__init__`, `prepare_qkv_latent`；`forward_prepare`/`forward_core` 与 upstream 一致，不 hook）
- 移除 `DeepseekV2AttentionMLA` 整类 REPLACE
- fork 内联在 class 上的 method body 仍取自 `deepseek_v2_impl`（P3 再迁出）

### P3 — 删除 impl ✅

- `deepseek_v2_impl.py` 已删除
- 方法体位于各 `deepseek_v2/methods/`、`deepseek_v4/methods/`；共享 import 为 `_runtime.py`
- 重新生成：`python scripts/extract_deepseek_v2_methods.py`（需临时保留 impl 源文件）

## Hook 注册示例

```python
register_method_replaces(
    "sglang.srt.models.deepseek_v2",
    "DeepseekV2MLP",
    ("forward",),
    source_cls=impl.DeepseekV2MLP,
)
```

## 风险点

1. **`__init__` REPLACE**：MoE / DecoderLayer / Attention 的 init 签名有扩展（`rms_weight` 等），需与 upstream 调用方同步（DecoderLayer 传参）。
2. **Mixin MRO**：P2 必须验证 `forward_prepare → forward_absorb_prepare` 走 hook 后的 mixin 而非 stale 引用。
3. **模块常量时序**：`load_plugins()` 须在 model import 之前完成 module patch。
