# DeepSeek V2/V4 插件目录结构

DeepSeek 相关插件按模型/模块划分子包：

```
sglang_hcu_plugin/models/
  model_utils/                 # sglang.srt.models.utils
    constants.py
    methods/utils.py           # enable_fused_set_kv_buffer
  deepseek/
    hook_utils.py              # 共享 hook 工具 + post-apply module patch
    deepseek_v2/               # deepseek_v2.py
    deepseek_v4/               # deepseek_v4.py
    deepseek_common/           # deepseek_common/*
    deepseek_nextn/            # deepseek_nextn.py
```

## deepseek_v2 / deepseek_v4

对称布局：`constants.py`、`methods/`、`hooks/`。

## deepseek_common

| 上游文件 | 插件 |
|---------|------|
| `deepseek_weight_loader.py` | `methods/weight_loader.py` + hooks |
| `attention_backend_handler.py` | `methods/attention_backend_handler.py` + post-apply 注册 `hcu_mla` |
| `utils.py` (`_is_hcu`) | `constants.py` + module setattr |
| `forward_mla.py` (skip_topk) | 已由 `deepseek_v2/methods/attention.py` mixin hook 覆盖 |

## deepseek_nextn

| Target | Source |
|--------|--------|
| `DeepseekModelNextN.__init__/forward` | `deepseek_nextn/methods/model.py` |

## models/utils

| Target | Source |
|--------|--------|
| `enable_fused_set_kv_buffer` | `model_utils/methods/utils.py` |

重新生成 method 体：

```bash
python scripts/extract_deepseek_v2_methods.py
python scripts/extract_deepseek_v4_methods.py
python scripts/extract_deepseek_common_methods.py   # 从 hcu-sglang-now fork 读取
```
