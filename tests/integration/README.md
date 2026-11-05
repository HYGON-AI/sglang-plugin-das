# 集成验证（L1.5 叠加格与后续 L1.6）

本目录是 [CI 分层](../../docs/ci.md#3-验证分层与运行时门禁规划)中需要
GPU + sglang 树 + adapter + plugin 的集成层入口。**不进 `unittest
discover`**（它必须保持无 GPU 全绿），运行配方与证据约定如下。

## L1.5 叠加格（官方 + adapter + plugin）

工具在 adapter 仓（`adapter-das`），本仓不复制门禁脚本，只登记配方：

```bash
# 前提：HCU 容器内已 pip 安装本仓 wheel（entry point 经 importlib.metadata
# 发现，PYTHONPATH 方式无效）；sglang 树干净且与 binding manifest 匹配。
cd /path/to/adapter-das
python3 -m tools.l15_model_smoke \
  --host adapter+plugin \
  --up-tree /path/to/sglang-v0522 \
  --binding-manifest tests/manifests/l15_v0522.json \
  --families dense,moe,mla,mamba,hybrid --modes eager,graph \
  --out results/l15-stacked-v0522/matrix.json
```

要点：

- `--host adapter+plugin` 使 harness 设 `SGLANG_PLUGINS=hcu`（白名单只装
  本 plugin）；`--host adapter`（默认）设哨兵值 `none` 禁装一切 plugin，
  两格互不污染。
- 装载证据是断言不是日志：preflight 与每个 worker 都会断言
  `HookRegistry` 中存在 plugin 名为 `hcu` 的 hook 来源——
  sglang 的 `load_plugins()` 会把 plugin 异常吞成日志行，没有该断言时
  plugin 未装载也会假绿。
- 证据字段：`runtime.host`、`runtime.plugin`（loaded/hook_targets/sources）、
  每 job 的 `evidence.host`。与 adapter 格基线（同树同家族）按 job 级
  PASS/FAIL 对照；差异即 plugin 叠加引入，归本仓修复。

## 状态

| 轮次 | 树 | 格 | 结果 | 证据 |
|---|---|---|---|---|
| 2026-11-05 | sglang v0.5.22（35ce3132 树） | adapter+plugin | **10/10 PASS**（dense/moe/mla/mamba/hybrid × eager/graph，EXIT=0；plugin loaded=True，596 hook targets，sources=['hcu']） | `adapter-das:results/l15-stacked-v0522/matrix.json`、`logs/l15_0522_stacked.log`（bw47/lizhg_dtk2610，DTK 26.10，torch 2.11.0+hip） |

首轮点亮修复了两处（均为 2025 年「为性能引入、框架持续买单」的历史默认，随本轮一起反转）：

1. **KV 池布局**：`SGLANG_KV_LAYOUT_HCU_FA` 默认 true→false。原默认启用 HCU page-major 池
   （`hcu_mha_pool` 4D 布局），官方 triton kernel 期望 3D 官方布局，叠加态首跑即
   `unexpected KV buffer ndim=4`。现默认官方两种布局，vllm_ 前缀 HCU FA 算子的配套布局
   改为显式 opt-in（性能 A/B）。
2. **fork 源 JIT overlay**：新增 `SGLANG_HCU_JIT_OVERLAY`（默认 false）。overlay 的
   `prefer_overlay_headers` 把 fork 头全局前插，官方源（qwen/deepseek fused_rope 等）
   JIT 构建被遮蔽头破坏（moe/mla `Failed to build JIT module`）。DTK 上官方源的编译
   期 shim 由 adapter 负责（OCML 重定向/PDL stripper 等），overlay 仅 fork 语义需要时
   显式开启。

另记录一项 adapter 侧待修（不阻断，已在本轮日志验证为无害）：`prefill_bcg_gate` 的
`registry.register("cuda_graph", …)` 不在 `BACKEND_CATALOG`，登记失败被 bridges 的
except 吞成 warning——wrap 在 register 之前已生效，功能无损，但每轮跑都打噪音，应在
adapter 仓补 catalog 条目或跳过登记。

## L1.6（规划）

特有模型 tiny 家族（bailing/dspark/hunyuan/glm4_moe/minimax_m2/qwen3_5）
与 layer hook 直调断言的配方与清单待建；`tests/runtime/` 现有单点按
「vendor 原生 API 直调归 L1、layer hook 归 L1.6」的边界收编后登记于此。
