# Qwen4-Exp W4A8 与 INT8 PLE 适配

> **用途**：说明 Qwen3.8-Flash-Next W4A8 checkpoint 在官方 v0.5.22 与 adapter 组合下的接入、开关和验证范围。开发入口为 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md)，输入是固定的三仓版本、checkpoint、TP/DP/EP 和执行资源；交付为两仓实现、回归和本次服务证据。

## 固定来源与职责

官方基线为 `baa9befa381511cd4d56f83980951e5a45fffc6d`，plugin/adapter 分别从 `sglang_v0.5.22` 的 `fa592b5` / `8bed438` 开分支。迁移依据为 DAS `35719b43ce247a2ddb72297a62f95554d828f56a`（PR #586 当前 head）及 `optimized-compact-local-int8-with-mm-fix.patch`。运行时不导入 DAS 仓库；源码哈希见 [来源清单](qwen4-w4a8-source.json)。

plugin 负责 INT8 PLE 表/逐行 scale、ETP reduce-scatter、compact prefill、局部 BF16 routing、INT8 activation 通信与 AITER 消费、保留所有远端 route 的 native sort、QSA/FLA host 元数据复用、MTP 量化投影与分组归一化。adapter 提供 DTK mapped-host tensor、QSA scatter、HIP QSA 预处理/MQA 路由和 gfx936 QSA prefill LDS 配置护栏。插件自己的 HCU FA KV 布局由插件 QSA 提取内核读回 NHD scratch。

官方当前的 StageBoundary 继续拥有 FFN 出口归约与 residual 更新。局部 routing 在量化前执行；global expert 收到 gathered INT8 codes/scales 一次，shared expert 使用反量化 BF16。异常会清理 ContextVar，接口不通过捕获 TypeError 重试。

官方已有多模态 `mm_input_embeds` 补齐、MTP packed-module mapping、draft Graph 框架和部分 host 长度处理，沿用这些当前实现。未复制旧版整层 forward 或整套 scheduler。

## 配置与有效基线

安装本分支 adapter 和 plugin，令 `PYTHONPATH` 指向固定官方源码，使用 `SGLANG_PLUGINS=hcu` 与 `python -m hcu_sglang.launch_server`。共享容器可用独立 venv，安装后核查实际模块路径。

| 变量 | 默认值 | 用途 |
|---|---|---|
| `SGLANG_QWEN4_COMPACT_PREFILL` | 0 | 纯 EXTEND/IDLE 时使用 SUM_LEN，idle rank 保留一行 |
| `SGLANG_QWEN4_LOCAL_ROUTING` | 0 | compact prefill 在本 rank 原始 BF16 上选专家 |
| `SGLANG_QWEN4_ROUTED_INT8` | 0 | 局部 routing 后传输 INT8 codes/scales，直接交给 AITER |
| `SGLANG_MOE_C_NATIVE_SORT` | 0 | E=129、top-k=10 的 EP drop-route 保留排序 |
| `SGLANG_QWEN4_PREFILL_DIAGNOSTICS` | 0 | 输出 compact 几何诊断 |
| `SGLANG_PLE_ETP_REDUCE_SCATTER` | 1 | PLE 分片输出按 DP rank 求和并切分 |
| `SGLANG_PLE_ETP_REDUCE_SCATTER_MIN_TOKENS` | 2048 | 小消息保留 all-reduce + scatter |
| `SGLANG_ENABLE_DP_GATHER_INT8` | 0 | Qwen4 FFN 内的独立 INT8 transport 兼容入口 |

优化组合要求 TP=DP=EP、attention TP/CP=1、`SGLANG_DP_USE_GATHERV=1`、无 MoE A2A/TBO、prefill Graph disabled、AITER `moe_c` 与非融合 shared expert。verify/draft/decode 不进入 compact prefill。关闭 compact/local-routing/routed-INT8/native-sort 后回到当前官方边界与插件量化实现，checkpoint 支持继续保留。

验收配方使用 TP4/DP4/EP4、INT8 ngram PLE pinned offload、BF16 计算、FP8 E5M2 KV 存储、fa3、EAGLE 3 steps/4 draft tokens、full decode Graph、prefill Graph disabled。HCU 上 FP8 KV 先转换到 BF16 scratch 再做注意力计算。paged decode 提取传入 layer scale；当前 prefix packing API 没有 scale 参数，本轮 E5M2 checkpoint 使用 unit scale。非 unit-scale FP8 prefix 尚未支持，不能从 paged decode 的非 unit-scale 数学测试推导该能力。pinned PLE 后端受支持；file offload 调用时明确拒绝。

## 验证状态

本次结果统一维护于 [开发记录](hcu-sglang-development.md#2026-11-03-qwen4-exp-w4a8-与-pr-586-适配)。算子数学、Graph、Hook 应用和实际服务分别记录；smoke 不代表全量精度或性能复测。
