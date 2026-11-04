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
| `SGLANG_QWEN4_SPLIT_HC_MIX` | 0 | gfx936、BF16、1–16行、HC4/hidden2560/lowrank320采用分阶段HC mix候选 |
| `SGLANG_MOE_C_NATIVE_SORT` | 0 | E=129、top-k=10 的 EP drop-route 保留排序 |
| `SGLANG_QWEN4_PREFILL_DIAGNOSTICS` | 0 | 输出 compact 几何诊断 |
| `SGLANG_PLE_ETP_REDUCE_SCATTER` | 1 | PLE 分片输出按 DP rank 求和并切分 |
| `SGLANG_PLE_ETP_REDUCE_SCATTER_MIN_TOKENS` | 2048 | 小消息保留 all-reduce + scatter |
| `SGLANG_ENABLE_DP_GATHER_INT8` | 0 | Qwen4 FFN 内的独立 INT8 transport 兼容入口 |

优化组合要求 TP=DP=EP、attention TP/CP=1、`SGLANG_DP_USE_GATHERV=1`、无 MoE A2A/TBO、prefill Graph disabled、AITER `moe_c` 与非融合 shared expert。verify/draft/decode 不进入 compact prefill。关闭 compact/local-routing/routed-INT8/native-sort 后回到当前官方边界与插件量化实现，checkpoint 支持继续保留。

验收配方使用 TP4/DP4/EP4、INT8 ngram PLE pinned offload、BF16 计算、FP8 E5M2 KV 存储、fa3、EAGLE 3 steps/4 draft tokens、full decode Graph、prefill Graph disabled。HCU 上 FP8 KV 先转换到 BF16 scratch 再做注意力计算。paged decode 提取传入 layer scale；当前 prefix packing API 没有 scale 参数，本轮 E5M2 checkpoint 使用 unit scale。非 unit-scale FP8 prefix 尚未支持，不能从 paged decode 的非 unit-scale 数学测试推导该能力。pinned PLE 后端受支持；file offload 调用时明确拒绝。

## 128K 性能路径修复

初次功能适配的 CLI 请求 fa3，但官方 `_attention_backend_fa3_fp8_fallback` 将 E5M2 KV 切到 triton；adapter 同时关闭了 in-tree TileLang MQA，实际使用 Torch reference。bw47 四卡、C1、131072 输入/1024 输出、seed42、零缓存命中的预热后三次中位数为 45.825 s TTFT / 142.405 ms TPOT。不能仅根据 CLI 认定实际 backend。

adapter 后续通过公共 MQA 分发和 `qsa_indexer` 消费者接入独立 BoltOPs vendor，并保留 `HCU_DISABLE_OPS=qsa_mqa` 回退。仅修复 MQA、其他配方不变时三次中位数为 29.560 s / 65.182 ms，仍未达到历史 DAS 18.245 s / 12.170 ms。

plugin 在上述具名 fallback pass 上注册 AROUND：仅 gfx936、Qwen4 target/MTP、显式 BF16、page64、启用 HCU FA KV 布局、fa3 + E5M2 组合保留 fa3；其他组合委托当前有效 original。返回声明字典，不写冻结输入。实际 `attention_hook` 预导入别名的参数解析回归覆盖支持与拒绝组合；该反例在修复前被切为 triton。当前服务性能及完整精度复验仍在进行，结果随后补入开发记录。

PLE 预取通过 `Qwen4ExpPLELayer.start_prefetch` 的窄 AROUND 对齐来源：HCU 在当前流 gather，避免将 mapped-host lookup 放在 HIP Graph 侧流；其他设备调用当前有效 original。保留官方 `_get_prefetch_buffer` 的 eager/full/breakable 分配规则、idle rank 的全局 lookup、语义/物理行计数和消费清理。公共构造与 Hook 回归覆盖状态重用拒绝、空/关闭/idle、有效回退，以及真实 INT8/BF16 mapped provider 的变 ID Graph 重放。

HC 输出保留来源的两处独立存储：终端 mix 前的 HC 快照，以及 logits 交给 MTP 前的输出快照。模型 forward 与终端 `_MixStreams` 的窄 Hook 用每次调用的 ContextVar 传递快照，保留官方 residual 完成和 mixer 计算；异常和嵌套模型调用恢复原 scope。非 HCU、非末级 PP、idle 和非 logits 输出沿用有效基线。公共出口测试覆盖 mixer 复用输入、模型缓存随后覆盖，以及真实 HCU grouped norm/mix 的变输入 Graph。此项尚待完整服务性能与精度复验，不能从存储测试推断 TPOT 已达标。

本模型 QSA 的 pending index-key ring 只容纳一个压缩组，官方要求 `speculative_num_draft_tokens <= 4`。GDN kernel 的 width6 数学与 Graph 通过不代表整个 QSA 服务支持6个 draft tokens；不能绕过该守卫增加 MTP 深度。

`SGLANG_QWEN4_SPLIT_HC_MIX=1` 为2026-11-04性能候选：在公共 `hyperconnection.fused_hc_mix` 别名注册窄AROUND，将低秩投影分成partial、reduce+SiLU和up+gate三步，避免persistent kernel的软件全局barrier和atomic累加。仅接受上表形状、连续同设备权重；其他形状、dtype、架构或关闭状态委托当前有效original。上游`fused_hc_mix_supported`的确定性模式守卫继续有效。原型在bw46/DTK2610的1/4/16行随机BF16输入与原persistent输出完全一致，延迟约77→32μs；公共构造、变输入/权重Graph、完整服务性能和GSM8K尚待该候选复验，不能从原型数据推导整模型收益。

## 验证状态

本次结果统一维护于 [开发记录](hcu-sglang-development.md#2026-11-03-qwen4-exp-w4a8-与-pr-586-适配)。算子数学、Graph、Hook 应用和实际服务分别记录；smoke 不代表全量精度或性能复测。
