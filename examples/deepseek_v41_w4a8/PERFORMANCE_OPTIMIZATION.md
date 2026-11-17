# DeepSeek V4.1 W4A8 plugin 性能优化记录

更新时间：2026-11-16。本文记录 32K Prefill 的问题、修改、实际验证和性能变化，供后续开发与回归使用。

当前已在全部 8 个 rank 上消除 Prefill attention 的 Torch paged 回退。TP4 的五步 GPU 包络中位数由 **1688.308 ms 降至 863.002 ms，缩短 48.88%**。DAS 历史 trace 为 506.012 ms，当前仍高约 70.55%。这组数值来自 prof，不作为关闭 profiler 后的正式性能成绩。

## 1 运行条件与比较口径

| 项目 | 本次条件 |
|---|---|
| 模型 | DeepSeek V4.1 Flash，Channel INT4 W4A8，含 DSPARK draft |
| 候选软件 | 官方 SGLang 0.5.22 + HCU adapter/plugin；Python 3.12、Torch 2.13、gfx936 |
| P 端并行 | TP8 / EP8 / CP8；LayerSplit 与 bounded replay 保持 |
| 预算 | mem_fraction=0.85；max_total_tokens=262144；max_running_requests=32；chunk32768 |
| 其它配置 | seed55965328；MegaMoE/DeepGEMM、BLAS、DSPARK gamma5/verify6 保持 |
| 请求负载 | 固定 128 组完整 32768 input IDs；每请求输出 1 token |
| 采集范围 | start_step=2、num_steps=5、CPU/GPU、with_stack、record_shapes |
| 统计对象 | `step[EXTEND bs=1 toks=32768]`；保留全部 8 个 rank，TP4 为固定代表 |

GPU 包络按相同 External id 合并各 stream，取最早开始到最晚结束；它不是 kernel duration 求和。跨 stream 的 kernel 累计时间可能重叠，CPU inclusive 时间也不能与 GPU 时间直接相加。

plugin 修改前后使用相同 input IDs、完整 CLI 和除后端开关外相同的环境。DAS 为用户提供的历史 trace，输入 IDs 未提供，Torch/HIP 快照与当前候选不同，因此仅用于路径和耗时参考。

## 2 已完成的优化阶段

| 阶段 | 主要修改 | TP4 GPU 包络中位数 ms | 无 profiler 输入吞吐中位数 token/s |
|---|---|---:|---:|
| 初始 plugin | 原多阶段 MoE 与 Torch attention 路径 | 3983.047 | 未在本记录引用 |
| DeepGEMM 与 Triton sparse Prefill | 接通 W4A8 融合 consumer、布局转换及 Triton sparse，配方同步参考脚本 | 1797.297 | 23582.28 |
| BF16 FlashMLA 原生 Prefill | 私有兼容 provider；BF16 分支走原生，paged 分支仍走 Torch | 1668.576 | 25855.67 |
| 修复前重新采集 | BF16 原生、FP8 paged 仍为 Torch；与下一行使用同一负载 | 1688.308 | 本次未新增 |
| FP8 paged 原生路径修复 | 仅 P 端将 `SGLANG_HACK_FLASHMLA_BACKEND` 改为 `kernel` | **863.002** | **尚未复测** |
| DAS 历史参考 | 用户提供的原 DAS trace | 506.012 | 历史采集受 profiler 扰动，不作正式基线 |

1668.576 ms 与 1688.308 ms 是同一 BF16 原生阶段的两次独立采集窗口。最新修改前后对比使用 1688.308 → 863.002 ms。早期 DeepGEMM/Triton 阶段同时调整多个实现与配方，不把其收益归因于单一修改。历史吞吐结果仅属于对应阶段，不能继承给最新版本。

### DeepGEMM W4A8 路径

原 standalone MegaMoE 经多阶段 barrier、Marlin/Gather/Scatter 路径执行。plugin 接通已有 DeepGEMM W4A8 consumer，将 W13 权重和 scale 成对转换为 tile256 布局，保留空 rank 参与和 routed scaling。相关实现位于 `hcu_sglang/layers/moe/mega_moe.py`。

该阶段结合 BLAS 和服务预算对齐后，GPU 包络从 3983.047 ms 降到 1797.297 ms。相同 kernel 模板仍可能包含同步等待与资源竞争，不能把 device elapsed 全部解释为纯 GEMM 计算时间。

### BF16 sparse Prefill 路径

原 FlashMLA 在当前容器报 `no kernel image available`。先通过 plugin 的 Triton sparse 路径维持功能，再核对原生二进制：原 wheel 已含 gfx936 对象，但原对象的函数解析失败。固定环境中的私有兼容副本调整 28 个 code-object ABI 标记字节后，完成原生执行验证；ISA 与其它字节保持。

这属于当前 gfx936/Torch/DTK 组合的实验兼容部署，不是供应方源码重编或通用兼容方案。原全局安装包保留。BF16 原生阶段通过独立数值/Graph 17 项、公开入口 64 项，并在整模型 trace 中命中 105 次原生 sparse Prefill kernel。

### FP8 paged 路径

Prefill 会复用名为 `_forward_flash_mla_decode` 的 paged 接口。仅设置 `SGLANG_HCU_SPARSE_PREFILL_BACKEND=kernel` 只覆盖 BF16 sparse 分支；当公共后端仍为 `torch` 时，Prefill 内的另一分支仍执行 Torch paged、KV 解包和 accumulator。

确认原生 FP8 paged 数值与 Graph 后，在节点 plugin 的 `examples/deepseek_v41_w4a8/prefill.sh` 中，紧接 `source "${SCRIPT_DIR}/env.sh"` 增加：

```bash
export SGLANG_HACK_FLASHMLA_BACKEND=kernel
```

实际 P 启动环境同时保留已有的 `SGLANG_HCU_SPARSE_PREFILL_BACKEND=kernel`。共享 `env.sh` 保留 `torch` 供 D 使用；该轮仅重启 P，adapter、Python runtime、KV 布局、数学实现以及 D/router 均保持。

## 3 最新 prof 的实际路径与收益

| TP4 五步合计 | 修改前 plugin | 修改后 plugin | DAS 历史参考 |
|---|---:|---:|---:|
| BF16 Sparse Prefill 原生调用 | 105 | 105 | 105 |
| FP8 paged 原生调用 | 0 | 95 | 95 |
| Torch paged 函数调用 | 95 | 0 | 0 |
| cache accumulator 调用 | 190 | 0 | 0 |
| Torch KV unpack 调用 | 950 | 0 | 0 |
| GPU 包络中位数 ms | 1688.308 | 863.002 | 506.012 |
| FP8 paged 原生 kernel 累计 ms | 未走此路径 | 10.122 | 9.203 |

全部 8 个 rank 均为每步 21 次 BF16 + 19 次 FP8、五步合计 105 + 95 次原生调用；Torch paged、accumulator 和 unpack 均为 0。新版本各 rank 的 GPU 包络中位数范围为 **862.746–863.394 ms**，未挑选最快 rank。

这次修改使 TP4 包络减少约 825.306 ms。attention 的两类实际路径已与 DAS 对应，整步仍有约 356.990 ms 差距；该差值不能直接归因于某个算子或 plugin 开销。

## 4 验证结果与精度边界

| 检查 | 实际结果 | 结论范围 |
|---|---|---|
| FP8 paged 独立数值与 Graph | 13 项通过；rtol=.02、atol=.004；最大绝对误差 .001953125 | H64/D512、584B KV、SWA+extra、ragged、无效索引/NaN/stride、3 次 Graph KV/query 更新 |
| 整模型加载 | 8 scheduler 均映射同一私有 SO；effective 配置同修改前；router 连续健康 90 秒 | 当前 P 端运行环境 |
| 同组 GSM8K 快速检查 | 修改前 20/20，修改后 19/20 | sample12 标准答案 13，实际回答 12；完整原始响应保留，根因未确定 |
| 32K→1 请求 | 128/128 完成 | 当前固定负载与输出协议 |
| prof 完整性 | 8 rank 文件清单/SHA/gzip 通过；TP4 另做完整 JSON 逐步解析 | 确认原生路径与采集范围 |
| 工程检查 | bash、L0（630 Python/8 skills）、导览构建/新鲜度、git diff check 通过 | 修改与交付检查 |

20 题快测的相对得分变化为 5%，不能替代完整 HumanEval/GSM8K 验收；新后端尚未做全量精度复测。此前完整数据集得分属于旧运行版本，不作为本次原生 FP8 路径的通过结论。

## 5 复现与回退

复现时先核对当前 provider、源码和配置清单，再启动 P。实际加载来源、完整 CLI、raw/effective 配置、输入 IDs 与 prof 参数均需留档；不能只凭环境变量或 kernel 名称判断命中。

抓取同档位 32K 压测的 prof 请求参数为：

```json
{
  "start_step": 2,
  "num_steps": 5,
  "activities": ["CPU", "GPU"],
  "with_stack": true,
  "record_shapes": true
}
```

`output_dir` 指定本次独立目录，服务地址使用实际 P 端；在 `/home/run/test/weiguan_test/bench.sh` 对应的 32K 档位触发采集。本次复测复用了其 `prefill_throughput.py` 的固定 128 请求和完整 IDs，不把默认整套压测描述为已经重新运行。

回退本轮后端修改时，仅 P 改回 `SGLANG_HACK_FLASHMLA_BACKEND=torch`，核对实际 PID/PGID/start_ticks 归属后重启 P。已加载 SO 不做热替换；共享 D 环境保持原值。回退私有 BF16 provider 属于上一阶段的独立操作，按对应历史报告执行。

## 6 证据位置与身份

本地记录位于工作区 `doc/`，新 trace 和核验文件位于 `doc/prof/20261116-flashmla-paged-kernel-prefill32k/`。节点全部 rank trace 位于本次运行根目录下的 `flashmla-model07-profile01/traces/`；运行根目录由实际部署清单确定。

| 文件或对象 | 身份 |
|---|---|
| 新 TP4 trace | `1794816376.5560822-TP-4-EP-4.trace.json.gz` |
| 新 trace SHA256 | `66c951b1869d7d7cbd91723733ea02e749a2bd7a2c43e1ee521217f8be6a907e` |
| 修改前 TP4 trace SHA256 | `1becdd430c808ab686fc101d5e01053de5b89130f2b57c78a83bfebffabace71` |
| DAS TP4 trace SHA256 | `a72bd48e9e176834e519bb9bb59c71324feab2b06f3b44682b489a1c27dbcf1c` |
| 私有 FlashMLA SO SHA256 | `d0d4018a4e6d63573e0b55b9cd435320415c41f4ac94b37bc8bdd1403cb7bd7b` |
| 原全局 SO SHA256 | `7ad05bfbb2fa67148b3e6bd6cf288f54ea1baa074438dc457d9d69b90bb92301` |
| 模型/请求/prof 证据包 SHA256 | `648667350225c4c570c2de904ee8cc1c724776e9a37811a43c972e4e66ba4bf0` |
| 源码/报告补包 SHA256 | `28b3ccb3749945b17595cae7e50b4302b647bde1b670ea7a85fd01247f7b387d` |

本地 `local-proof01.json` 保存逐请求与 trace 复核，`capture-metadata.json` 保存全部 8 rank 清单，`delivery01/local-proof01.json` 保存节点源码与报告交付核验。原始详细报告分别为 `DeepSeek-V4.1-W4A8-32K-Prefill性能对比报告.md`、`FlashMLA-gfx936原生Prefill修复与验证.md`、`DeepSeek-V4.1-Attention原生路径修复与prof.md`；在节点示例目录中，后两项对应 `FLASHMLA_NATIVE_PREFILL.md`、`ATTENTION_NATIVE_PATH_FIX.md`。

## 7 后续优化与记录要求

下一轮优先用最新 trace 比较 MegaMoE W4A8、通信等待、CPU 提交与小算子；先关联依赖和各 rank 到达时间，再选一个有证据的修改点。旧 trace 的剩余 kernel 累计不能直接当作最新版本的瓶颈排序。

正式性能复测需关闭 profiler，并固定同负载、输入、配置和资源；新版本全量精度结论需独立复测。每次新增优化记录问题证据、最小修改、源码/provider 身份、数值/Graph/整模型结果及全部 rank prof，分别保存正式性能和 prof 数据。

运行时能力与后端优化继续归 plugin。使用仓库 `skills/hcu-plugin-dev/SKILL.md` 组织能力实现与回归，使用 `skills/hcu-plugin-triage/SKILL.md` 定位已有路径回归；输入为固定源码、启动日志、请求协议和 trace，交付需说明实际命中、收益和未验证范围。

## 8 2026-11-16 wo_a 原生分组 INT8

截图红框的 G8 wo_a 逐 group Torch 路径已迁到 plugin 自有 HIP C ABI：融合量化、hipBLASLt strided batch GEMM、融合缩放。MQALayer19 代表调用 64→3 个 kernel，DAS 为 3；全8rank五步均200次 native project/quantize/scale。CPU/G1/非支持形状保留回退，adapter不改。

数值14项零容差全过，新模型快测19/20；新TP4 GPU step 759.428ms，上一版863.002ms，历史DAS 506.012ms。无profiler三轮384请求输入吞吐中位数42177.942token/s，相对P06 +63.13%，该对比合计包含FP8 paged和wo_a，不全部归因本轮。新完整数据集精度和新DAS无profiler基线未复测，性能未宣称完全对齐。

详细报告：本地 `DeepSeek-V4.1-wo_a原生迁移与性能复测.md`；节点 `WOA_NATIVE_PATH_FIX.md`。新trace `1794823528.1714485-TP-4-EP-4.trace.json.gz`，SHA `866469038411c3d51905d1936a0ca5c4f1eb0ebe75281e9b33fca864d71bd1c7`；单位数值/模型证据与原两次失败保全。

### 8.1 性能收益汇总（已测量）

固定负载：输入32768 IDs、输出1、并发128、prefix_ratio=0；P/D、TP/EP/CP8、seed及预算保持。以下两类比较分别记录，避免将累计收益全部记到本轮 wo_a。

**本轮 wo_a 单项变化：P07 → P08，来自同负载 prof。**

| 指标 | P07 修改前 | P08 修改后 | 下降幅度 |
|---|---:|---:|---:|
| MQALayer19代表 wo_a GPU kernel 数 | 64 | 3 | 95.31% |
| wo_a CPU wrapper平均耗时 | 4.229ms | 0.324ms | 92.33% |
| 同层代表 wo_a GPU时间并集 | 11.402ms | 7.082ms | 37.89% |
| TP4 GPU step包络中位数 | 863.002ms | 759.428ms | 12.00% |

代表调用固定为TP4/MQALayer19/第3个记录EXTEND，CPU与GPU通过HIP runtime correlation关联。CPU inclusive、kernel时间并集和跨stream step包络是不同指标，不能相加。prof测量仅用于路径及耗时对照，不用作正式吞吐成绩。

**累计端到端收益：P06 → P08，来自无profiler复测。包含 FP8 paged原生化和 wo_a原生化两项。**

| 指标 | P06 | P08 | 收益 |
|---|---:|---:|---:|
| 三轮输入吞吐中位数 | 25855.666 token/s | 42177.942 token/s | +63.13% |
| 3次32K→1单请求TTFT中位数 | 1.318s | 0.811s | 降低38.44% |

P08三轮输入吞吐分别为 42127.032、42186.200、42177.942 token/s；每轮128请求，共384请求，错误0。三组完整输入及controller与P06逐SHA核同。P07没有单独新的无profiler成绩，不能据此给出“wo_a单项吞吐提升63.13%”；当前也不是交替多轮A/B实验。

**正确性与剩余差距。** 原生算子14项数值/Graph/双流零容差通过；P07/P08同组模型快测均19/20，完整数据集未重新验证。历史DAS TP4 step为506.012ms，P08仍高50.08%；代表GEMM为5.645ms而DAS为0.842ms。不能因3个kernel已相同就宣称DAS性能完全对齐；下一优化点是当前hipBLASLt同shape算法与workspace选择。

收益依据：`woa-model08-evidence01.zip` SHA `46233af02dea4f4272d2a5d62de755218a84aa2ce5d894c66024c817cfe0076e`；新TP4 trace SHA `866469038411c3d51905d1936a0ca5c4f1eb0ebe75281e9b33fca864d71bd1c7`；本地完整JSON/correlation/评分/输入与TTFT复核见 `doc/prof/20261116-woa-native-prefill32k/local-proof01.json` 和 `same-operator-comparison01.json`。本节只记录现有结果，没有新增压测、精度、prof或服务重启。

## 9 2026-11-16 P09：wo_a、RoPE 与 FP4 定向优化

本轮按 P08 与历史 DAS trace 的实际调用差异实施三项 plugin 优化，adapter 未改，D/router 沿用原服务。启动脚本、环境、seed、预算和完整输入 IDs 与 P08 对齐。

### 9.1 修改与实际命中

- **wo_a 算法选择**：在当前 hipBLASLt 上比较283个支持算法并核对 INT32 结果；M4096/G8/N1024/K4096 改选 MT256×256×32/WGM5、workspace0。生产按当前库精确 kernel name 和支持检查选择，不照搬跨版本编号或在线测速；其他形状保留原选择。
- **RoPE**：单独为 HCU 接通现有 flat-row 路径，不翻转全局平台检测或连带开启 CK/AITER。
- **FP4 前处理**：plugin 自有 Triton kernel 融合 tail RoPE 和 fake quant，保留 BF16 中间舍入、per32 UE8M0/per16 E4M3FN scale、ties-even及 NaN 传播。positions gather和不支持输入的旧回退保留。

全8rank五步记录均核到：wo_a native project/quant/scale各200次，其中大形状 tuned GEMM105次、小M16原路径95次；flat RoPE400次；新FP4 fused40次；BF16 attention105次、FP8 paged95次、Torch paged0。

### 9.2 端到端收益：P08 → P09

固定负载：输入32768 IDs、输出1、并发128、prefix_ratio=0；无profiler三轮共384请求，另测3次固定输入的单请求TTFT。prof另用同一128请求、start_step2/num_steps5、CPU+GPU，保留全部rank；不将prof吞吐当正式成绩。

| 指标 | P08 | P09 | 变化 |
|---|---:|---:|---:|
| 无profiler三轮输入吞吐中位数 | 42177.942 token/s | 52422.368 token/s | +24.29% |
| 32K→1单请求TTFT中位数 | 811.252ms | 657.535ms | 降低18.95% |
| TP4五步GPU step中位数（prof） | 759.428ms | 605.077ms | 降低20.32% |

三轮吞吐分别为52296.960、52450.286、52422.368 token/s，384请求错误0；P09所有rank的GPU step中位数范围604.858–605.241ms，未挑选最快rank。

### 9.3 路径耗时与收益归属

下表为五步GPU kernel累计时间÷5，单位ms，包含并行工作，不能相加作为端到端可回收延迟。

| 范围 | P08 | P09 |
|---|---:|---:|
| wo_a完整投影 | 150.477 | 51.148 |
| 其中GEMM | 120.371 | 21.254 |
| Q/输出RoPE | 37.096 | 4.710 |
| FP4 compressor/indexer前处理 | 33.121 | 12.913 |

FP4前处理相关kernel数1595→355，其中新融合kernel40次，不能将其表述为“300个原生HIP算子已全部迁入”。RoPE当前4.710ms与历史DAS4.679ms接近；kernel数仍为400，收益来自flat-row布局。

### 9.4 正确性、限制与证据

- 新wo_a选择14项零容差检查通过，含大形状Graph输入/权重/scale更新及双流。RoPE/FP4原语77项、实际入口20项和节点runtime unittest两项通过。
- FP4维持rtol0/atol0/equal_nan；RoPE因FMA顺序差异使用明确BF16合同rtol2^-7/atol2^-20，不声称逐位相等。早期exact/1ULP及NaN反例失败记录保留。
- 新模型同20题快测20/20，P08为19/20；未重跑全量GSM8K/HumanEval/MATH500或完整长上下文矩阵，快测不替代全量精度验收。
- 历史DAS TP4 step506.012ms，P09仍高19.58%；历史软件版本和完整输入身份不同，不据此宣称严格同环境性能已对齐。
- MHC、普通W8A8和通信本轮未改。后续按最新trace的调用、依赖和owner继续定位，不能盲开ASM/融合或关闭安全门禁。

详细报告：[P09 Prefill定向优化与复测](PREFILL_TRACE_OPT09.md)。新TP4 trace：`1794832343.259407-TP-4-EP-4.trace.json.gz`，SHA `3285f0386b348df3a2977585be922b85524847b24ba20cda7042635c190a36df`。

完整219成员模型/数值/源码证据ZIP SHA `9c3c8f3f124b2ff3f7a1c90cd5da41040c8581ef0ecc86e528dbf2c42a8b5905`；本地独立核验见 `doc/prof/20261116-prefill-opt09/local-proof01.json`，全15574对kernel/runtime及事件保留核验见 `analysis-proof02.json`。本次仅补充文档记录，没有新增模型、精度、压测、prof或服务重启。


## 10. P09截图step间与后半层LayerSplit空泡修复（P16，2026-11-17）

本轮在相同32K→1/128请求/原并发与预算下，P16同组快测20/20，无profiler384请求0失败；吞吐中位55,155.501token/s，相对P09顺序复测+5.21%。P10复用完整Extra payload、P11后台异步SWA页索引快照、P12 Full索引分裂有界view、P13设备端常量sentinel以及P16 CPU Gloo捕获层共识复用已在节点plugin源码落地；adapter/D/router未改。

P16 rank4稳态四间隙/ms：0.311 / 0.309 / 0.317 / 0.322；同突发窗口：193.195 / 0.000 / 33.678 / 0.317。窗口分开报告，不将采集时机变化计为代码收益。所有8rank调用数active-pages30/compact-broadcast220，三处目标阻塞拷贝/分配消失。详见[P16空泡报告](PREFILL_BUBBLE_FIX_P16.md)。历史DAS不是本次重锁定的交替A/B，完整数据集未重测，不宣称全量精度或完全性能对齐。

P15 CPU-tail异步metadata试验实际prof挂起/无有效性能，负证据保留，P16已恢复P13 backend完整字节与GPU完成边界；不交付该未通过试验。


## P17–P18：有界native out、bootstrap空归约与残余控制等待（2026-11-17）

详见 [P18完整报告](PREFILL_BUBBLE_FIX_P18.md)。仅P plugin源码变化；adapter/D/router/原预算不动。P17构造期有界FP32 workspace+私有native out消除492MiB冷分配与44.155秒step；原native数学逆字节与实际installed差分27、workspace9/Graph/跨流通过。P18只省略单rank Gloo identity归约，真实跨rank MIN-ready→读prefix→MAX-prefix顺序保持，真实四进程32case/rank及packaged回归通过。新out每rank五步25次、原dense数学入口0；Extra active-pages200→30、compact390→220；后19层GPU包络46.941→38.365ms。

P18固定same20=20/20，三轮无prof384请求错误0；吞吐中位55114.928 token/s，相对P09累计+5.14%，相对P17约-0.12%，不宣称最后一项有额外吞吐收益。single中位653.386ms。完整数据集没有重跑。首steady rank4间隙0.322/0.324/0.321/31.945ms，独立复核33.868/9.793/13.476/0.330ms；24份全rank raw保留，不能只报最好的步骤。残余为准入/分页/真实CPU共识及到达偏斜，OS/GIL/profiler分别贡献未单独证明；全部间隔对齐DAS约0.3ms尚未达成。P15异步tail失败试验不交付，保留原GPU完成语义。


## P19：Sparse Prefill cache-builder 标量同步消除（2026-11-17）

HCU CP 容量断言改用已有 CPU 长度/interleave 元数据；原生 producer、物理 padding、数学和 GPU 完成语义不改，非 HCU 保留原分支。详见 [P19 完整报告](PREFILL_SPARSE_QUERY_SYNC_P19.md)。

同组20/20；无prof384请求零错误，吞吐中位 55206.683 token/s（相对P18 +0.166%，累计相对P09 +5.311%），单条 644.027ms。两独立窗口全8rank×5step：目标sum/item/scalar/sync全0。CPU140+3、真实HCU70+1和Graph3通过；有效metadata零容差。

本次只确认截图目标调用域，无全模型所有step空泡消除或完整新数据集授予；其他准入/CPU协议间隙见报告原值。运行policy58为87b451...，backend041c17...；原P18负链和native1–4 fixture失败保留。CPU wheel18042f8d，独立资格包60e4cd6f。
