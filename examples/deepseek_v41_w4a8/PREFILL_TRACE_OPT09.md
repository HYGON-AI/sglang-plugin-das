# DeepSeek V4.1 Plugin Prefill 定向优化与复测（P09）

本文记录根据原生 DAS/P08 trace 差异实施的三项 plugin 优化，以及 2026-11-16 的数值、模型快测、prof 和无 profiler 性能结果。源码修改位于节点 `sglang-plugin-das`，adapter 未修改；D/router 沿用原服务。没有提交或推送 Git commit。

## 结果

| 指标 | P08 | P09 | 变化 |
|---|---:|---:|---:|
| 无 profiler 输入吞吐中位数，token/s | 42,177.942 | 52,422.368 | +24.29% |
| 单请求 32K→1 TTFT 中位数，ms | 811.252 | 657.535 | -18.95% |
| TP4 五步 GPU step 中位数，ms（prof） | 759.428 | 605.077 | -20.32% |
| 同一 20 题模型快测 | 19/20 | 20/20 | 快测通过 |

三轮无 profiler 共 384 个请求，输入完整 IDs 与 P08 逐字节一致，每请求输入 32768、输出 1、prefix ratio0；三轮均无错误。另保留三次固定 IDs 的单请求 TTFT。prof 使用独立的同一 128 请求负载，`start_step=2/num_steps=5/activities=CPU,GPU`，保留全部 8 rank。prof 时间不作为无 profiler 成绩。

| 轮次 | 请求数 | 输入吞吐 token/s | 错误 |
|---|---:|---:|---:|
| 1 | 128 | 52,296.960 | 0 |
| 2 | 128 | 52,450.286 | 0 |
| 3 | 128 | 52,422.368 | 0 |

历史 DAS TP4 GPU step 为 506.012 ms；P09 相对它为 +19.58%。历史 DAS 与当前 HIP/NCCL 版本、完整输入身份不同，不能当作严格同软件的新 A/B 性能验收，也没有虚构一轮新的 DAS 无 profiler 吞吐。

## 实施内容

1. **wo_a 算法选择。** 在当前 hipBLASLt 上枚举 283 个支持 M4096/G8/N1024/K4096、workspace≤64MiB 的算法，每个先核 INT32 零误差。原启发式 index6441 重复中位约5.650ms，最佳当前库 index13100/MT256×256×32/WGM5 约0.804ms、workspace0。生产按当前库的精确 kernel name、支持检查和原8MiB上限选择，没有硬编码跨版本编号或在线试跑算法。仅该大形状更换选择；M16 等其他形状保留原启发式。
2. **HCU RoPE 独立准入。** 原代码已有 flat-row 实现和 setter，但 setter 与 CK/AITER 一起受 `_is_hip` 门禁控制。现在单独为 HCU 接通现有 RoPE setter，没有翻转全局平台检测或开放 CK/AITER。逐 token/head 改为 flat-row；kernel 次数不因此减少。
3. **plugin 自有 FP4 前处理。** `_rope_fq4` 对 gfx936/BF16、D128或512、rope64 使用融合 Triton kernel；保留 tail 旋转后的 BF16 舍入，再执行 per32 UE8M0 或 per16 E4M3FN scale、ties-even fake quant。positions 沿用原 gather；输入不原地修改，输出新建，unsupported 保留旧路径。能力没有塞入 adapter 或全局 `sgl_kernel`。

## Kernel 对照

下表是五个 GPU step 内 kernel 累计时间÷5，单位 ms，存在并行，**不能相加当作可回收端到端延迟**。按同 trace runtime correlation 与真实 Python caller 分组，FP4 统计包含相关 compressor/indexer 前处理，不仅看单个新 kernel 名。

| 范围 | DAS 历史 | P08 | P09 | P08→P09 kernel 数（五步） |
|---|---:|---:|---:|---:|
| wo_a 完整投影 | 39.065 | 150.477 | 51.148 | 600 → 600 |
| Q/输出 RoPE | 4.679 | 37.096 | 4.710 | 400 → 400 |
| FP4 compressor/indexer 前处理 | 8.724 | 33.121 | 12.913 | 1595 → 355 |
| MHC 完整边界（含 norm） | 50.254 | 66.167 | 66.263 | 2015 → 2015 |
| 通信 | 42.011 | 68.895 | 74.007 | 855 → 855 |
| MegaMoE 主核 | 207.623 | 205.944 | 213.992 | 200 → 200 |
| BF16 attention | 53.031 | 52.677 | 52.711 | 105 → 105 |
| FP8 paged attention | 1.841 | 2.032 | 2.019 | 95 → 95 |

全部 8 rank 均核到：wo_a native project/quant/scale 各200次，其中大形状 tuned GEMM105次，小形状95次；flat RoPE400次；FP4 fused 每 rank 40–40次；原 BF16 attention105次、FP8 paged95次、Torch paged0。P09 各 rank GPU step 中位数范围为 604.858–605.241 ms，没有挑选最快 rank。

## 重叠和后续空间

| 五步均值，ms | DAS 历史 | P08 | P09 |
|---|---:|---:|---:|
| GPU 无 kernel/copy 覆盖区间均值 | 1.649 | 17.219 | 16.720 |
| 无并行 compute 的通信区间均值 | 28.015 | 48.539 | 53.251 |

| 融合/路径 | 本轮状态 | 保留的边界 |
|---|---|---|
| wo_a quant→GEMM→scale | 保持3个 kernel，改进大形状 GEMM | 不以3 kernel等同性能对齐；quant/scale仍有独立成本 |
| Q/输出 RoPE | flat-row 已实际命中 | FMA舍入顺序与原逐head路径不同，见数值合同 |
| RoPE+FP4 | plugin 融合 kernel 已实际命中 | positions gather 和 unsupported fallback 保留 |
| HCU Mega MHC/CP重排/通信 | 本轮未改 | 未关闭现有 MHC safety guard、盲开 ASM/融合或替换 collective owner |

通信独占区间和 GPU 无覆盖区间不等于全部可移除成本；本轮没有将整 trace 的 profiler 启停复制/分配时间算作模型瓶颈。剩余差距需继续按调用来源和生命周期逐项核验。

## 验证与限制

- wo_a 新选择：14项零容差 BF16 检查，含 M1/2/15/16/31/32/4096、zero/ties/stride，以及 M4096 Graph 三次输入/权重/scale更新和双流。
- RoPE/FP4：77项检查；实际 `_rope_fq4` 入口、positions gather、CPU/FP32回退、Graph和真实非连续视图另20项；节点实际新增 runtime unittest 两项通过。FP4 保持 `rtol=0/atol=0/equal_nan=True`，不宣称 NaN payload 逐位一致。
- RoPE 的 exact 与随后 1-ULP 检查均有原始失败记录。flat 与逐head的 FMA收缩顺序不同，抵消近零处 ULP 数不稳定。最终显式 BF16 合同为 `rtol=2^-7/atol=2^-20`，保留 maxabs/ULP，**没有称 RoPE 逐位等价**；新模型快测为 20/20。
- FP4 初版 NaN反例失败；最终在 scale 计算前显式传播 block NaN，原 exact 门槛未放宽。原失败源码/日志保全。
- Windows候选写入引入 CRLF，首次 diff check拒绝；只将本次候选恢复为与 Git HEAD 一致的 LF 后再通过 L0/guide/diff，不改原始证据或其他仓库文件。原 Torch.library weakref 退出告警保留；实际单测两项为 OK/exit0。
- L0634 Python/8 skills/errors空、guide build/check、diff check通过。未重跑全量 GSM8K/HumanEval/MATH500，也未将快测视为全量精度验收；未补完整长上下文性能矩阵。
- 原实验 FlashMLA provider 保持原字节，仍是本镜像/当前 gfx936/DTK 组合的已测边界，没有扩大为通用 wheel ABI 保证。

## 源码与证据

| 节点 plugin 文件 | SHA256 |
|---|---|
| `hcu_sglang/jit_overlay/csrc/dsv41_int8_projection.cu` | `f9f8f1b32daf3f6ae6b539e3129e7d44a47ff66a0bc6dab99055ce018707e81e` |
| `hcu_sglang/models/deepseek/current_v4/execution/core.py` | `a40110ccc271d9d7578ac78f983fcaacbffc83192af6fc75950eacccb6c573b8` |
| `hcu_sglang/models/deepseek/current_v4/hcu_backend/dsv41_sparse.py` | `9844547afe66cf153a7489f8bd6b36c448118ff7737ca4db392307d6a5ff61e1` |
| `hcu_sglang/models/deepseek/current_v4/hcu_backend/fp4_rope_native.py` | `43c8e94f10bf9f973d33897279f8f117b2bb596c85bc77315673d93e6731c4e4` |
| `tests/runtime/dsv41_prefill_primitives.py` | `aee0b7dd453cc891f03ddfb8e0d5ee0d70c5a90d2bb4a0a4f39b71bd680d7b50` |

新 TP4 trace：`1794832343.259407-TP-4-EP-4.trace.json.gz`，SHA256 `3285f0386b348df3a2977585be922b85524847b24ba20cda7042635c190a36df`。全部8份 trace 的 SHA/大小及 capture request 在 evidence manifest 中。

模型/数值/源码原始证据：`prefill-opt-model09-evidence01.zip`，SHA256 `9c3c8f3f124b2ff3f7a1c90cd5da41040c8581ef0ecc86e528dbf2c42a8b5905`。本地完整成员、原始请求/评分、384输入、native SO/build来源、全TP4 JSON由 `local-proof01.json` 独立核验；`analysis-proof02.json` 另核全部 15574 对 kernel/runtime correlation、原始/filtered 各事件类别 hash 及三处源码逆变换（core/sparse另显式逆LF/CRLF回原runtime字节）。完整未改 raw trace 保留。
