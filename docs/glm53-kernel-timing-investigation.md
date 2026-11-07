# GLM5.3 decode kernel 异常耗时追查（2026-11-06）

## 结论

现有 profile 同时包含稳定的名称/耗时错位嫌疑和真实的整段延迟。不能继续按 kernel 名称汇总就认定 RMSNorm、SiLU 或 FlashMLA combine 自身退化。真实额外时间主要落在 MoE dispatch → compute → combine 区间。本轮静态调用链确认并修正了第二个调度差异：adapter 的 CUDA 兼容标记绕过原生 HCU 多流限制。

没有更改环境变量、服务或底层库，没有启动新的 GPU/graph/整模型验证。所有数字均来自用户提供的两份历史 profile，不能用作新提交的性能验收。

## 多层重复的相邻事件错位

原生 DP6 和 adapter DP3 均为 bs=1，统计全部五次 target-verify，按同一 stream 内相邻事件读取原始 dur，单位 µs，表内使用中位数。不跨 stream 取“下一条”。以下只是原始数据和诊断假设，未移动、重命名或修复原始事件。

| 事件序列 | 原生显示 | adapter 显示 | 观测次数（每组） |
|---|---|---|---|
| GEMM → RMSNorm → 后续量化 | 108.96 / 7.36 / 8.64 | 4.48 / 109.44 / 7.36 | 405 |
| GEMM → dense SiLU → 后续量化 | 144.32 / 6.72 / 4.96 | 4.48 / 145.92 / 6.56 | 15 |
| FlashMLA sparse → combine → 后续 BMM | 109.28 / 12.00 / 23.52 | 4.48 / 112.32 / 12.00 | 390 |
| paged MQA logits → topk | 84.16 / 26.72 | 4.48 / 84.16 | 105 |
| SiLU+INT8 → 后续 W4A8 GEMM | 8.32 / 26.72 | 166.40 / 9.76 | 375 |

例如 FlashMLA 的长耗时从 sparse 名称挪到了 combine 名称，combine 的约 12µs 又挪到了后面的 BMM；RMSNorm 的约 7µs 挪到了其后量化。16 个 adapter rank 均出现相似模式，bs=2/3 的 FlashMLA 长耗时随 batch 增加，但仍落在 combine 标签下。原生 16 rank 的对应邻接统计没有同样的整体偏移。W4A8 调用源码有两次 GEMM，而 adapter trace 在多数 EP 层只显示一次该 GEMM 名称，原生两次均可见；所以旧 trace 的 kernel 名称计数也不能作为漏执行算子的充分证据。

这对“adapter trace 的 graph 名称/时间关联存在问题”构成很强的证据，但尚未拿到所用 vendor HIP/ROCTracer 源码或独立硬件计时证明，不能宣称已定位驱动中的具体 bug，更不能把整份 trace 强行统一平移一条来重建真实耗时。

两份 trace 自报 HIP driver/runtime 均为 60326255、ROCTracer 4.1；实际镜像 torch version.py 为 2.11.0+das.opt1.dtk2604（git a346d24b0aa21d34ebda15ba9839433496b9d887）。adapter 日志各 rank 有 `ROCTracer produced duplicate flow start`，这是关联质量的旁证，不足以单独证明 kernel duration 错位的原因。公开 ROCm/kineto 最新代码也不能当作此 vendor 二进制的源码；本轮未据公开代码改库或环境。

## 真实延迟主要位于 MoE 区间

区间定义：每个 stream 内从第一条 dispatch 开始，到该组最后一条 combine 结束，包含 MoE 计算与跨 rank 等待；同 graph 内区间取并集。不是简单累加通信 kernel dur，因此没有将重叠流重复计时。区间边界仍依赖 trace 名称，若关联错位，边缘有误差；结果用于定位大段延迟，不当作精确通信分账。

| verify 次数 | 原生总跨度 ms | adapter 总跨度 ms | 原生 MoE 区间 ms | adapter MoE 区间 ms | 原生区间外 ms | adapter 区间外 ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 70.57 | 176.30 | 15.82 | 120.15 | 54.75 | 56.14 |
| 2 | 70.20 | 177.83 | 15.66 | 121.64 | 54.54 | 56.20 |
| 3 | 70.19 | 199.12 | 15.60 | 142.91 | 54.59 | 56.20 |
| 4 | 70.18 | 159.89 | 15.60 | 103.64 | 54.58 | 56.24 |
| 5 | 70.02 | 398.75 | 15.51 | 340.68 | 54.51 | 58.07 |

这不是等负载 benchmark。adapter 的 16 DP rank 均有请求，bs 合计 33；原生只有 DP6 的逻辑请求数为 1，其他 rank 仍执行 graph 并参与 EP。全组 expert token 数、路由偏斜、上下文长度不同，不能将 MoE 差额全部归咎于调度实现或网络吞吐。

## 单 rank 停顿与全组等待

以各 rank 第一条 GPU event 为相对零点（只作近似跨 rank 对齐，不能当作已校准全局时钟），约 1185.486ms 处，DP0 有一条标为 BF16 fill 的 35.221ms event；其余 15 rank 在 1186.318–1186.673ms 附近进入长 dispatch，持续 34.862–35.313ms。该时段符合一个 rank 延迟传播成 EP 全组等待的模式，但 BF16 fill 标签可能错位，不能据此认定 fill kernel 是根因。

两节点绝对时钟约有 9.74 秒差异，报告未将绝对 ts 直接混合排序。第 5 个 verify 在全组都膨胀到约 398ms，而原生五次基本稳定在 70ms 左右；这是真实 timeline 膨胀的证据，不能仅用名称错位解释。

## 第二个已确认的调用差异与代码修正

原生 `models/deepseek_v2.py`：HCU 的主 attention alt stream 仅在显式多流开启或 DSA prefill CP 时使用；MoE 构造器在 HCU 且 `SGLANG_ROCM_USE_MULTI_STREAM=0` 时强制清空 alt_stream。原生 NextN 的 `_is_cuda=False`，默认也不创建 CUDA alt stream。

adapter 路径中框架 `_is_cuda=True` 是兼容表面，框架主模型/NextN 因此创建 alt stream，plugin attention/MoE 直接保留这个对象，触发 q/k RMSNorm、shared/routed MoE 的备用流分支。两边 recipe 都是 `SGLANG_ROCM_USE_MULTI_STREAM=0`，但运行行为不同。trace 支持这一差异：adapter DP3 target verify 的 kernel stream 为 645/647，原生 DP6 只有 513；全部 16 rank 的 target-verify 均为 adapter 两条 stream、原生一条。

本次仅对 HCU + GLM MoE DSA 做最小门控：attention 在传给 Indexer 之前按原生主模型/NextN 条件筛选 alt stream；保留主模型 DSA prefill CP、显式多流以及 NextN 显式 NPU/SBO 条件。MoE 在原有多流开关关闭时不继承 alt/routed-quant stream；NextN 不再因为 CUDA 兼容标记默认创建 stream。没有改全局设备判断，也没有将同样策略扩散至其他模型。

该修正补齐真实调度差异，未验证它是否消除 profiler 错位、长停顿或 kernel fault。此前提交的 recv-hook 接入仍保留；两者共同恢复原生主要通信/流调度条件。

## 后续调查范围

优先关注 MoE/通信流依赖、单 rank 停顿、graph 活动记录关联；暂不根据旧 profile 把 RMSNorm、FlashMLA combine 或 SiLU 换成另一个数学实现。kernel fault 的首个故障内核仍未确认。用户本轮不要求运行验证，因此在静态证据之外保留这些未验证项。

分析产物：`analyze_anomalies.py`、`anomaly-analysis.json`、`runtime-profiler-source-inventory.json`；存于验收产物的 `profile-comparison-20261106-v1/` 目录。原始 tar 的 SHA256 存于 anomaly-analysis.json，所有 rank、graph、邻接 event 样例及区间统计可追溯。
