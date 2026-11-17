# DeepSeek V4.1 wo_a 原生迁移与性能复测

日期：2026-11-16。范围：plugin 的 Prefill `wo_a` 分组 INT8 投影。adapter、D 服务和 router 没有修改或重启。

## 结论

已将截图红框中的逐 group Torch 回退替换为 plugin 自有原生路径：融合量化 → hipBLASLt strided-batched INT8 GEMM/INT32 累加 → 融合缩放至 BF16。当前固定 gfx936/BF16/G8/N1024/K4096 形状命中；G1 lm_head、CPU 和其他形状保留原实现。

新 TP4 的第 19 层第三个记录 step 中，wo_a 从 64 个 GPU kernel 变为 3 个；原 DAS 为 3 个。五步所有 8 rank 都确认 200 次 native project、200 次量化和 200 次缩放。算子数值 14 项零容差通过；整模型快速检查 19/20。此为新版本快测，不替代之前完整 HumanEval/GSM8K 得分。

新 TP4 GPU step 包络中位数 759.428ms，相对上一版 863.002ms 变化 -12.00%。相对历史 DAS 506.012ms 仍相差 +50.08%，不宣称整模型性能已完全对齐。prof 包络包含跨 stream 等待，不是 kernel busy time，也不作为正式吞吐成绩。

## 同位置 trace 对照

选择同一 rank TP4、MQALayer19、第三个记录 EXTEND；通过 CPU wrapper 内的 HIP runtime correlation 对应 GPU kernel。DAS 原 trace 的完整输入 IDs 未保留，且 Python/Torch/HIP 版本不同，属于历史路径和时间参考。

| 指标 | 历史 DAS | 上一版 plugin | 新 plugin |
|---|---:|---:|---:|
| 五步 wo_a wrapper 数 | 200 | 200 | 200 |
| wrapper CPU 均值(ms) | 0.340 | 4.229 | 0.324 |
| 每步 wo_a CPU 总计均值(ms) | 13.601 | 169.171 | 12.970 |
| 代表调用 GPU kernel 数 | 3 | 64 | 3 |
| 代表调用 GPU 时间并集(ms) | 1.780 | 11.402 | 7.082 |
| 代表调用 aten::pad 数 | 0 | 8 | 0 |
| 代表调用 aten::clone 数 | 0 | 16 | 0 |
| GPU step 包络中位数(ms) | 506.012 | 863.002 | 759.428 |

CPU inclusive 时间不与嵌套节点累加，也不能直接当作端到端可回收时长。完整 kernel 名称、HIP runtime 与 CPU op 表保存在 `same-operator-comparison01.json`。

本轮并未把 wo_a 的 GPU 时间追平。代表调用新 GEMM 为 5.645ms，原 DAS 为 0.842ms；新量化为 0.996ms，DAS 为 0.449ms；新缩放为 0.441ms，DAS 为 0.488ms。新 GEMM 实际选中 MT64x64x64，DAS 为 MT256x256x32。当前实现设 8MiB workspace 上限，并只向 `hipblasLtMatmulAlgoGetHeuristic` 请求一个候选；不同库版本、布局和启发式选择的影响尚未由独立同 shape 算法比较分离，不能直接断言 tile 大小就是全部原因。kernel 数相同不等于吞吐相同。

## 无 profiler 性能复测

P/D 拆分，P 为 a12r4n16、D 为 a14r4n13，容器 `sglang_deepseek_V4.1_LC_adapter`。P 的完整启动脚本与上一版逐字节相同，environment_delta={}，TP/EP/CP=8，seed、显存和上下文预算不变。使用原 `bench.sh` 对应 `prefill_throughput.py`，固定 32768 输入 IDs、输出 1、并发 128、prefix_ratio=0；三轮共 384 请求，另 3 次单请求。未重跑整套输入长度矩阵。

| 轮次 | 输入吞吐(token/s) | 平均 TTFT(s) | P99 TTFT(s) | 请求错误 |
|---|---:|---:|---:|---:|
| 1 | 42127.032 | 51.858 | 98.922 | 0 |
| 2 | 42186.200 | 51.800 | 98.865 | 0 |
| 3 | 42177.942 | 51.777 | 98.896 | 0 |

输入吞吐中位数 **42177.942 token/s**，单请求 TTFT 中位数 **0.811s**。计算窗口遵循原 benchmark，prompt 构造位于计时外；外层 elapsed 另含构造开销，不能直接拿总 token 除外层 elapsed 反推该指标。

相对 P06 的同三组完整输入和同一 controller，吞吐中位数由 25855.666 变为 42177.942 token/s，变化 +63.13%。P06 尚有 FP8 paged Torch 回退，P07 没有新的无 profiler 成绩，因此这个收益合计包含 FP8 paged 与本轮 wo_a 两阶段，不能全部归因于 wo_a；这也不是交替多轮 A/B 或新的 DAS 无 profiler 性能基线。

## 全部 rank 与 attention 保持

| rank | GPU step 中位数(ms) | native wo_a 调用 | 量化 / 缩放调用 |
|---|---:|---:|---:|
| TP0 | 758.489 | 200 | 200 / 200 |
| TP1 | 759.059 | 200 | 200 / 200 |
| TP2 | 759.235 | 200 | 200 / 200 |
| TP3 | 759.101 | 200 | 200 / 200 |
| TP4 | 759.428 | 200 | 200 / 200 |
| TP5 | 759.332 | 200 | 200 / 200 |
| TP6 | 759.235 | 200 | 200 / 200 |
| TP7 | 759.367 | 200 | 200 / 200 |

每个 rank 均有 5 个 `step[EXTEND bs=1 toks=32768]`，BF16 sparse105、FP8 paged95，Torch paged0。当前 FlashMLA 私有兼容 SO 仍为 d0d4018a…，原全局 SO 不变；它只获得当前 gfx936/Torch/DTK 的实验兼容资格，不能称供应方源码重编或其他平台资格。lm_head G1 仍走原回退，所以不能要求全 trace 的 `_int_mm` 为零。

## 修改、数值与边界

- `hcu_sglang/layers/quantization/channelwise_int8.py`：新增受限 native dispatch，原权重加载、scale 检查和 fallback 保持。
- `hcu_sglang/layers/quantization/dsv41_int8_native.py`：当前 HIP SDK 编译/缓存 C ABI 库、按 device/shape 缓存 plan，使用当前 stream，每次独立分配输出和 workspace。
- `hcu_sglang/jit_overlay/csrc/dsv41_int8_projection.cu`：融合 BF16→INT8 row/group 量化、直接 strided batch GEMM、融合 FP32 scale→BF16 输出，不做八次 Python GEMM/转置拷贝/pad。
- `tests/runtime/channelwise_int8_native.py`：实际 gfx936 上 random M1/2/15/16/31/32/4096、zero、half-away ties、非连续输入、3 次 Graph x/w/scale 更新和同 plan 双 stream，共 14 案例均 max_abs_error=0、rtol=atol=0。

没有借用旧 Python/Torch 二进制、动态 import DAS、安装或修改全局 sgl_kernel/Torch、添加 adapter 功能，也未更改 attention 的既有修复。

原失败证据保留：第一次 build 成功但实际 512-thread launch 超出默认限制，候选仅补显式 launch bound；随后一个 BF16 值差 1ULP，独立诊断证明 q/INT32 完全相同、8 个 FP32 scale 差 1ULP。将除 127 按原 Torch 的 FP32 reciprocal 乘法顺序实现后，原零容差才通过，没有放宽阈值。编译成功不能代替数值通过。测试退出时原 Torch.library 清理警告也保留，未为此改生产逻辑。

新 quick check 错误样本：`[{"index": 12, "target": "13", "answer": "12", "correct": false}]`。20 题得分与上一版同为19/20，原始完整请求/响应/评分均保留；未复测完整数据集，不把旧 HumanEval/GSM8K 得分继承为本轮通过。

## 文件与复现

- 本地新 trace：`D:/git/dcu_sglang/sglang_plugin/doc/prof/20261116-woa-native-prefill32k/1794823528.1714485-TP-4-EP-4.trace.json.gz`。
- 新 trace SHA256：`866469038411c3d51905d1936a0ca5c4f1eb0ebe75281e9b33fca864d71bd1c7`。
- 全部 8 rank：`/home/plugin_adapt/redevelop-20261114-node-switch/prefill32k-megamoe-compare01/woa-model08-profile01/traces`。
- native SO SHA256：`fb8e5d71b6c36a3d33d62d6b6aadf253e2c05a284d3e9aa068b6a1a2d389b169`；源码/输入/环境、native 映射与完整配置均在本次 evidence ZIP。
- 模型/请求/prof/性能证据：`woa-model08-evidence01.zip`，SHA256 `46233af02dea4f4272d2a5d62de755218a84aa2ce5d894c66024c817cfe0076e`。
- 独立单位数值证据：`woa-native-units-evidence01.zip`，SHA256 `d45bfee33814fdf75e5570b73a7d7c0af0df4bca9d49f901c0cb5bb771a550ce`。
- 本地 `local-proof01.json` 独立重算20题评分、全部成员/父包/源码/SO、同三轮完整 IDs/TTFT、全 TP4 JSON 5 步和原生调用。

采集参数保持 start_step2 / num_steps5 / activities CPU+GPU / with_stack / record_shapes。当前路径的 native 库使用当前 HIP26.04 与 hipBLASLt 编译，部署需有对应 SDK，首次新 shape 需在 Graph capture 前预热。

回退只需恢复 evidence 中 `woa-native-change01/channelwise-before.py` 的 dispatch 文件并按实际 owner 归属重启 P；不要热替换已映射 SO，也不要关闭已通过的 FlashMLA 路径。D 和 adapter 无需改动。

剩余范围与具体下一修改点：先在当前库和同一 G8/M4096/N1024/K4096 布局下，有界比较 hipBLASLt 可用算法与 workspace，数值/Graph 通过后再固定有实际收益的算法选择；不要照抄历史算法编号或只为相同 kernel 名称换库。这优先处理已证实的 5.645ms GEMM，而不是盲改 MoE/通信/模型预算。整步仍存在历史 DAS 差距；本次不因路径 kernel 数对齐就宣称模型性能完全相同，也不从 prof 数据填造正式 DAS 吞吐。
