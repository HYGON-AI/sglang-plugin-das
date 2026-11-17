# DeepSeek V4.1 W4A8：Sparse Prefill cache-builder 同步消除（P19）

日期：2026-11-17。状态：本次截图指定调用域已完成源码修复、独立 HCU 数值/Graph、整模型同组回归和两组全 8 rank prof 验证。

## 1. 问题与最小修复

截图调用链为 `_build_sparse_prefill_chunk_cache → aten::item → aten::_local_scalar_dense → hipStreamSynchronize`。P18 在 HCU CP 路径使用 `int(query_lens.sum())` 做 Python 容量断言，将原生 producer 产生的 GPU 长度回读到 CPU，等待当前流。它检查容量，不参与 attention 数学。

P19 在原生 producer 之前，利用已有 CPU `extend_seq_lens_cpu`、当前 CP rank/size 和既有 `interleave_rows_per_request` 计算所需真实 query 行数，检查 `num_qo_tokens` 容量。原生 producer 仍按 `query_positions.shape[0]` 处理物理 padding，metadata/cache 数学、FlashMLA/LightOp、跨流完成边界均不改。非 HCU 分支保留原断言。容量不足在 native 调用前拒绝。

唯一运行时变更位于 `hcu_sglang/models/deepseek/current_v4/hcu_backend/backend.py` 的该方法，两处具名编辑逆变换后与 P18 原字节一致，CRLF 保持。新增持久化回归 `tests/runtime/sparse_prefill_query_capacity.py`。未修改 adapter、D、router、平台检测或服务默认参数。

- P18 backend：`f4fdc66d0a51dc81dde9634e297f670ac8c335b37e50cb97e86a68eeee7928b6`。
- P19 backend：`041c17b257e70d90ae7acac3c58be1b4a529614f68fbdcf110b98e259cc57f8a`。
- P19 运行策略：`87b45101173c2a76a80b267dc910fc4cc27fb902becb0f4dbc28cf3fb975b8a4`，58 份来源。

## 2. 实际 prof 验证

固定 128 个完整 32768-token 输入、每请求输出 1 token、客户端并发 128；TP8/CP8/EP8、LayerSplit/bounded/DSPARK/MegaMoE 保持。CPU+GPU，`start_step=2`、`num_steps=5`、stack/shapes 开启，全部 8 个 rank。burst 在该组 128 请求前 arm；另一个独立 steady 组在首 16 个成功完成后 arm，不 flush cache。没有挑 rank 或替换慢值。

两窗口合计 **8 rank × 5 step × 2 = 80 次**实际目标调用，`aten::sum/item/_local_scalar_dense` 及流/设备同步 API 均为 **0**。原 P18 steady 每 rank 五次目标调用有五次标量回读和五次流同步。

| 调用域 | 每 rank 五步 scalar / sync | rank4 CPU 中位/ms | 全 rank 目标 GPU 域最大内部未覆盖间隙/ms |
| --- | ---: | ---: | ---: |
| P18 steady | 5 / 5 | 12.760 | 2.365120 |
| P19 burst | 0 / 0 | 1.180 | 0.062721 |
| P19 steady | 0 / 0 | 1.138 | 0.000001 |

GPU 域由同一 trace 的 runtime correlation 关联该调用提交的 metadata 工作，再对全部 stream 的真实 GPU 事件取并集。CPU inclusive 时间不是 GPU idle，重叠 kernel 不累加。上述结果确认截图位置的同步链和毫秒级同步空泡已经消除，微小 launch 间隙仍按原值保留。

原始 trace：

- burst rank4：`doc/prof/20261117-prefill-opt19/1794882329.0064635-TP-4-EP-4.trace.json.gz`。
- steady rank4：`doc/prof/20261117-prefill-opt19-steady/1794882826.629038-TP-4-EP-4.trace.json.gz`。
- 每组其余七个 rank 位于相应 `all-rank-traces/`；完整 16 份文件均按原 size/SHA、gzip CRC 和完整 JSON 独立核验。
- 目标统计：`p19-sparse-scope-burst01.json`、`p19-sparse-scope-steady01.json`；整步/后半层统计：两目录的 `all-ranks-analysis19.json`。

## 3. 数值、模型与无 profiler 性能

- CPU：140 个 CP/容量/空 rank 等 case，加三个 fallback；明确区分 native producer/consumer peers。
- 实际 gfx936 HCU：70 个 CP/空 rank/padding case，一个真实 HCU 非 CP fallback，三个修改 query positions 的 Graph replay。执行实际原生 producer 和完整 cache consumer；定义有效索引、长度及其他 metadata `rtol=0, atol=0`。未定义且 consumer 不扫描的 index 容量尾部不作为数学输出。非 HCU fallback 仅 CPU 覆盖。
- 模型：同组 GSM8K 快速回归 **20/20**，固定请求/评分器，与原回归输入逐字节一致。这不是全新完整数据集得分。
- 无 profiler：三轮各 128 请求，合计 **384 请求、0 错误**；输入与 P09 原对照 SHA 一致。另三条单请求，均 32768→1。

| 阶段 | 无 profiler 输入吞吐中位/token·s⁻¹ | 单请求 TTFT 中位/ms |
| --- | ---: | ---: |
| P09 | 52,422.368 | 657.535 |
| P18 | 55,114.928 | 653.386 |
| P19 | 55,206.683 | 644.027 |

P19 三轮吞吐：`55102.360901 / 55206.683120 / 55214.737987`。相对 P18 **+0.166%**，相对 P09 累计 **+5.311%**。相邻两版增幅较小，不据此宣称超出噪声的额外吞吐收益；本次直接证据是目标同步消除。

两窗口全部 rank 的原生路径维持：每五步 BF16 sparse 105 次、FP8 paged 95 次、新 dense MQA out 25 次、旧 dense 数学入口 0 次；后 19 层每窗口/rank 95 个 scope，继承 P18 的数学和通信路径。

## 4. 范围与保留问题

本次完成截图指定 cache-builder 调用域。**不声称整模型所有 step 间空泡为零**：两组 rank4 的整步真实未覆盖间隙分别为 `[144.932279296875, 102.42041162109375, 44.71538671875, 0.0]` 和 `[30.28546484375, 11.9309970703125, 0.32509375, 16.76315234375]` ms；全 rank 最大分别为 145.287 和 30.515 ms。它们在调用域之外，先前准入/CPU Gloo 到达偏斜等协议等待仍保留，不能为美化时间线删除。

原 DAS trace 仅为历史路径参考，当前环境不同，未新增严格受控 DAS A/B。没有重跑全量 HumanEval/GSM8K/MATH500，也未更改五个百分点或其他精度门槛。正式性能不取 profiler 扰动数据。

前四次 native 资格的 fixture ABI/未定义容量/混合平台/Graph 参数错误均保留原失败和原始日志，未当作模型数值或性能失败、未重写成功。最终通过 test05；生产候选字节始终相同。

## 5. 源码、产物与回退

节点源码：`/home/sglang/sglang-plugin-das`。CPU wheel 为实验 `0.3.2.dev77`，SHA `18042f8d6c54fe5f46145b6ac2599498248b2350428867a054514720f7a25d15`；只验证构建/RECORD，运行服务使用实际源码，未用 wheel 覆盖环境。

独立资格包 `p19-sparse-query-qualification01.zip`：SHA `60e4cd6fdafa2b341627d19100a687840c1df6ba8c2e4d2968f3d1f342b9ef11`，120 成员。最终模型包 `prefill-opt-model19-evidence01.zip` 和本地 `independent-proof01.json` 另绑定完整来源、报告、20 响应重评、384 输入、两窗 16 raw、实际八 scheduler native maps 和所有前置失败。

回退仅恢复 P19 保存的 backend 原字节并按 owned PID/PGID/start_ticks 重启 P；不要回退到已失败的异步 tail 实验。当前 P/D/router 配置与 seed55965328、mem.85/KV262144/running32、chunk32768 均未变；没有 git stage/commit/push。
