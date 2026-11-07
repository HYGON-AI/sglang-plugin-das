# GLM5.3 CT W4A8 整理与验证交接

本文仅覆盖非 Flash GLM5.3 的 compressed-tensors W4A8、P/D、CP/LayerSplit、EAGLE/VP 与适用 graph。用户明确排除其他模型；DSV4、GLM5-next 等专用路径不纳入本次迁移或验收。共享模块只整理 GLM 链路依赖，保留其他模型原有分支。

## 固定来源与边界

- 整理起点：plugin `9faa706d8b8c1ebcec680395634fe21ed2fd0f15`，adapter `6d3abe8`。两仓均使用隔离分支 `refactor/hcu-feature-ownership`；未修改正在测试的工作树或节点。
- 用户指定的框架基线：HYGON SGLang `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`。社区接口对照：`baa9befa381511cd4d56f83980951e5a45fffc6d`。本轮未改这两个框架工作树。
- VP/LayerSplit 来源为原生镜像实际安装文件，对应 `ac17298aeae1b149a17c4ce964619c782832d6a4`，见 [VP](../plans/glm53-vp-source.json)、[LayerSplit](../plans/glm53-layersplit-source.json)。CT 分发迁移来源另见 [CT](https://github.com/maxiaobetter/sglang-plugin-das-fork/blob/feat/glm53-ct-w4a8-moe-20261107/plans/glm53-ct-w4a8-source.json)，不能将两类来源混作同一版本。
- 本轮交付是代码整理候选。实际算子对齐、模型精度与性能均未由本轮静态检查证明。旧组件证据只能按其固定输入、版本和拓扑引用，不能继承为本轮整体验收。

## 功能归属与调用链

| 功能 | 所有者与代码 | 整理结果 | 新组合运行状态 |
|---|---|---|---|
| CT W4A8 MoE | plugin `quantization/compressed_w4a8.py` → `moe/deepep_current.py` → `moe/ct_w4a8.py` | CT 分发、权重方法、INT8 通信、normal/masked core 在 plugin；旧 SlimQuant 类名仅保留兼容导入；非匹配方案回当前有效 original | 未验证 |
| W8A8 dense/attention Linear | plugin `quantization/compressed_int8.py`、`quant_ops.py`、`lightop_int8_marlin.py` | 保留当前已选公共 LightOp 路径及 BF16 舍入位置；不以其他 GEMM 的近似通过替代 | 未验证 |
| Router / Top-K | plugin `moe/deepseek_gate.py`、`routing.py`、`topk_source.py` | GLM CT 的 BF16 gate 和专家选择门控留在 plugin；不修改其他 architecture 的选择 | 未验证 |
| RMSNorm / residual | adapter `sgl_kernel/layernorm.py` | 保留既有 plain/fused RMSNorm 兼容与 FP32 residual sum；没有把既有算子的行为还原移成模型功能 | 未验证；已有独立组件证据需复核版本 |
| RoPE / DSA / Index-K | plugin `rotary.py`、`attention/current_dsa/`、`mem_cache/dsa*`，adapter 提供公共算子面 | 保留已选原生 BF16 RoPE、INT8 Index-K、排序与索引顺序；条件算子从原模块显式解析 | 未验证 |
| CP / DP token counts | plugin `cp_runtime.py`、`cp_token_counts.py`、`forward_batch.py`、`layers/dp_attention.py` | CP 门控、物理行数及 DP 缓冲区由 plugin 持有；去掉 DP 整模块 globals 注入 | 未验证 |
| LayerSplit / draft scratch | plugin `mem_cache/dsa_cache_layer_split.py`、`dsa_layer_split_integration.py` | 配置器和模型别名接线、source 生命周期、单层 draft owner、独立 Index-K scratch 保留；紧凑广播与同步预取默认 1，显式 0 可关闭 | 未验证 |
| VP | plugin `speculative_vp/`、`prefill_output.py` | CLI → 子进程配置 → 拓扑校验 → lm_head 初始化 → logits → draft 消费全部接线；保留原生核心 | 未验证 |
| P/D | plugin `disagg/` | 描述符、CP rank 映射、传输、完成/释放协议归 plugin；scheduler/worker 不再重建函数 globals；transfer 导入使用 plugin 自有路径 | 未验证 |
| EAGLE / graph | plugin `distributed_backend.py`、`deepep_graph.py`、`graph_*`、`prefill_graph_*` | verify 广播限定 attention TP/CP；VP draft 与原 graph runner 共用函数；保留 DeepEP 捕获/重放模式 | 未验证 |
| 平台 / ABI / 启动配方 | adapter 公共兼容层；`examples/glm5.3-w4a8/pd/` 组合两仓 | 使用配套镜像算子库；CT native、SlimQuant 关闭；prefill LightOp Marlin=0/HIPC=1；不额外指定 dsa-topk-backend torch | 安装路径与 ABI 待新进程核验 |

以上表格表示代码归属及接线审阅，不能解释为所有算子已对齐。`examples/glm5.3-w4a8/launch.sh` 是早期单机 Triton 探针，本次交接使用 `pd/prefill.sh`、`pd/decode.sh`，不以早期探针结果验收 P/D。

## 测试方执行顺序

1. 固定两仓最终提交和实际挂载路径，核对安装包路径、版本、文件哈希、ABI、镜像 digest；保留当前工作区修改。禁止将两个任意最新 HEAD 混用。
2. 在新进程确认 plugin 已加载、全部相关 Hook 实际应用、旧别名已指向正确类。记录 CT 方法/core、LayerSplit 池/configurator、VP worker/logits helper 的真实类和调用链。日志中 Hook 失败不能当作启动成功忽略。
3. 复用既有原生 32-rank P/D profile 和实际输入归档，先逐算子比较新代码。CT 必须覆盖权重 nibble/scale、源端 INT8 量化、两个 GEMM、激活量化、scatter/gather/combine；同时复核 W8A8、router、norm、RoPE、DSA/cache 的真实调用。对照表逐项填“已对齐 / 存在差异 / 未验证”。
4. 验证 VP 的所有组和参与 rank，包含 idle/非 idle、top-1、真实共享权重与节点内分组；验证 LayerSplit owner/remote、共享 data_ptr、失效和流等待、P/D 描述符映射。PP2 prefill 与 PP1 draft scratch 适用范围分别记录。
5. 对适用的 decode/MTP/graph 更换输入与有效 token 数重放，检查结果和 collective 顺序；捕获成功不算通过。不为补覆盖而擅自切换数学策略或降为 W4A16。
6. 组件对齐项全部关闭后再按原先不变阈值跑整模型精度矩阵。通过后整理 PR；未通过则在当前候选上做对应最小修复，更新证据并重新 lint。性能优化排在精度之后。

## 本轮本地检查

只运行 CI 等价 lint、Python AST、结构/快照检查和 `git diff --check`，没有新增单测或运行整模型。P/D scheduler 与 transfer worker 的函数体，在消除命名空间限定后与整理前 AST 一致；CT normal HIPC/masked 分支的源码对照见来源记录。具体最终检查结果记录在开发日志。

其他模型的全仓盘点仍是独立历史台账，本轮不清空、也不把其 pending 项伪装成 GLM 运行失败。
