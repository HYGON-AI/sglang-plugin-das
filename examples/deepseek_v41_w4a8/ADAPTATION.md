# DeepSeek V4.1 W4A8 的 HCU Adapter / Plugin 适配与验证

> 2026-11-15范围更新：按用户要求只保留HumanEval和GSM8K两轮，取消MATH500。当前第二轮GSM8K继续运行；历史源码锁定清单中的MATH pending为取消前快照，以本次有效范围为准。

更新日期：2026-11-15。本文说明本次源码归属、关键修复、P/D 部署条件、验证结果和交付限制。源码整理目标为节点容器内的两个仓库；本地源码副本不参与本次合入。

**当前状态：整模型已完成加载与 P/D 正确答案验证，HumanEval 和 GSM8K 第一轮已有完整得分；完整两轮精度、原可运行 DAS 新基线对照及性能报告尚未完成。当前不是完整精度验收通过版本。**

## 1. 版本、模型与执行环境

| 对象 | 本次固定输入 |
|---|---|
| Adapter 源码 | 节点 `/home/sglang/adapter-das`，HEAD `766879bee10a91aaac132f003c51a7a7c8cfc4cd` |
| Plugin 源码 | 节点 `/home/sglang/sglang-plugin-das`，HEAD `9258ceff25c9735827aeadb2c2049f365a2129d5` 加当前未提交的模型修复 |
| 分支 | 两仓均为 `deepseek_v4.1_w4a8_cp8ep8` |
| 迁移参照 | `/home/sglang/sglang-das`，HEAD `d9e87ae59573c7374e5de6e90c6b2c7054a4d9fc` |
| 官方宿主 | SGLang `0.5.22`；plugin 包版本 `0.3.2.dev77` |
| 模型 | `DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8-wo_a-markov_head-lm_head` |
| 容器 | `sglang_deepseek_V4.1_LC_adapter`，Python 3.12 / 当前配套 HCU DTK 环境 |
| P 节点 | `a12r4n16`，IB `12.1.12.76`，8卡，TP8/CP8/EP8 |
| D 节点 | `a14r4n13`，IB `12.1.14.73`，8卡，TP8/attention-DP8/EP8 |
| 推测解码 | DSPARK block size 5，verify window 6；保留实际 checkpoint 配置 |
| 网络 | 每节点四个 SHCA，Mooncake P/D 传输；`/home` 共享，`/opt` 为节点本地 |

本轮36个选定文件中，21个属于 plugin 运行时代码，其余为测试、目标合同和开发记录。逐文件 SHA256、源码版本及依赖 wheel 哈希记录在本目录 `deepseek-v4.1-w4a8-source-manifest.json`，节点部署示例中为 `source-lock.json`。HEAD 本身不能代表包含未提交修改的完整候选。

本次仅整理节点源码，不提交、推送或重置分支，不改官方 SGLang 工作树，不把临时评测器、进程控制器或失败诊断注入生产源码。

## 2. Adapter 与 Plugin 的职责

| 范围 | 归属 | 本次处理 |
|---|---|---|
| 官方已有的启动、ABI、影子算子和平台兼容 | Adapter | 使用现有已固定 adapter；没有为了支持模型新增能力而修改全局平台判断或伪造 vendor 符号 |
| CT channel INT4 + token INT8 专家路径及模型投影 | Plugin | 保留真实 activation 配置，接入本模型的 provider、loader 和 INT8 消费链 |
| CP 状态描述、P/D 传输命名空间、DSPARK 与模型专项执行 | Plugin | 使用 plugin 自有实现和正式注册入口，修复 producer/consumer 不一致 |
| HCU 专项 FlashMLA 回退、Engram 指针和 logits gather | Plugin | 精确门控；不支持的原生路径保留限制，不强塞进 adapter |
| SHCA / Mooncake 二进制与 host ABI | 外部构建和部署依赖 | 头文件、运行库、wheel 配套；不作为 adapter 的 Python 修复 |
| 评测中断恢复、并发和证据封存 | 独立评测客户端 | 保留原始成功结果和负日志；不改变模型数学或实际生成参数 |

加载顺序为：正常 Python 启动 → adapter bootstrap → 官方 SGLang → `SGLANG_PLUGINS=hcu` → plugin 注册及 apply → target/draft 加载与 Graph → P/D 服务。Plugin 使用 `hcu = hcu_sglang:register` 入口；存在前序包装时，非匹配路径委托当前有效的 `original`。

## 3. 已整理到节点源码的关键修复

以下文件已在当前节点源码中，且模型22按这些实际字节启动。各项算子/CPU/Graph 结果不能代替整套数据集验收。

| 领域 | 主要源码 | 处理内容 |
|---|---|---|
| CT W4A8 接入 | `layers/quantization/compressed_w4a8.py`、`slimquant_w4a8_marlin.py`、量化注册入口 | 对本模型保留 INT8/token 动态激活，避免被通用 W4A16 回退覆盖；兼容真实专家投影别名 |
| 权重 loader | 上述 W4A8 方法 | 在参数所属 loader 边界匹配 checkpoint 方向；EP 所有权、TP 分片继续委托当前官方 loader。MegaMoE 与 DeepEP packed 布局分别处理 |
| 特殊 INT8 投影 | `layers/quantization/channelwise_int8.py`、模型 `attention.py` / `dspark.py` | 接通 wo_a、LM head、Markov 等 INT8 权重及 FP32 channel scales；保持原 TP/填充词表 loader 和 Markov 分片 scale 偏移 |
| HCU 模型协议 | `models/deepseek/current_v4/execution/core.py`、`speculative_dspark_worker.py` | 在 HCU 范围选择现有受支持的模型执行协议，避免进入不匹配的 CUDA replay/boundary 路径；不全局翻转平台真值 |
| MegaMoE 准入 | `server_args_hooks.py` 等 | 只对自有 HCU standalone prefill runtime 处理能力门；不对不支持的 overlap 组合放行 |
| CP P/D 命名空间 | `disagg/prefill.py`、`disagg/decode.py` | 两侧均使用 plugin-owned transfer 配置，不创建不存在的官方 stub 或 import 别名 |
| CP 请求状态 | `mem_cache/cp_cache_layer_split/transfer.py` | 为 DSPARK hidden rows、valid-from 等请求状态提供稳定 descriptor，保留匹配及 stride 检查 |
| CP materialize | `cp_runtime.py` | non-v2 分支绑定 plugin 已有 HCU helper，保持 interleave/zigzag 与 collective 顺序 |
| FlashMLA / sparse prefill | `layers/attention/debug_flash_mla_adapter.py`、两个模型 backend | 显式 Torch 功能回退；无效 KV 的 K/V 操作数在数学运算前屏蔽，保留有效位置的 NaN，而非全局替换 NaN |
| MHC 编译 ABI | `kernels/mhc.py` | HCU lazy 编译使用已验证的 code-object-version=5，数学与全局编译器配置不变 |
| Engram host 地址 | `layers/engram.py` | 对已注册 host 指针调用实际已加载 HIP runtime 查询设备地址；失败严格拒绝，非 HCU 委托原路径 |
| DeepEP combine overlap | `batch_overlap.py` | 对当前不提供 combine 信号的 INT8/FP8 或不匹配布局关闭该段 overlap；保留 dispatch/shared overlap 及有效原路径 |
| Logits gather | `layers/logits_gather.py`、主注册入口 | HCU 在 native multimem `_build` 边界返回 None，使用官方已有 all-gather 回退；不在不支持的对称内存路径上强行分配 |

### 3.1 数值与状态约束

- checkpoint 的专家 INT4、token INT8、dense/shared/embedding/KV 等混合精度分别处理，不能把整个模型统一解释成同一量化格式。
- INT8 投影保留原量化 floor、舍入及 scale 顺序；不能忽略权重 scale 或用 BF16 权重冒充 W4A8。
- CP descriptor 的身份来自真实 state type 和 buffer index，不用 header/config 代替实际状态生产者。
- invalid KV masking 只隔离无效槽位；有效 KV 中的 NaN 仍应暴露。
- host 地址不默认等同设备地址；CUDA facade 不等于 HCU UVA 已验证。
- Graph 必须检查改变输入和 buffer 状态后的 replay；仅命中 Python Hook 不能证明 replay 正确。
- 不从参照 `sglang-das` 动态导入运行实现；它仅用于源码和数学对照。

### 3.2 通信回退与已知故障

本轮旧 custom all-reduce 路径在独立对照中出现错误，模型22的 P/D 均显式使用官方 `--disable-custom-all-reduce`，并保留 `HCU_DISABLE_OPS=aiter_ar`。实际服务日志记录每节点16条 NCCL all-reduce 选择、0条 AiterCustomAllReduce。这里的日志选择不等于整模型数值已全部验收，但不能省略这项当前配方约束。

当前 native FlashMLA 包在本环境存在 code-object 兼容问题。现有 Torch 回退已经用于模型验证；不修改原 vendor ELF 头，不宣称原生 kernel 已修复，也不由此推断与 DAS 的性能一致。

## 4. 外部 Provider 与 Mooncake ABI

| 依赖产物 | 本轮 wheel SHA256 |
|---|---|
| MegaMoE | `ab15547a36160117e7a74d7d2b202b713f325c183b649c82c90d285361c56adc` |
| DeepGEMM W4A8 | `2a2ec3ef4bfdc49e172596c4d2bb3b479f4071fc1c21958b23c8bbd826fb6ecd` |
| Mooncake SHCA | `4390f26651d435e370418b4dbc1aa5307535a56622fdad503c7ac8cda74c4e1c` |

本次 Mooncake 按官方来源 `7be21800f0c6b04853009e0b8b3e58dd70c4f6c8` 构建，1261个来源文件未改。实际 host/provider 的 `ibv_context` 大小/async_fd 偏移为376/316，而原容器头文件为328/268；仅替换 Python 路径或一个动态库不能修正这个 ABI。

本轮使用配套 host SDK 头文件、`libibverbs.so.1`、`libshca-rdmav34.so` 和完整 wheel。实际部署保留原 Mooncake package/dist-info 备份，仅替换精确入口链接；这些部署动作不属于两仓运行代码。新环境应在隔离环境内安装完整配套 wheel，核对实际模块、engine、动态库来源，不能照抄某次全局链接操作。

当前节点本地依赖目录：

```text
/opt/dsv41-redevelop-20261114-switch/providers01/megamoe-build01/site
/opt/dsv41-redevelop-20261114-switch/providers01/deepgemm-w4a8-build01/site
/opt/dsv41-redevelop-20261114-switch/mooncake-shca22-runtime01
```

两个节点均须独立具备这些文件；共享 `/home` 不代表 `/opt` 已同步。`LD_LIBRARY_PATH` 只决定动态库搜索，不能代替 Python Mooncake package 的实际安装/激活。源码清单中的外部 wheel 哈希不代表这些二进制已加入 Git。

## 5. 整理后的部署示例

节点 plugin 仓库中的目录为：

```text
examples/deepseek_v41_w4a8/
  README.md
  env.sh
  prefill.sh
  decode.sh
  router.sh
  source-lock.json
  ADAPTATION.md
```

示例将当前实际 p1/d1 的网络地址、调优目录、专家位置改为显式输入；量化、预算、Graph、DSPARK 和原配方其余参数保持。它们不会自动停止已有服务或续发本次已消费任务。

### 5.1 环境输入

在两个节点各自的容器中设置：

```bash
export ADAPTER_ROOT=/home/sglang/adapter-das
export PLUGIN_ROOT=/home/sglang/sglang-plugin-das
export PROVIDER_PYTHONPATH=/opt/dsv41-redevelop-20261114-switch/providers01/megamoe-build01/site:/opt/dsv41-redevelop-20261114-switch/providers01/deepgemm-w4a8-build01/site
export MOONCAKE_ROOT=/opt/dsv41-redevelop-20261114-switch/mooncake-shca22-runtime01
export BLAS_TUNING_ROOT=/home/run/PD/test/blas_asm_tuning_20261111/packages/gemm_asm_gfx936_80cu_20261111
export PREFILL_HOST=12.1.12.76
export DECODE_HOST=12.1.14.73
export PREFILL_EXPERT_LOCATION=/home/run/test/weiguan_test/launch_PD/pc093_tasks_20261101/eplb/static_load.pt
export DECODE_EXPERT_LOCATION=/home/run/PD/EP8_1/static_eplb_20261015/decode/static_load.pt
export MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017-Channel-INT4-w4a8-wo_a-markov_head-lm_head
```

当前 adapter/plugin 的依赖已由容器配置。构建或安装源码产物时使用 `--no-deps --no-build-isolation`，避免 pip 自动替换当前 DTK Torch/Triton 或拉取不适用的 CUDA wheel。分支已有安装前提、provider 与模型数据需要逐项核对，不能仅凭包版本相同判定环境相同。

### 5.2 启动 P、D 与路由器

P 节点在独占资源且旧自有服务已按归属停止后：

```bash
bash "$PLUGIN_ROOT/examples/deepseek_v41_w4a8/prefill.sh" 30000 "$MODEL_PATH"
```

D 节点：

```bash
bash "$PLUGIN_ROOT/examples/deepseek_v41_w4a8/decode.sh" 30000 "$MODEL_PATH"
```

主要参数：

| 项目 | P | D |
|---|---|---|
| 拓扑 | TP8 / CP8 / EP8，interleave，CP cache layer split | TP8 / attention-DP8 / EP8，DP LM head |
| 专家 | MegaMoE，normal；draft DeepEP/DeepGEMM | DeepEP low-latency / DeepGEMM |
| 静态显存比例 | 0.85 | 0.86 |
| 最大运行请求 | 32 | 64 |
| Prefill budget | chunk32768 / max-prefill32768 | 按原 decode 配方 |
| Graph | prefill disabled | decode max batch64 |
| 推测 | DSPARK block5 | DSPARK block5 |
| 传输 | Mooncake，四SHCA，bootstrap8998 | Mooncake，四SHCA |
| All-reduce | 官方 custom AR disable | 官方 custom AR disable |

两端完成各自 target/draft 加载与 ready，普通健康检查正常后，在 P 节点运行：

```bash
export ROUTER_PYTHON=/home/plugin_adapt/redevelop-20261113/eval-client01/venv/bin/python
bash "$PLUGIN_ROOT/examples/deepseek_v41_w4a8/router.sh"
```

路由器版本为0.3.2；API `http://12.1.12.76:30003/v1`，metrics29003；显式 HTTP request timeout120000秒和关闭自动重试。服务内部 KV transfer 等待超时与该值是不同层次的参数，不能由 HTTP 超时增大推定 KV 故障已解决。

## 6. 验证与精度结果

### 6.1 已有验证及其边界

| 层级 | 本轮已有证据 | 边界 |
|---|---|---|
| 源码与 CPU | CT loader/投影、CP descriptor/materialize、overlap、Engram及 logits 的 producer/consumer 合同 | 框架或设备替身必须单列，不当作真实硬件结论 |
| 算子与 Graph | INT8投影、FlashMLA回退、masked KV、collective/all-gather 等独立真实 GPU 和变输入 Graph 对照 | 不代替完整模型精度；旧销毁超时等负结果保留 |
| 当前模型 | P/D各16次 target/draft load end、8个Mooncake引擎；P首次token有限且非平坦；PD算术完整结果42 | ready或单个答案不代表数据集全部通过 |
| 数据集 | HumanEval和GSM第一轮完整逐题 prediction/review/report 本地复核 | 剩余轮次及新DAS对照尚未完成 |
| 本次示例 | Bash语法、完整P/D argv与关键环境同模型22原配方的CPU干运行一致 | 使用替身可执行文件/库，不再次启动模型，不宣称新增硬件验收 |

### 6.2 已完成轮次

| 数据集 | 轮次 | 正确 / 总题数 | 精度 | 状态 |
|---|---|---:|---:|---|
| HumanEval | 1 | 158 / 164 | **96.34%** | 完整结束并本地独立复核 |
| GSM8K | 1 | 1285 / 1319 | **97.42%** | 完整结束并本地独立复核 |
| GSM8K | 2 | — | — | 原336条成功保留，第二轮按缺失题续测 |
| MATH500 | 1 | — | — | 已按用户要求取消，不执行 |
| MATH500 | 2 | — | — | 已按用户要求取消，不执行 |

GSM第一轮1319条预测与review的题目ID、响应、二值评分逐项一致，无重复ID。3题达到输出token上限，仍计入总题数并按错误计分。原中断前273条字节前缀及行数据未改，失败样本119在具名恢复中完成。

验收采用 HumanEval164一次、GSM8K1319两轮，每侧总2802题；MATH500已按用户要求取消，不作为本次验收条件；候选与原可运行 DAS 新基线同条件比较，相对精度下降不超过5%。不要求逐题输出或得分完全相等。

公式为：`relative_drop = (DAS_score - plugin_score) / DAS_score`。GSM8K保留两轮各自得分及固定轮次合计，不追加挑选更好分数的轮次。新DAS基线尚未实跑，因此本文不签署“完整精度已对齐”。

### 6.3 第二轮中断与恢复

2026-11-15北京时间10:58:24，GSM第二轮sample319出现502；D端记录对应bootstrap room在 `KVPoll.WaitingForInput` 等待300秒超时，路由器记录向P请求发送失败。两端模型未退出。失败时已保存336条唯一成功结果，无完整评分，后续MATH未被派发。

失败27成员负证据包哈希为 `5ab69d59a26c5f13538da39b7c77b90da93ff9f4c9191d484a5ee350c5f144ae`，已下载独立核验。底层根因未证明，不能把健康检查200称为故障不存在。

12:05:41启动具名 `accuracy-model22-recovery03`：保留Human与第一轮GSM完整结果，336条成功预测原字节复用，只生成第二轮剩983题，原计划随后执行MATH两轮；该后续安排已按用户最新要求取消。实际完整 `run_task` 的CPU请求前哨兵已确认336复用/983剩余/HTTP0。模型、provider、采样参数、输出预算和并发8均保持；任一新错误立即持久化并停止后续派发。

恢复控制器和评测并发工具只属于独立验收客户端，不复制到 adapter/plugin 运行时，不把未完成预测当作完整得分。

## 7. 复核、回归与交付条件

新增部署示例先做 `bash -n`，再用明确CPU替身捕获原配方与整理后配方的完整argv和关键环境，P的61项argv、D的59项argv均一致；缺少必要provider配置时在调用服务前拒绝。这里的项数为argv数量，不是GPU测试通过数。

节点仓库结构检查使用现有 `ci/l0_gate.py`。相关源码回归位于 `tests/runtime/`：

```text
compressed_w4a8.py / compressed_w4a8_loader.py
channelwise_int8.py / hcu_mega_compat.py
cp_transfer_import.py / cp_state_descriptors.py / cp_materialize_current.py
prefill_tilelang_abi.py / engram_host_pointer.py / batch_overlap_current.py
flashmla_masked_kv_gpu.py / logits_gather_gpu.py
```

GPU测试须根据各测试脚本的显式准入、资源和输出目录要求单独运行。当前完整模型评测期间不额外起GPU回归、服务层逐层Hook或性能负载。CPU/source检查、测试脚本存在和历史结果分别记录，不能宣称本轮重新运行了全部GPU测试。

完整交付仍需：候选GSM8K第二轮及两轮汇总、全量本地独立复核、新DAS同条件对照、5%精度判定、完整产物来源与归档。精度通过后复测代表性能并分别采集DAS/plugin的prof；正式性能不使用profiler扰动数据，且用户不要求性能一定对齐。

使用现有 `skills/hcu-plugin-dev/SKILL.md` 维护模型新增能力，使用 `skills/hcu-plugin-triage/SKILL.md` 定位错误。输入需包括固定源码、实际配方、provider版本和负日志；输出必须保留源码差异、独立验证和未覆盖范围。

## 8. 本次源码整理的实际检查记录

2026-11-15已将部署示例、源码锁定清单和本文镜像加入节点 plugin 的 `examples/deepseek_v41_w4a8/`，并在 `examples/README.md` 增加入口。本次没有对本地两仓执行合入，没有提交或推送。

| 检查 | 结果 |
|---|---|
| 节点 L0 | 通过，625个Python文件、8个Skill、errors为空；测试夹具修复后再次通过 |
| 部署示例 | 四份Shell语法通过；P61项argv、D59项argv及关键环境与现有模型22配方一致；缺少必要配置拒绝启动 |
| 当前运行源码保护 | 两仓运行文件与整理前一致，plugin原36项选定源文件SHA保持；没有重启模型 |
| 文档 | UTF-8完整回读通过，代码块成对，源码清单与快照逐项一致 |
| 默认单测 | 共发现135项，首轮90秒到时后继续未完成项；不把超时改写为成功 |
| 两项冷启动检查 | 在明确的CPU隔离环境中通过；不代表真实HCU启动验收 |
| 剩余环境限制 | `test_mrope_delta` 导入官方框架时调用设备属性查询，在GPU隐藏的CPU环境中报无可用设备；默认整套单测不宣称全通过 |

CLI测试夹具使用空的普通 `ServerArgs` 类，但只提供了 `is_record`，漏供当前 `server_args.py` 实际导入和调用的 `field_names`。本次仅在 `tests/test_hcu_runtime.py` 对应夹具增加 `module.field_names = lambda cls: ()`，保留原CRLF及全部原断言。它没有增加或模拟模型功能，也没有修改adapter或运行数学。

另经独立观察确认，容器的正常Python启动在导入plugin前已预加载Torch；任务私有的CPU `sitecustomize` 夹具用于隔离这项环境影响。它不安装到源码仓库或全局环境，不用于实际模型/ABI资格。原环境中的两项失败、首轮超时及GPU隐藏的模块导入失败均保留在独立交付证据中。
