# hcu_sglang 开发迁移记录 — 2026 年 10 月卷（归档）

> 本卷是 [hcu-sglang-development-202610.md](hcu-sglang-development-202610.md) 的 2026-10 切分归档，正文逐段原文保留；
> 当前开发与验证记录见主册（主册时间索引含本卷回链）。
> 跨卷引用以文内显式文件名为准；历史结果不代表当前代码已重新验证。

---

## 2026-10-26 V3 2026-10-26 实施与验证记录

> **用途**：记录 `plugin-v3-1026.txt` 指定输入下的插件开发、内嵌源码生成、HCU 逻辑对照及本轮验证。执行跨 2026-10-26 至 2026-10-27（Asia/Shanghai）。方法库保留 Skill、工具和记录，运行时代码位于独立 `output-code` 仓库。

> **当前交付**：以下 §1–7 保留各阶段快照；正式输出、最终验证和限制统一见 [§8](#2026-10-26-8-最终交付与需求验收)。

### 2026-10-26 1. 固定输入与产物

工作根目录以 `${BASE}` 表示；下表保留历史相对布局，不要求当前部署沿用。`${BASE}` 仅是历史位置标识，执行任务使用 [路径配置](configuration.md#1-当前工具的路径配置) 中的本地变量。

| 对象 | 路径及版本 | 状态与用途 |
|---|---|---|
| 社区输入 | `input-code/sglang`；`baa9befa381511cd4d56f83980951e5a45fffc6d` | 干净来源，提供当前框架和正式插件接口 |
| HCU 输入 | `input-code/sglang-das`；`15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e` | 干净来源；较上轮增加 139 个文件改动 |
| 插件输出 | `output-code/sglang-plugin`；v0.3.1，`39038d5394c23e6b918e6e50911ee5292a1e4d34` | 独立包 `sglang_dcu_plugin`，入口 `dcu` |
| 回溯输出 | `output-code/sglang-das` | 由上述社区与插件两个干净输入生成，清单记录来源 SHA |
| 方法库 | `hcu-sglang-plugin` | 本文、Skill、抽取工具修正与验证记录 |
| 执行容器 | 历史验证容器 | Python 3.10.12；宿主 Python 3.6 不用于工具执行 |
| 证据目录 | `hcu-sglang-plugin/artifacts/v3-20261026` | 原始日志、资源准入、响应、哈希与逻辑对照，Git 忽略 |

开始执行时 `output-code` 不存在。插件起点从保留的源码快照恢复，运行时逐文件与历史 `45e67e0` wheel 一致，记录见 `baseline-recovery.json`；新仓库恢复提交为 `bb76e1b`，没有冒充恢复原仓库已移除的 Git 历史。历史验证结果没有计入本轮通过数量。

### 2026-10-26 2. 已实现模块与证据边界

| 模块 / 实现位置（插件仓库内） | 本轮建设内容 | 本轮证据与限制 |
|---|---|---|
| 注册与参数解析：`__init__.py`、`server_args*.py` | 保留正式入口，新增配置与通信目标，目标合同更新为 210 项 | 严格 loader 检查非空、预期集合、全量 apply、实际来源与三个 resolution seam；注册不代表所有分支均已执行 |
| V4.1 配置：`configs.py` | HCU Channel FP8 转为 compressed-tensors channel-weight / token-activation 配置，接在官方规范化入口之后 | 通过真实 `DeepseekV41Config` 构造入口；其他量化格式和非 HCU 保留基线。未运行完整 V4.1 模型 |
| 通信：`distributed.py` | HCU native all-reduce 在 Graph capture 中使用 copy-in，保留 disabled/shape 准入和非 capture 的有效基线 | CPU 分支断言及真实 TP4：每 rank 两个 Graph 分配，三轮按 rank 交叉选择重放；不外推 AITER Fabric、DP 模型或全部拓扑 |
| 方法抽取：`models/deepseek/**/methods` | 七个 V2/V4/NextN 构造函数显式绑定拥有类；修复八处二次常量限定 | 修复前八项执行失败，修复后通过；覆盖构造父类边界和 MHC norm 分支，不能替代整模型构造与生成 |
| 常量别名：`models/deepseek/hook_utils.py` | 覆盖前保存旧引用，仅刷新已存在的 DeepSeek 消费者别名；缺失属性不视为 `None` 别名 | 两项修复前失败 / 修复后通过；不向无关 vendor 模块注入属性 |
| JIT 原生资源：`jit_overlay` | 更新 `c1.cuh`、`c2.cuh`、`c_plan.cuh`、`fp4_indexer_rope.cuh`、`fp4_utils.cuh` | 283 个 native 文件及一项许可资源，共 284 项与当前 HCU / embedded 哈希一致。新增 FP4/压缩算子的完整数值验证未执行 |
| MiniMax 与层边界：`models/minimax_m2*`、`layers/layer_boundary.py` 等 | 保留 residual-stream 与 FFN 出口归约契约；没有重新引入旧生产者重复 all-reduce | 当前 CPU、GPU 与本轮 TP4 服务结果。该兼容实现与历史 fork 文本可不同，需按最终输出与归约所有权比较 |
| 内嵌集成 | 复制 442 个 plugin 文件，注入官方 bootstrap、loader、依赖、许可及 package-data | 448 项生成清单通过；SGLang 入口 bootstrap 替代外部 `usercustomize.py`，属于显式集成差异 |

### 2026-10-26 3. 故障复现与工具修正

#### 2026-10-26 3.1 常量重写与类闭包

旧正则先把 `_is_hcu` 改成 `C._is_hcu`，又在下一轮替换 `_is_hcu`，产生 `C.C._is_hcu`。类方法抽成普通函数后，原来的 `super()` 也失去隐式 `__class__` 闭包。仅检查 AST、注册和 apply 均无法发现这些执行错误。

当前 [方法同步工具](../tools/sync_deepseek_methods.py) 根据 AST Name 的 UTF-8 字节位置修改引用：不改变属性名、字符串和注释；重复执行不会再次限定；遇到本地同名变量拒绝猜测。构造方法使用明确拥有类和 receiver；保留普通装饰器，遇到 classmethod/staticmethod/property 描述符时要求先声明绑定契约。该改进不代表历史抽取 profile 已适配当前 HCU 的类拆分，`--check` 仍需完整预检。

回归包括真实函数执行、父类调用、重复变换、中文文本偏移、局部变量冲突、装饰器行为以及来源符号漂移时不写输出。方法库完整门禁结果单独记录，不计入插件测试数。

#### 2026-10-26 3.2 预导入别名

旧 patcher 先修改 owner 常量，再调用传播函数；传播函数读到的已是新对象，因而跳过旧消费者。修正后先捕获旧值并限制作用域，只更新明确存在、仍引用旧对象的 DeepSeek 消费者。测试同时检查无关模块和缺失属性，避免修复变成全局副作用。

#### 2026-10-26 3.3 执行身份

容器 root 执行只读 Git 审计时触发仓库所有权检查。审计、源码构建及回溯改由源码 UID 执行；运行服务沿用容器已有合法设备用户。未修改全局 Git safe.directory、HOME、系统 Rust 或设备权限。

### 2026-10-26 4. 源码、内嵌与 HCU 逻辑对照

本轮执行了三个不同层次的比较，原始结果为 `fork-logic-comparison.json`：

1. **产物保真**：442 个内嵌插件文件逐字节等于本轮插件；448 项生成清单核验通过。启动集成使用生成器定义的 bootstrap 映射，不能把不生成全局 `usercustomize.py` 误报为丢文件，也不能忽略入口顺序。
2. **来源保真**：284 项 JIT 资源逐项比较当前 HCU、插件、embedded 三方；Channel FP8 转换 helper 的 AST 与当前 HCU 相同。
3. **行为与版本差异**：检查整个 `python/sglang` 当前树，社区与 HCU 有 1206 项内容不同、387 项 HCU 独有、20 项社区独有。抽查的 24 个方法中，21 项需要适配/陈旧行为复核，3 项源拥有类或方法已变更。不能把文件名、签名、AST 相似或 210 个目标全部 apply 当成整个 fork 逻辑等价。

**对照结论：本轮交付形态和明确验证的契约已有证据；整个 HCU fork 的全逻辑等价尚未证明。** 当前 HCU 的 CP LayerSplit 全链路、候选/分页 FP4 索引、PD/DSpark 辅助状态、V4.1 中间 chunk decoder、EnGram、W4A8/DeepEP，以及 AOT `rope_fp4` 构建导出没有在本轮形成完整运行验收。这些能力不得标记为已迁移或已验证。

对旧源码差异的处理遵循当前 caller/consumer 契约。例如 MiniMax 的框架 `FfnExit` 已负责归约，保留 fork 旧生产者归约会重复求和；此处应保留有回归与服务证据的适配差异，不能为了文本相等恢复旧操作。

### 2026-10-26 5. 验证矩阵

| 检查 | 插件形态 | 内嵌形态 | 原始证据 |
|---|---|---|---|
| CPU 与契约 | 304 passed / 8 skipped | 304 passed / 8 skipped | `plugin-cpu-final.log`、`embedded-cpu.log` |
| 严格加载 | 210/210，三个 resolution seam | 210/210，屏蔽外部入口 | `plugin-runtime-final.json`、`embedded-runtime.json` |
| 单设备 GPU / Graph | 8 passed | 8 passed | `plugin-gpu-final.log`、`embedded-gpu.log` |
| TP4 native collective Graph | 四 rank × 三轮通过 | 四 rank × 三轮通过 | `plugin-collective-graph-final/rank-*.json`、`embedded-collective-graph/rank-*.json` |
| MiniMax TP4 服务 | health 200；`2、2、Paris`；正常 stop；自有进程清理 | health 200；`2、2、Paris`；正常 stop；自有进程清理 | `*-service-attempt1/*-service-result.json` 与原始响应 |
| 方法库门禁 | L0 通过；45 项工具单测通过 | 同一方法库，不重复计数 | `method-l0-final.json`、`method-tests-final.log` |

原始配方为上级 `minimax-debug/d.sh`；本轮按已有授权使用物理 4–7 卡 TP4，仅修改 TP 数和 NUMA 列表，保留 FA3、FP8 权重/KV、Graph、context 131072、chunk/prefill 和服务端口。`recipe-manifest.json` 保存原文与派生脚本哈希、精确差异及可见设备映射。

确定性响应通过 [响应校验工具](../tools/validate_generation_response.py) 校验实际模型、完整最终答案、正常 stop 和控制字符。该 checkpoint 在本次响应 content 中返回闭合前导 think 段，使用本次明确允许的 `allow_think_prefix`，未将 `enable_thinking=False` 宣称为全局停用思考的保证。重复请求记为 repeat，不作为 cache hit 证据；本轮不宣称 TP8、数据集精度或性能已通过。

### 2026-10-26 6. 复现与 Skill 入口

- 三方输入/差异审计：按 [迁移指南](migration.md) 调用 [迁移 Skill](../skills/sglang-hcu-plugin-migration/SKILL.md)，固定本轮 SHA 后生成新的审计和台账。
- 修改能力或抽取方法：按 [插件开发指南](plugins.md) 调用 [开发 Skill](../skills/hcu-plugin-dev/SKILL.md)，先保留真实构造、装饰器、状态与出口边界，再执行对应回归。
- 生成新 embedded：按 [回溯指南](backtracking.md) 调用 [回溯 Skill](../skills/hcu-plugin-backtrack/SKILL.md)。两个生成输入为 official/plugin；提供的 HCU 输入用于独立逻辑对照，不替换生成来源。
- 安装交付：按 [安装指南](installation.md) 使用已有源代码或 wheel，先验证实际解释器和模块来源，再运行目标服务。

证据脚本在本轮 ignored artifacts 中固定路径与 profile，只作为可复现实验记录；未来版本按对应操作指南重新固定输入和执行范围，不复用本页通过状态。

### 2026-10-26 7. 逻辑对齐后续工作

以下保留 v0.3.1 起的检查点与后续开发过程；各阶段的 pending 状态按当时范围解释，当前需求验收见 §8。完整 HCU fork 的所有模型与分支等价不属于这些局部通过结论。

后续开发位于独立的 `output-code/.sglang-plugin-v3-parity`（分支 `v3-hcu-parity`），以本页插件提交为起点。初始提交 `f2394ab` 迁移了 CP LayerSplit 的十个模块及分页/状态/描述符回归；后续 `0.3.2.dev1` 已接入父池、容量预算、工厂和握手规范化，进展见 §7.2。完整模型前向与 PD 网络字节传输尚未闭合，不能标记为可启用的完整 LayerSplit 服务。


#### 2026-10-26 7.1 CP 索引重排的设备准入修复

候选第二次局部回归为 23 项通过、1 项失败：HCU 主机上的 CPU 张量进入 LightOP 的 device-only remap，报 `indices must be device int32 or int64`。设备能力与输入 tensor 的实际设备不是同一个条件；当前候选同时检查 HCU 与 `indices.is_cuda`，CPU 张量使用原 PyTorch 路径。

修复后，候选局部测试为 **32 passed / 136 subtests passed**，证据 `cp-layer-split-candidate-attempt3.log`。新增对照包含真实 LightOP 的 int32/int64、乱序页、静态负值 padding、缺页、负 token ID，以及三轮变输入 Graph 重放；CPU 路径另检查不导入设备专用依赖。状态机测试中的部分边界沿用 HCU 的 AST 节点提取方式，不能代替完整模块导入、pool 构造或模型运行。方法来源及适配哈希在候选 `source-provenance.json`，能力接线状态在 `PARITY-CANDIDATE.md`。


#### 2026-10-26 7.2 CP 父池、容量与握手接线

候选 `0.3.2.dev1` 的实现提交为 `fe45284b9f17e0fde572d72a0751e9b8cddebb56`，显式 GPU 测试准入提交为 `1f1ff9d75d0857e5914b8fa328bce8acafd9cf87`。与最初仅搬迁模块的检查点相比，本次保留当前 HCU 的完整缓存池类，复用 AST 相同的官方值类型和独立 unified pool，通过正式 class REPLACE 接入五个缓存池类。同步迁移 DSV4 预算器、CP 工厂、容量 MIN 及 DSpark 全局描述符规范化。原有六个局部池 Hook 由一致的池构造/状态契约替代，候选的 HCU 目标集合变为 214 项。

声明的父池、INT8 helper、masked store、预算类、工厂及握手共 37 个类/函数节点，在明确 namespace 和插件进程选项映射后与当前 HCU AST 一致，证据 `hcu-pool-source-comparison.json`。该对照覆盖声明的节点与复用边界，不能替代其余模型/网络链路审阅。

本次增加了真实父池和四个 CP rank 的构造、公共工厂、分配缓冲与传输描述符、masked BF16 Graph，以及 packed FP4 threshold / RNE 舍入和变位置 Graph 读写。四卡验证使用真实 PyNCCL、真实 `ServerArgs` 投影和池分配，覆盖三轮 SWA/低压缩比 KV、非 owner 未预取拒绝、空查询 rank 参与、共享 staging 重用、容量最小值和真实分片/未分片池握手；没有启动 DeepSeek 模型引擎。

| 检查 | 候选源码 | 候选内嵌 | 原始证据 |
|---|---|---|---|
| 原有 CPU 回归 | 304 passed | 本阶段未重复全量 CPU；按改动执行下列池/设备验证 | `cp-parent-baseline-cpu.log` |
| CP CPU 合约 | 40 passed / 1 skipped，138 subtests；排除 11 个 GPU 用例 | 局部合约 38 passed / 2 skipped，2 subtests；排除读取候选源码的 AST 测试，包含实际池和设备边界 | `candidate-cp-cpu-final.log`、`candidate-embedded-cp-contracts.log` |
| 显式 GPU lane | 18 passed / 1 skipped | 18 passed / 1 skipped | `candidate-gpu-optin-final.log`、`candidate-embedded-gpu-optin-final.log` |
| CP4 真实池/通信 | 四 rank 全部通过 | 四 rank 全部通过 | `cp-tp4-pool-attempt2/rank-*.json`、`candidate-embedded-tp4-pool/rank-*.json` |
| 严格加载 | 214/214，正确源码来源 | 214/214，屏蔽外部入口并固定内嵌来源 | `candidate-pool-runtime-final.json`、`candidate-embedded-runtime.json` |

GPU lane 包含原有八项和新增十项实际通过用例。最初未设置 opt-in 的原有 GPU 运行仅产生八项 skip，该日志不计通过；随后明确设置 `HCU_PLUGIN_RUN_GPU_TESTS=1` 执行。CPU 与内嵌局部 lane 范围不同，不按通过数直接比较覆盖率。

回溯产物位于 `output-code/.sglang-das-v3-parity`，由干净候选和官方两输入生成。466 项清单核验通过，含 460 个插件文件。内嵌检查屏蔽外部入口，并确认 SGLang 与插件模块均来自该输出树。AST 提取测试若通过测试路径定位候选源码，就不能计作 embedded 覆盖；因此内嵌局部回归排除了这类测试，另以实际导入、构造和设备行为验收。

首轮 CP4 失败是验证器误把 PyNCCL 初始化后的默认 `disabled=True` 当作不可用。直接探针确认 RCCL 2.22.3 与基础符号正常；当前先验证 `available`，再在 `change_state(enable=True)` 作用域内断言真实归约数值，随后执行池广播。初始失败日志保留，没有把跳过 collective 当作通过。

两项 INT8 池用例要求源实现支持的 gfx936；当前节点为 gfx938，按 gate 跳过。此处不声明该特性已在本节点验证。候选 wheel 已构建并核对 460 个插件文件及新增 provenance 资源；尚未把它作为隔离 wheel 安装形态运行，不计作 wheel 运行验收。

CP 父池阶段结束时仍待完成正式启用入口、模型 prefetch/write/read 消费者、prefill/decode 描述符生产、Mooncake 追加字段及匹配复制和 DSpark 辅助状态。后续 PD 协议进展见 §7.3；整体集成完成后再合入正式输出并重新验收服务。`HCU_PLUGIN_CP_CACHE_LAYER_SPLIT` 目前仅供候选池/通信契约使用，不是完整模型能力承诺。


#### 2026-10-26 7.3 PD 描述符生产、协商与复制

候选升级为 `0.3.2.dev2`，提交 `f9a838c0de11e369bee70125c9703d06636c6763`，仍位于独立 `output-code/.sglang-plugin-v3-parity`。本阶段从公开 `setup_state_kv_args` 补齐 V4 描述符生产，将 CP 拓扑、KV 布局、类型信息接入 HTTP bootstrap，连接 decode 真实 receiver 构造、ZMQ 注册与 prefill peer 发布。注册消息保留旧 0–19 字段（ZMQ 帧），20/21/22 分别追加 state formats、state types 和 V4 描述符；短旧消息仍可解析。

复制入口按照 `(family, global layer)` 匹配指针和每项字节跨度。对来源实现保留的两处问题，本次补充了失败反例与适配：

- 描述符合法但目标顺序不同，位置式 stride 前检可能先返回 `-1`；函数级四项反例覆盖未缓存 verdict、单批/分批/线程路径。现在描述符路径由身份匹配检查 ABI，位置式检查继续用于原有路径。
- 状态组件已按类型和重复次序解析目标组件，目标描述符却仍用原索引；一项重排反例在修复前拒绝合法布局。现在指针、长度、索引与描述符都沿解析后的目标组件读取。

peer 发布前检查描述符与指针/stride 数量、二进制字段与 JSON 中 KV 大小的一致性；坏的重新注册会撤销旧有效 verdict。上游 worker 保留原控制流，只增加其当前注册快照的描述符参数，保证与该次指针列表一致，未改成重新读取可变 peer 表。

32 个 HCU 来源节点及一个上游 worker 调用适配完成明确映射后的 AST 对照，包含 async HTTP handler、classmethod 恢复、显式 super owner、文档字符串缩进及上述功能修复。`pd/pd-source-comparison.json` 和插件内 `disagg/source-provenance.json` 保存范围与哈希。该结论不覆盖未迁移模型及流式状态分支。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 369 passed / 1 skipped；11 GPU deselected，138 subtests | 本阶段按 PD 改动面验证下列项 | `pd/full-cpu-attempt1.log` |
| 新增写入 drain / 公开链路 | 3 passed（2 个 drain、1 个隔离公开入口） | PD 单测及公开链路共 28 passed | `pd/drain-public-integration.log`、`pd/embedded-contracts.log` |
| 严格加载 | 241/241，正确来源 | 241/241，外部入口屏蔽 | `pd/runtime-final.json`、`pd/embedded-runtime.json` |
| 原生 Mooncake TCP 公开链路 | 通过 | 通过 | `pd/native-public-path-final.json`、`pd/embedded-native-public.json` |
| 显式 GPU lane | 18 passed / 1 skipped | 18 passed / 1 skipped | `pd/gpu-optin.log`、`pd/embedded-gpu.log` |

公开链路采用真实 CPU 池分配、HTTP/ZMQ 协商、真实 receiver 构造和原生 Mooncake TCP session / memory registration。通过公开 KV 与状态发送方法后，逐字节比较目标缓冲，包括未应写入区域。独立的有界 CPU engine 模式保留作回归，它与原生 TCP 证据分开记录；两者均明确不是模型服务或 RDMA 硬件验收。恢复旧 peer 前检的正常顺序组合对照也通过，因此不将函数级失败反例描述为所有基线调用都会失败。

新增写入完成测试覆盖一个 future 异常或返回错误、另一写入仍在执行的情况：确认 helper 等待已启动写入结束后才返回或抛出，防止提前释放 buffer。CPU 失败处理证据不等同于完整流式源事件/ACK 生命周期已闭合。

回溯使用干净的官方 `baa9befa...` 与候选 `f9a838c...` 两个输入，475 项清单核验通过，其中 469 个插件文件；证据 `pd/embedded-manifest-verify.json`。内嵌加载屏蔽外部入口，原生 TCP 验证器同时固定 SGLang 与插件的实际导入根路径。

已验证的 v0.3.1 主输出与其 448 项生成清单保留。本阶段没有把新候选的协议/设备结果写成重新通过 MiniMax 模型服务。

下一阶段仍需模型 prefetch/write/read 消费者、DSpark hidden-state 流式生产与 ACK、source event / 在途 buffer 生命周期、候选索引/稀疏 prefill、W4A8/DeepEP 和其余模型 caller 差异。整体目标仍在进行，不将本阶段协议路径通过写作完整 HCU fork 等价。


#### 2026-10-26 7.4 MQA 类图、缓存读取与原生边界

候选 `0.3.2.dev3` 的实现提交为 `5b182d7f8e39cd1637ca2c10bafb033c9e619911`。从当前 HCU 保留完整 `MqaAttentionBase`、`MQALayer` 与 `DSparkAttention` 类，替代四个旧 MQA 方法 Hook；DSpark 显式继承候选基类，避免仅更新模块别名仍留下旧继承关系。后端新增 metadata prepare、tail reset、普通/sparse-prefill 缓存读取及原生 FlashMLA 接口，HCU 严格目标集合为 246 项。

实际池与 ForwardContext 回归覆盖 CP 空查询的 prepare→SWA/extra prefetch→wait→返回顺序；旧 forward 在同一边界的反例为 1 failed / 1 passed，保留于 `attention/empty-cp-before.log`。普通 attention 的页索引和 sparse-prefill 独立 flat indices 分别重映射，检查共享 metadata 未被改写。昂贵计算和通信载荷可替换，但不绕过决定调用顺序的控制流。

真实 GPU 验证中发现并修复三项问题：

- 独立无 residual 的 RMS 进入当前环境缺少的 vLLM `_C.rms_norm`；HCU 设备路径改用实际可用 LightOp，连续/非连续输入及变输入 Graph 有独立参考。原 residual 和 batch-invariant 路径按现有契约保留。
- overlay 哈希和 extra include 正确，但 loader 的 `DEFAULT_INCLUDE + extra` 使官方同名头文件仍优先，真实 JIT 报缺少 HIP warp 宏。注册时在 BuildSpec 冻结前调整共享默认列表的有效优先级，验证引用、幂等和非 HCU 分支；没有修改输入源码或全局工具链。
- 当前 FlashMLA dtype matcher 拒绝 FP8 FNUZ typed carrier；在 HCU native 入参边界使用 uint8 view，保持 pointer、shape、stride 与 bytes。BF16 和非 HCU 保持各自载体。

24 个来源节点/构造器后缀、9 个实际复用的官方 HIP helper 完成限定 AST 对照，证据 `attention/source-comparison.json`；插件内 `models/deepseek/current_v4/source-provenance.json` 记录源/目标哈希、装饰器映射、carrier 适配及命名空间边界。构造器沿官方初始化后追加 HCU 选项，不把整个 HCU 后端构造器标记为搬迁完成。

小型 MQA 使用真实 GPU 权重、norm、RoPE、KV store、FlashMLA 和输出投影，分别执行 FP8 与 BF16 gather。Q 对独立 PyTorch 参考、输出对直接原生 attention 加独立投影参考均为最大差值 0。backend metadata 由显式 fixture 提供；这不是完整模型引擎，也不是原生 attention 算法的独立 FP32 精度证明。

当前回溯由干净官方与 `5b182d7...` 两输入生成，482 项清单通过，其中 476 个插件文件。wheel 的全部 476 个插件文件与候选逐字节一致，含新 provenance；wheel 资源验证不计作隔离安装运行验证。验证结果及边界如下表。

| 验证 | 候选源码 | 候选内嵌 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 384 passed / 1 skipped，16 GPU deselected，138 subtests | attention 与 CP 导入/构造合约 40 passed / 1 skipped，16 GPU deselected，2 subtests；排除从候选测试路径读源码的 AST 用例 | `attention/full-cpu-corrected.log`、`attention/embedded-contracts.log` |
| 严格加载 | 246/246 | 246/246，屏蔽外部入口 | `attention/runtime-final.json`、`attention/embedded-runtime.json` |
| 显式 GPU / Graph | 23 passed / 1 skipped | 23 passed / 1 skipped | `attention/gpu-regression.log`、`attention/embedded-gpu-attempt2.log` |
| 小型 MQA FP8 / BF16 gather | 两种模式均通过 | 两种模式均通过，实际后端来自内嵌树 | `attention/mqa-gpu-attempt5.json`、`attention/mqa-bf16-gpu.json`、`attention/embedded-mqa-*.json` |

GPU lane 新增 packed KV/upconvert、变选择与长度 Graph、超过 int32 的 4GiB output stride，以及独立 RMS；gfx936 专用 INT8 用例在当前 gfx938 继续跳过。首次 CPU 回归的唯一失败是旧注册测试同时要求并禁止 class REPLACE，修正为按平台检查目标集合后全量通过。

内嵌 GPU 首轮为 22 passed / 1 failed / 1 skipped：MiniMax 测试的 PyTorch 参考在 `hipblasCreate` 报分配失败。该例隔离运行通过，同一完整 GPU 组重跑为 23 passed / 1 skipped；未改生产代码，也未确定首次失败的根因，保留 `attention/embedded-gpu.log` 与 `attention/embedded-router-isolated.log`。

完整 CP compressor/indexer、DecoderReplayTail runner/模型循环、DSpark streaming source-event/ACK、W4A8/DeepEP 和完整 DeepSeek 服务仍待集成。本阶段不替换原 v0.3.1 主输出，也未将局部 attention 结果记作重新通过 MiniMax 服务。相关经验与开发 Skill 使用方式见 [类图与空查询](development-methodology.md#14-attention-类图与空查询通信顺序)、[JIT 头文件](development-methodology.md#15-jit-实际头文件优先级) 和 [原生接口](development-methodology.md#16-原生算子-provider-与-fp8-carrier)。


本阶段保留性审计还发现官方索引文件 SHA 从 `606b51f2...` 变为 `acca1b00...`，mtime 对应回溯来源检查；HEAD 和源码工作区均未变，当前 index tree 与 HEAD tree 同为 `27762040...`，暂存 diff 为空。原始索引字节未备份，故不声称原索引字节保持不变，细节见 `attention/official-index-refresh.json`。生成器现对子进程设置 `GIT_OPTIONAL_LOCKS=0`，新增真实双来源 mtime 反例；旧实现两项子检查失败，修复后方法工具 46 项单测通过。后续实际回溯再次检查前后输入索引哈希，结果保存在 `attention/index-preservation-regeneration.json`。这一修复不修改插件实现，内嵌运行结论仍限定于已核验的相同产物字节。


#### 2026-10-26 7.5 低压缩比缓存的持久写入边界

候选 `0.3.2.dev4` 已提交为 `02e11c96418310bd3a69fcdfc606c988aea29791`：正式接入 C1/C2 decode、C2 prefill 的 Python wrapper 与 backend `_low_ratio_compress_fused`，新增四个目标，HCU 合同总数为 250。该调用点使用 `get_extra_key_buffer_for_write`，不再把读取用紧凑 staging 当持久 KV；非 owner 取得 None，由 Python 零页载体与既有 native no-store 分支配合，继续计算 latent、维护 C2 ring 并发布复制的索引 K。

7 个 HCU 来源函数完成 AST 对照，仅调整模块级相对 import 路径；使用的 `c1.cuh`、`c2.cuh` 与固定 HCU 来源逐字节一致。证据为 `compressor/source-comparison.json` 和插件内 `compressor-provenance.json`，不动态导入 fork。

反例分为两层：旧 wrapper 的三个无主 KV 场景在 `.view` 处失败；旧 backend 的六个池边界对照为 3 failed / 3 passed，失败均为非 owner 错把读 staging 传作写目标。新的六个真实 CP 池边界均通过，断言 no-store 的同时仍发布索引 latent，且共享 staging 字节不变。首版测试误以为读 getter 在所有阶段均应抛错，改为按实际池生命周期验证写入所有权，没有为迎合测试修改生产池行为。

原生六项 GPU 验证覆盖 C1、C2 decode、C2 prefill 的 owner/no-store，每项三轮变输入 Graph；对照独立 PyTorch latent 与 pair-state 参考，另检查 V4 页内 payload、scale 和保留区的写入范围。正式 loader 的函数 REPLACE 使用 `functools.wraps` 保留原 metadata，首次 provider 断言因把 `__module__` 当实现来源而失败；后续从正式公开函数调用验证实际行为，由严格加载报告核对目标。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU | 390 passed / 1 skipped，22 GPU deselected，138 subtests | 本阶段按 attention/写入改动验证，18 passed，11 GPU deselected | `compressor/full-cpu.log`、`compressor/embedded-contracts.log` |
| 严格加载 | 250/250 | 250/250，屏蔽外部入口且固定内嵌来源 | `compressor/runtime-final.json`、`compressor/embedded-runtime.json` |
| 显式 GPU / Graph | 29 passed / 1 skipped | 29 passed / 1 skipped | `compressor/gpu-full.log`、`compressor/embedded-gpu.log` |

回溯的 485 项文件哈希通过，包含 479 个插件文件；输入索引前后相同，见 `compressor/index-preservation-reconstruction.json`。wheel 内全部 479 个插件文件与当前候选相同，含新增来源记录；仅做资源核验，尚未作隔离 wheel 运行验收。

完整 CP query-lens、candidate/top-k indexer、模型循环和流式 ACK 生命周期仍待集成；本节不构成完整 CP 模型服务声明。新增经验与开发 Skill 的检查入口见 [持久写入边界案例](development-methodology.md#18-缓存读取-staging-与持久写入目标)。


#### 2026-10-26 7.6 完整后端依赖图与 CP 入口集成

候选进入 `0.3.2.dev5`，在此前逐方法接线后改为完整 `DeepseekV4AttnBackend` 类及相关 metadata、C1/C2/C4 indexer、compressor、候选与 CP strategy 依赖图。固定 HCU 来源的 42 个模块共 482 个顶层类/函数节点、9 个额外桥接函数节点完成声明映射后的 AST 对照；`backend/source-comparison.json` 与插件内 `hcu_backend/source-provenance.json` 分开记录原始哈希、导入路由、来源缺口和有意修正。旧后端成员 Hook 已退出，避免 class REPLACE 后再由旧方法覆盖。新增状态与 API 后 HCU 严格目标集合为 363 项。

新增十个 ForwardBatch 字段采用继承官方记录的 dataclass 声明，保留原有方法，使构造、复制及 pickle 保留新增状态。ServerArgs 继续使用固定官方 schema；`--enable-cp-cache-layer-split`（别名 `--hcu-cp-cache-layer-split`）和 `--disable-cp-cache-layer-split` 转为进程选项并由 worker 继承。CLI 解析及完整 ModelConfig 派生分别验证，不将仅解析成功当作运行时配置完整。

正式启用检查复用当前 HCU LayerSplit 约束：DeepSeek V4.1、compression ratio 0/1/2、prefill PD、interleave CP、CP>1、PP1、Mooncake 与 Compressor V2，且 prefill Graph disabled。HiSparse/HiCache、encoder SWA replay、单独 staging transport、Q8 sparse prefill 和不支持的 speculative 组合仍被拒绝。这些是来源的支持边界，不为通过测试删除约束。

本阶段复现并处理以下数据接口问题：

- Torch 候选源在宽度 259 时给出未对齐 FP32 行，原生 selector 拒绝。对齐 carrier 仅在该 caller 准备，低层 selector 的非法 dtype/offset/stride 拒绝保持不变；三种布局的变输入 Graph 与独立 mask/ID oracle 验证实际 copy 和只读合同。
- legacy/interleave 的 Python list/tuple 被当成 Tensor 使用 range 高级索引；interleave 四 rank 的反例均失败，修复为 slice 后通过，Tensor 路径保持来源行为。

BF16 KV 对照还复现旧 compressor Hook 读取环境变量的接口错误：代码导入 `environ` 模块，却按 `envs` 对象访问 descriptor，旧单测又向错误模块添加虚构属性。Hook 改为读取实际 `envs`，两项回归改为注册实际 descriptor 并通过环境变量驱动分支；CPU 两项通过后继续执行真实 BF16 设备复验，见 [配置对象案例](development-methodology.md#22-环境变量对象与测试替身的接口一致性)。

输入 HCU 源码还保留 graph selector 对缺失 `glm5_next` 模块的无条件 HCU 导入，节点工作区与固定 HEAD 均确认文件不存在。候选移除该失效分支并保留可用模型的官方选择行为；该条作为来源缺口记录，不声明迁移了 GLM5Next HCU 优化。

真实 GPU 层链路采用实际 pool、SWA allocator、声明的 ForwardBatch、公共 metadata 初始化与两层随机权重 MQA。验证包括 ratio 1、ratio 2、264-token 候选源/消费者，以及 1030-token 跨页 top-k（从 1025 个历史位置选择 512 项）。确认命中原生 dense logits、候选 logits 与 paged sparse consumer；原生评分对同一 packed K 的独立 FP32 运算，输出对 FP32 评分的参考路径，已执行场景最终输出最大差为 0。该参考不构成独立 FP32 attention 算法或模型准确率验证。

真实四 rank 使用公开模型并行初始化，新增第二个 CP communicator 沿来源分区创建；验证 equal/variable all-to-all、peer 零计数、整 rank 空输入/空接收、两个通信器的双 stream 归约及销毁。`backend/collectives-tp4/rank-*.json` 保存逐 rank 结果；不是模型引擎或 RDMA 传输验收。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 最后完整回归 425 passed / 1 skipped，147 subtests；配置对象 2 passed；池视图相关 37 passed | 65 passed，9 subtests | `backend/full-cpu-final.log`、`backend/bf16-view-after.log`、`backend/embedded-contracts-final.log` |
| 严格加载 | 363/363 passed | 363/363 passed | `backend/runtime-final.json` |
| GPU / Graph | 37 passed / 1 skipped，13 subtests | 待记录 | `backend/gpu-regression-attempt1.log` |
| 实际 allocator / metadata / MQA | ratio 1/2、候选/paged consumer、跨页 top-k 均通过 | 待记录 | `backend/gpu-public-*.json` |
| CP4 独立通信器与交换 | 四 rank 均通过 | 待记录 | `backend/collectives-tp4/rank-*.json` |
| CP4 完整后端/层链路 | FP8；显式匹配批次参考，四 rank 最大差 0 | BF16；四 rank 最大差 0，实际来源为内嵌目录 | `backend/cp-mqa-tp4-*` |

新增的完整模型配置与四卡层探针暴露了合成 HF fixture 缺失 `sliding_window` / `index_head_dim`；补齐实际 ModelConfig 所需字段后再验证，不修改生产代码绕过配置派生。原始失败与每轮后续结果均保留。

设备验证前按已有授权释放了 `debug_lzg` 在物理 4–7 卡的四个调优进程，核对 cmdline、容器与进程 start-time 后发送 SIGTERM；保留该容器在前四卡的服务。`backend/stopped-tuners.json` 和设备/PID 快照记录实际范围，未将整容器停止记作执行结果。

本阶段仍未宣称完整 decoder/model loop、DSpark streaming source-event/ACK、W4A8/DeepEP、AOT FP4 或完整 HCU fork 等价。最终集成后仍须生成正式输出并重验原 MiniMax 配方。当前使用方法按 [插件开发指南](plugins.md#使用-skill-开发插件能力) 调用开发 Skill，选择 [前向记录](development-methodology.md#19-前向批次-schema-与复制生命周期)、[评分行对齐](development-methodology.md#20-候选评分的行对齐与消费边界) 或 [序列分片](development-methodology.md#21-python-序列分片与-tensor-高级索引) 对应回归，历史案例不构成全部任务的固定检查清单。

候选后端集成提交为 `19ec9e554a0cc27ab112bc143ba205f278692f46`。本次生成包含 532 个插件文件、538 项清单文件，逐项哈希核验通过；生成前后两个来源的索引字节哈希保持不变。wheel 的 532 个插件文件逐字节相符；尚未执行隔离 wheel 运行验收。

CP4 与单卡原生完整批次的对照仍未通过原数值阈值（`rtol=0.02`、`atol=0.003`）：四层最大绝对差约 0.01733。首层 Q 与单卡输入相同，首层注意力输出已有差异；该结果不能写成完整 CP 模型验收通过。关闭参考 compressor 融合后偏差未消失，继续检查实际原生输入与独立数学参考。相关原始失败保存在 `backend/cp-mqa-tp4-*.log`。

网络恢复后，后四卡被其他用户容器的 TP8 MiniMax 服务占用。按已有授权尝试停止容器，但节点权限以创建者不是当前执行用户而拒绝，未绕过权限或记录为停止成功。随后诊断在每卡约 7.6–8.0 GB 剩余显存下执行，探针设置约 3.9 GB 的 PyTorch 单进程分配上限；这不是设备空闲或独占的声明，也不构成性能测试。完整服务复验与需要较大显存的 GPU 回归仍需满足资源条件。

首层原生输入捕获与独立对照已完成：15 个查询的 Q、解码 KV 集合和 FP32 数学输出一致（`backend/cp-native-input-comparison.json`）。单卡重放相同输入时，完整批次与 CP 批次形状原生输出差为 0.0078125，按相同批次形状执行则与 CP 输出差为 0（`backend/native-shape-replay.json`）。验证器新增显式 `--match-reference-native-batch`，用于隔离这一因素；完整批次结果与匹配批次结果分别记录，未修改原阈值或生产注意力实现。见 [数值参考案例](development-methodology.md#23-原生计算批次形状与数值参考)。

BF16 设备复验在修复配置读取后进一步暴露 SWA 视图的单位错误：父池 getter 继承了来源中 packed FP8 的 584 字节宽度，但实际 BF16 分配的最后轴应为 512 个元素。修复前，真实父池及四个 CP rank 的五项 BF16 consumer view 均失败，五项 FP8 对照通过；修复后后端与父池合计 37 项 CPU 测试通过。该来源修正保留 FP8 和 uniform-FP8 行为，单独保存为 `backend/bf16-view-source-comparison.json`，不纳入 42 个后端模块原样映射的声明。修复提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，前一个验证器提交为 `6db84b4f27ce0c1da1a4068c6d3633f724b02032`。

最终内嵌 BF16 四卡复验通过：`backend/embedded-cp-mqa-tp4-bf16/rank-*.json` 四项均为 passed，四层 Q、注意力输出和最终层输出与参考逐元素一致，最大差为 0；实际 SGLang/plugin 路径均位于候选内嵌目录。使用显式匹配原生批次形状及 Torch compressor 参考；本次 BF16 逐层完整原生批次与匹配批次的差也均为 0。该结果限定于真实 allocator、metadata、CP 池与小型随机权重 MQA 层链路，不是完整模型引擎或 PD 服务验收。内嵌 CPU 最终回归为 65 passed、9 subtests；源池相关回归为 37 passed。

本检查点最终候选为 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，由该干净提交重新生成内嵌源码并重建 wheel。方法库仅保存文档、Skill 和工具，未放入运行时包；原 v0.3.1 主输出与其 448 项内嵌清单继续保留。候选尚未替换主输出，完整 MiniMax 配方的新候选服务复验仍需可用设备，历史通过结果不继承为本候选通过。

### 2026-10-26 8. 最终交付与需求验收

2026-10-27 在历史验证容器完成本轮验证，并将已验证提交同步到正式输出。运行时代码继续位于方法库之外；方法库的 HEAD 和暂存区保留原状，文档及 Skill 更新保存在工作区。

| 产物 | 正式位置（工作根目录下） | 来源与状态 |
|---|---|---|
| 源码插件 | `output-code/sglang-plugin` | v0.3.2.dev5，干净提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c` |
| 内嵌源码 | `output-code/sglang-das` | official `baa9befa381511cd4d56f83980951e5a45fffc6d` + 上述插件；538 项清单、532 个插件文件逐项核验 |
| wheel | `output-code/dist/sglang_dcu_plugin-0.3.2.dev5-py3-none-any.whl` | SHA256 `c2260d30015e932faf33c09082758dee675113785353717a2750f75775544fac`；独立 CPU 安装/加载已验证 |
| v0.3.1 回退 | 插件分支 `validated/v0.3.1-before-v3-1026`；`output-code/.sglang-das-v0.3.1-validated-backup` | 旧插件 `39038d5` 和原 448 项内嵌文件保留 |

#### 2026-10-26 8.1 最终验证矩阵

下列日志相对 `artifacts/v3-20261026/backend/`。服务原始响应分别位于上级 `backend-source-service-attempt3/` 与 `backend-embedded-service-attempt2/`。

| 验证 | 源码插件 | 内嵌源码 | wheel |
|---|---|---|---|
| 完整 CPU 合约 | 435 passed / 1 skipped，147 subtests | 435 passed / 1 skipped，147 subtests | 435 passed / 1 skipped，147 subtests |
| 严格加载 | 363/363，三个具名 seam | 363/363，屏蔽外部入口 | 363/363，实际模块来自独立安装目录 |
| GPU / Graph | 37 passed / 1 skipped，13 subtests | 37 passed / 1 skipped，13 subtests | 未执行 |
| TP4 native all-reduce Graph | 四 rank × 两套捕获分配 × 三轮重放，通过 | 同范围通过 | 未执行 |
| MiniMax TP4 服务 | 自有 health 200；最终答案 `2`、`2`、`Paris`；正常 stop；清理通过 | 同范围通过 | 未执行 |

CPU 日志为 `final-source-cpu.log`、`final-embedded-cpu.log`、`final-wheel-cpu.log`；GPU 为 `final-source-gpu.log`、`final-embedded-gpu.log`；通信为 `final-*-collective-graph/rank-*.json`。部分 AST 抽取用例读取外部测试仓源码，因此独立加载、启动隔离和实际模块路径另行核对，不由总测试数推断产物来源。

服务仍使用原 `minimax-debug/d.sh` 的用户授权派生配方，仅将 TP8 改 TP4、NUMA 改为 `2 2 3 3`，物理卡为 4–7。保留 FA3、FP8 权重/KV、Graph、context 131072、chunk/prefill 等参数。每轮在待测目录下验证新进程来源，识别自有监听者后调用真实生成并按完整最终答案验收；前导 think 只按本轮明确允许的闭合前缀处理。重复请求不作为 cache hit 证据，也不作性能结论。

服务验证实际执行于候选目录；正式输出具有相同提交与 532 个相同插件文件，并在正式目录另做严格加载和模块来源验证。最终来源报告为 `delivery-source-runtime.json`、`delivery-embedded-runtime.json`；wheel 为 `final-wheel-runtime.json`、`final-wheel-startup.json`。

#### 2026-10-26 8.2 HCU 逻辑对照结论

`delivery-fork-logic-comparison.json` 比较正式内嵌核心与固定 HCU 当前树，列出 1208 个内容不同文件、387 个 HCU 独有文件及 20 个内嵌核心独有文件；这些计数是审阅清单，不直接代表行为相同或不同。生成器以官方源码为核心，HCU 行为经插件注册接入。

532 个插件文件与生成树相同，284 项当前 JIT 来源资源核验通过；完整后端的 42 个模块、482 个顶层节点、9 个桥接节点及十个前向字段有声明映射对照，BF16 SWA 维度修正单独记录。GLM 缺失模块导入、Python 序列切片、候选评分行对齐、FP8 原生载体和 BF16 元素维度均是明确的适配/修正。

**结论：明确迁移的节点及已执行契约有来源和行为证据；没有宣称整个 HCU fork 全逻辑等价。** 完整 DeepSeek 模型引擎、全部 decoder/DSpark streaming source-event/ACK、W4A8/DeepEP shared-expert、AOT FP4 构建导出及所有原生分支仍未形成全范围验收。CP4 的原生完整批次数值差异与显式匹配批次对照分别保留，详见 §7.6，不将对照选项解释为生产精度修复。

#### 2026-10-26 8.3 需求与证据对应

| 原需求 | 实际交付与证据 |
|---|---|
| 1. 使用给定社区/HCU 完善插件 | 正式源码提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`；各模块及修复见 §2、§7 |
| 2. official/plugin 两输入回溯 | 正式内嵌目录、来源 manifest 与 538 项文件核验 |
| 3. 空闲 GPU 测两种形态 | 指定容器内两份 GPU/Graph 与 TP4 native collective 通过 |
| 4. 原 d.sh 服务 | 两份本轮服务报告、配方哈希、三次生成和自有进程清理 |
| 5. 记录已完成功能模块 | 本文各阶段实现表、当前提交与支持边界 |
| 6. 比较 HCU 逻辑 | 正式核心差异清单、迁移节点映射及明确差异/未验证范围 |
| 6. 迭代 Skill 与文档 | [经验索引](development-methodology.md)、开发 Skill 的状态/原生/数值/来源测试契约及操作文档 |

最终逐项检查保存在 `backend/requirements-completion-audit.json`，包含来源保护、正式输出、测试、服务、文档与未覆盖边界。上述结论限定于本轮固定输入和执行配置；公共 PyPI 发布、TP8、全模型精度和性能不在通过声明内。

## 2026-10-27 V3 方法库导览与统一任务入口实施记录

> **本文用途**：记录 2026-10-27 方法库导航、统一 Skill 和 README 的建设范围、验证方法与限制。此次修改仅涉及方法库，不修改外部插件或内嵌运行时代码。

### 2026-10-27 1. 需求与交付对应

| 需求 | 实现 | 使用入口 |
|---|---|---|
| 修复网页导航与界面使用问题 | 离线文档阅读器、内部文档及章节路由、全文检索与类型筛选、搜索空态、请求复制反馈、窄屏导航状态；区分方法库和外部运行时 | [architecture.html](../architecture.html) |
| 新增总 Skill 检索并管理现有方法 | 新增统一入口，先识别意图并检索当前目录，再选择专项 Skill；利用已有信息，必要时补问，并在输入齐备后继续已授权任务 | [hcu-plugin-assistant](../skills/hcu-plugin-assistant/SKILL.md) |
| 从使用者角度完善 README | 补全文件树、功能及场景、简化调用方式，加入开发、安装和排障案例，说明人工使用、路径配置、维护检查及运行时边界 | [根 README](../README.md) |

统一入口是 Agent 的任务执行指引，不是单独运行的调度服务。它通过读取专项 `SKILL.md` 衔接执行；无需先安装全部 Skill 或修改客户端全局配置。总数为 1 个统一入口和 7 个专项 Skill。

### 2026-10-27 2. 网页构建与维护

网页的布局和交互维护在 [源模板](../templates/architecture.html)，由 [build_guide.py](../tools/build_guide.py) 读取当前共享文本构建。输出只写入 `architecture.html`；运行 `--check` 时不写入，快照过期返回非零。模板与内容不手工双份维护。

页面内嵌文档、Skill、参考和方法工具的阅读内容及检索信息，不依赖外部字体、脚本、CDN 或浏览时的目录请求。操作指南之间的链接转换为阅读器路由，源文件另有明确入口。外部网页链接保留为用户主动访问的链接。相对路径不得越出方法库，代码与原文按文本转义，不能关闭内嵌数据脚本或执行原文中的 HTML。

阅读器支持当前资料使用的标题、链接、代码块、列表和表格；Mermaid 图表以可展开源码展示，图形查看可使用支持 Mermaid 的 Markdown 编辑器。它不声称实现完整的 GitHub Markdown 渲染器。

新增构建工具后，本库共有 10 个方法工具。[runner 配方](../.workflow/l0-gate.yml)、[CI 说明](ci.md) 和 [仓库约定](../AGENTS.md) 已同步快照维护规则。实际平台自动 CI 接入状态保持原有范围。

### 2026-10-27 3. 统一入口的任务选择与补问

统一入口先枚举实际存在的 `skills/*/SKILL.md`，读取文档索引，再按目标、症状、组件和版本词搜索。候选表是初筛提示；实际文件的描述与适用范围是判断依据。根据阶段选择必要输入，避免将所有任务字段一次交给使用者填写。

| 场景 | 要求的处理方式 |
|---|---|
| 只问“怎样开发新插件” | 提供初始化指南和最小案例，不因缺少机器路径而阻断方法咨询 |
| 要求实际创建工程，尚未提供写入位置 | 选择初始化 Skill，询问外部新目录；等待时仍可解释结构、检查已有官方输入 |
| 已提供路径、日志或版本 | 优先读取和核实，不重复索取可得到的信息 |
| pip 或入口发现异常 | 选择安装 Skill，先核对解释器、metadata 和模块来源 |
| 已加载后的 Graph、TP 或数值回归 | 选择诊断 Skill，检索相似经验并定位本次首次分叉 |
| 要求生成 embedded 并对照 HCU | 选择回溯 Skill；HCU 作为独立比较来源，不伪装为生成器的第三输入 |
| 一个请求包含多项交付 | 按真实依赖衔接，不强制顺序执行全部专项 Skill |
| 找不到匹配入口 | 说明候选与差异，询问期望产物，不编造 Skill 或擅自扩展范围 |

上述行为完成了指引和调用资料的一致性审阅。此次没有把这些案例当作已执行的外部工程创建、安装或模型服务任务；Skill 元信息校验也不证明任意 Agent 都能正确理解所有自然语言请求。

### 2026-10-27 4. 验证与证据边界

| 检查 | 本轮结果及范围 |
|---|---|
| L0 | 通过：18 个 Python 文件、8 个 Skill；结构、资源链接和方法库阶段边界 |
| 工具契约测试 | 53 项通过；新增网页构建用例覆盖离线内链、代码与数据转义、表格代码中的管道符、目录边界和只读快照检查 |
| Skill 元信息 | 8 个入口通过 `quick_validate.py`；新入口与对应 UI 提示一致 |
| 文档与网页内链 | 文件、标题锚点、生成阅读器路由及内嵌原文一致性检查通过；计数以证据 JSON 为准 |
| 网页交互 | 无网络 JSDOM 测试通过：任务与章节跳转、跨文档链接、历史返回、错误路由恢复、检索及筛选、复制成功和拒绝反馈、窄屏菜单与 Escape |
| 响应式检查 | 检查样式规则及 390 / 1440 宽度下的 DOM 菜单状态；未测量真实浏览器布局几何 |
| 实际浏览器与截图 | 未执行：内置浏览器安全策略拒绝 `file://` 页面打开；未换用其他浏览器或方式绕过拒绝 |
| 外部插件、GPU 与模型服务 | 本轮未执行，不引用历史运行结果充当本次验证 |

证据位于方法库 ignored 目录 `artifacts/v3-20261027/`：`local-validation.json`、`guide-qa.json`、文档检查结果、远端检查日志及最终需求验收报告。ignored 证据需另行保管，不随 clone 自动分发。

新导航、Skill、文档和工具已落入方法库；原有未提交改动按原状保留。本轮不会自动提交或推送现有工作区。

## 2026-10-28 运行时代码迁入与入口验证

## 2. 迁移与提交台账

| 日期 / 步骤 | 完成内容 | 提交记录 | 验证与边界 |
|---|---|---|---|
| 2026-10-28 / M01 运行时代码迁入 | 525 个来源文件；180 个 Python 文件仅改命名空间；239 个源 Python 文件 AST 对照；286 个原生资源/许可证按原字节保留 | [`4d82a16`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/4d82a1663a54c5d4075c5215acc2d91dcfc66750) | 本次静态迁入对照通过；不把旧 GPU/模型结果记为新包验证 |
| 2026-10-28 / M02 安装与启动入口 | 本仓打包定义、`hcu` 入口、显式 launcher、命名空间可选严格加载工具、阶段门禁及当前使用文档 | 同一代码提交 `4d82a16` | 65 项仓库测试、源码/wheel 各 363/363 项 Hook 和 3 个具名接口通过；完整服务未执行 |
| 2026-10-28 / M03 HCU 发现入口 | entry point 改为 `hcu = hcu_sglang:register`；版本 `0.3.2.dev7`；同步当前安装与开发说明 | [`443eb83`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/443eb83eb56d91ec5a86c1c53989419fb0077667) | 源码/wheel 各 363/363 项 Hook、3 个具名接口；旧 `hcu` 名称隔离通过 |

代码提交：`4d82a1663a54c5d4075c5215acc2d91dcfc66750`，标题 `feat: import HCU runtime as hcu_sglang with isolated packaging`，共 557 个变更文件。范围包括 526 个运行时包文件、根打包定义、迁入清单、当前使用文档与 Skill、结构门禁和运行验证工具/测试；未迁入输出仓的 docs、scripts 或 tests。其后提交 `c8ba45a` 仅回填该代码 SHA 并重建导览。

当前入口在 M03 改为 `hcu`；M01/M02 中的 `hcu` 与 dev6 验证结果保留历史含义。

当前代码来源是已存在实现的迁入，不是宣称整个 DAS fork 已完成转换。模型能力、Native、CP/EP/PD 的历史范围仍以对应来源记录为线索，本命名空间必须建立自己的产物与运行证据。

## 4. 实际验证记录

### M01/M02：首次迁入（dev6 / hcu）

验证日期：2026-10-28。实际环境为 Python 3.10.12 与既有 HCU/DTK 依赖容器；官方源码固定在上述 `baa9befa`。安装、构建和仓库测试使用源码所属用户；真实 Hook 加载使用容器原有的可访问设备的执行身份。没有修改设备权限或共享 site-packages。

| 检查 | 实际结果 | ignored 原始证据（相对仓库根） |
|---|---|---|
| 在线来源核验 | 525 个来源/目标 SHA256 一致；official、fork、来源 plugin 的 HEAD 与干净状态保持 | `artifacts/v3-20261028/provenance.json` |
| 结构及仓库测试 | L0：261 个 Python、8 个 Skill；65 项 unittest 通过 | `artifacts/v3-20261028/l0.log`、`tests.log` |
| 源码隔离安装与冷导入 | distribution `hcu-sglang-plugin==0.3.2.dev6`，唯一 `hcu = hcu_sglang:register`；冷导入未载入 Torch/AITER/旧包 | `artifacts/v3-20261028/source-install.log`、`source-cold-import.log` |
| 源码真实加载 | 363 项目标集合完全匹配且全部应用；3 个具名 resolution 接口；软错误与未应用列表为空；未导入旧包 | `artifacts/v3-20261028/source-hooks.json`、`source-hooks.log` |
| wheel 构建与完整性 | wheel 内 526 个包文件路径/字节与源码一致；未携带外部 docs/tools、旧包或全局 Python 启动文件 | `artifacts/v3-20261028/runtime-validation.json`、`build-wheel.log` |
| 隔离 wheel 安装与冷导入 | 从独立 target、仓库外 cwd 导入；唯一新入口；冷导入无 Torch/AITER/旧包 | `artifacts/v3-20261028/wheel-install.log`、`wheel-cold-import.log` |
| wheel 真实加载 | 未将仓库根加入 PYTHONPATH；实际导入来自 wheel target；363/363 项和 3 个具名接口通过，无加载软错误或旧包导入 | `artifacts/v3-20261028/wheel-hooks.json`、`wheel-hooks.log` |
| 显式 CLI 参数入口 | 源码和 wheel 的 `python3 -m hcu_sglang --help` 均 exit 0；只证明参数入口可用 | `artifacts/v3-20261028/source-cli.log`、`wheel-cli.log` |
| 文档与 Skill | 8 个 Skill 元信息、导览快照及链接/锚点校验通过；迁移计划保持原三方快照并链接当前记录 | `artifacts/v3-20261028/local-validation.json`、`guide-qa.json` |

本次 wheel 为 `hcu_sglang_plugin-0.3.2.dev6-py3-none-any.whl`，SHA256：`dc992c7a420148204cee1e0a9274ca590ea555189fb9bc58d74b606e1f720e8a`。它包含 Python 与原生源码资源，实际运行仍依赖兼容 vendor 库及工具链；没有发布公共 PyPI 版本。

复现构建和隔离 wheel 安装（`ARTIFACTS` 为本次独立可写目录）：

```bash
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir "$ARTIFACTS/wheels" .
python3 -m pip install --no-deps --no-build-isolation --target "$ARTIFACTS/wheel-site" \
  "$ARTIFACTS/wheels/hcu_sglang_plugin-0.3.2.dev6-py3-none-any.whl"
```

wheel 加载进程在仓库外执行，只将固定官方 `python/` 与本次 `wheel-site/` 加入 PYTHONPATH，并将严格验证工具的 `--plugin-root` 指向 `wheel-site/`。目标合同仍使用本仓迁入清单。

**环境排查记录**：首次用无容器 passwd 记录的普通 UID 验证时，Torch 默认缓存落到不可写目录，AITER 随后尝试写不可用的用户 JIT 目录。配置本次任务的 `XDG_CACHE_HOME`、`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`、`AITER_JIT_DIR` 后缓存问题消除；首次 AITER 原生依赖构建完成。随后普通 UID 报设备不可用，而容器原有执行身份可见 8 张设备，真实加载因此使用后者。失败尝试日志保存在 `artifacts/v3-20261028/failed-attempts/`，不能将编译耗时或 loader 捕获异常后的退出视为加载通过。

**未执行范围**：新命名空间的模型生成、GPU 数值/Graph/多卡服务及 embedded 产物未在本轮复验。真实 Hook 加载使用了设备环境，但不能代替这些验收；旧命名空间的历史结果仅作为后续回归线索。

### M03：HCU 插件发现入口（dev7 / hcu）

代码提交：`443eb83eb56d91ec5a86c1c53989419fb0077667`（`fix: discover HCU plugin through the hcu entry point`，13 个文件）。

将根打包元数据入口改为 `hcu = hcu_sglang:register`，开发版本更新为 `0.3.2.dev7`。运行时实现及 363 项目标合同不变；同步当前安装说明、开发指南和 Skill，保留旧产物的历史入口。实际执行环境沿用固定官方 `baa9befa` 与 Python 3.10.12 / HCU 容器。源码与 wheel 均从新的隔离 target 安装；构建/安装使用源码所属用户，设备加载使用容器原有执行身份。

| 检查 | 本次结果 | ignored 证据（目录 `artifacts/hcu-entrypoint-20261028/`） |
|---|---|---|
| 安装 metadata 与冷导入 | dev7 仅提供 `hcu → hcu_sglang:register`；源码及 wheel 冷导入均无 Torch/AITER/旧包 | `source-cold-import.log`、`wheel-cold-import.log` |
| `SGLANG_PLUGINS=hcu` 源码加载 | 363/363 目标全部应用、3 个具名接口，无加载软错误或旧包导入 | `source-hooks.json`、`source-hooks.log` |
| `SGLANG_PLUGINS=hcu` wheel 加载 | 从仓库外 cwd、隔离 wheel target 加载，363/363 和 3 个具名接口通过 | `wheel-hooks.json`、`wheel-hooks.log` |
| 旧名称隔离 | 新进程设置 `SGLANG_PLUGINS=hcu`，0 项 Hook，新包未导入 | `old-name-isolation.log` |
| wheel 完整性 | 526 个包文件路径及字节与源码一致，无旧包或全局启动文件 | `runtime-validation.json` |
| 仓库门禁 | L0 与 65 项 unittest 通过；两个更新的 Skill、链接/锚点和导览快照校验通过 | `l0.log`、`tests.log`、`guide-qa.json` |

本次 wheel：`hcu_sglang_plugin-0.3.2.dev7-py3-none-any.whl`，SHA256：`5d61be21697b03c9c5e21cf26c7ec74a161357f7af15e93b0bf2c2e911740ecd`。新版本更新安装元数据后采用 `export SGLANG_PLUGINS=hcu`；dev6 的 `hcu` 结果不作为 dev7 的实际加载证据。本轮未执行模型服务、数值或 Graph 复验。

构建首次使用设备执行身份时被 Git 所有权校验拒绝，已改用源码所属用户完成构建；保留失败日志，未修改全局 Git 信任配置。

## 6. 2026-10-29 分阶段继续迁移

输入保持 official `baa9befa`、DAS `15c7204b`；本轮起点为本仓 `405b8dc`。输入只读，不推送远程。原固定盘点含 3,010 个非 same 路径，其中运行时范围差异 1,614 个；本轮执行台账将全部非 same 路径纳入，未审核项保持 pending，不能把 450 条历史映射线索当作当前行为覆盖。

| 阶段 | 本轮实施 | 退出条件与当前边界 |
|---|---|---|
| M0 基线与迁移工具 | 当前/旧包识别；清单和处置目标发现；抽取/签名 CLI；v4 embedded；全差异执行台账 | 离线契约与新包隔离加载夹具；当前类 profile 仍可能拒绝漂移 |
| M1 基础与量化 | 延续既有实现，逐项处置尚未完成 | shape/dtype/依赖/回退行为验证后更新人工决定 |
| M2 Attention/KV/模型 | 延续既有实现，逐项处置尚未完成 | 构造、状态与消费者、数值、Graph 和指定模型分别验收 |
| M3 通信/PD/扩展 | 新增 HCU StandardDispatcher 哨兵映射修复 | 公共 dispatcher CPU 回归；真实 EP 多进程、GPU Graph 与模型仍待验收 |
| M4 产物与交付 | 当前包 embedded 生成与核验能力 | 新包实际源码/wheel/embedded 严格加载分别记录；不代表全 DAS 等价 |

### StandardDispatcher 来源与行为

来源 `python/sglang/srt/layers/moe/token_dispatcher/standard.py` 的 helper 与 dispatch 方法迁入 `hcu_sglang/layers/moe/standard_dispatcher.py`。只把官方模块全局名显式限定为 `S.*`，保留来源注释；新增 HCU AROUND 门控与顶层注册。非 HCU 交还传入 original，保留前序包装。没有从 DAS 动态导入。

原始反例：映射 `[0, 1, -1, -1, 2]` 下，输入 `-1` 被 Python 负索引映射为共享专家 `2`；修复后负数与超出全局表的 ID 均保持 `-1`。正常 routed/shared IDs、DP pad 的后置屏蔽、空 batch 和跳过映射分别验证。当前完整目标集合为 [当前合同](hcu-sglang-development.md#历史-json-快照)（`hcu-sglang-targets.json`）；原迁入 manifest 的 363 项保持历史快照。

运行测试要求固定 official python 与本仓在 PYTHONPATH，使用已有兼容 HCU/Torch 容器，从仓库根执行：

```bash
python3 tests/runtime/standard_dispatcher.py
python3 tools/build_migration_execution.py --check
python3 tools/build_guide.py --check
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

本次原始证据在 `artifacts/v3-20261029/`。逐路径台账的 implemented 只表示对应文件范围的代码接入，不是全领域完成。GLM5-next、Muse Spark、各 KV/Graph/native 分支等仍需逐项迁移与完整验收；本轮不宣称全量 DAS 转换完成。

### 本轮本地提交与实际检查

本地工具提交 `af5c635`：当前包识别、清单/处置/抽取/签名和 v4 内嵌工具，含旧包兼容与新包隔离夹具。运行时提交 `42fe693`：StandardDispatcher 修复、当前 364 项 Hook 合同、公共 dispatcher 回归和全差异执行台账。提交均未推送。

| 检查 | 实际结果 | 本轮证据 |
|---|---|---|
| L0 / 离线契约 | 267 个 Python、8 个 Skill；70 项 unittest 通过 | `l0.log`、`tests.log` |
| 真实公共 dispatcher | 4 项 CPU 张量测试通过，含原行为反例、EP 映射与共享专家、DP pad、空 batch 和跳过映射 | `dispatcher.log` |
| 源码严格加载 | 364/364 项应用、3 个具名 seam、无加载软错误与旧包导入；集合与当前合同一致 | `source-hooks.json`、`source-hooks.log` |
| 隔离 wheel 加载 | 仓库外 cwd、独立 wheel-site；364/364 与 3 个 seam 通过 | `wheel-hooks.json`、`wheel-hooks.log` |
| wheel 构建 | 本地 dev7 产物 SHA256 `d4ed63db35ee5ad9ef42a058e1bde39205bed60dc254c3b314d002c5abec3fea`，与历史同版本产物按 SHA 区分 | `wheel-build.log`、`artifact-checks.json` |
| Skill 与网页 | 3 个修改 Skill 的 validator 通过；重建导览、实际桌面和 390px 窄屏阅读器与导航开关检查 | 本次浏览器观察；快照门禁日志 |

测试环境为已有 HCU/DTK 容器、Python 3.10.12、Torch 2.11.0；公共 dispatcher 测试使用 CPU 张量并只控制设备/并行环境及昂贵通信条件，保留真实官方构造和 dispatch。节点宿主 Python 3.6、本机 Python 3.9 均不作为本项目验证解释器。首次未安装 AITER bootstrap 的测试导入失败，加入与 launcher 相同的兼容启动后通过；失败日志不能当作通过证据。


### 本轮实际 embedded 复验

从干净本地提交 `08b027e3228ace34537a5cf4a3e334f00ef9b747` 与固定官方 `baa9befa` 生成独立 `embedded-hcu` 源码树，迁入 527 个包文件；532 项 manifest 哈希全部通过。禁用所有外部 entry point、仓库外 cwd，仅加入产物 `python/` 后，364/364 项 Hook 和 3 个具名 seam 均通过，官方与插件模块均来自该产物，无旧包导入。证据为 `embedded-build.log`、`embedded-hashes.json`、`embedded-hooks.json` 和 `embedded-hooks.log`。产物位于本任务新建的 sibling `tmp/hcu-embedded-v3-20261029/`，不是原始 DAS fork 的逐文件复刻，也未执行模型/Graph/多进程 EP。

当前来源的方法同步预检明确失败在 `DeepseekV4DecoderLayer`：DAS 当前提供 Base/Cuda/Platform 拆分类。工具退出 2、不写任何输出；本次通过包名适配不等于该旧 profile 可直接执行。下一步须审查现有 `current_v4` 接入与平台类继承、状态和消费者，再决定删除旧抽取目标或补充精确 profile，不能将类名机械替换后覆盖既有适配。原始诊断见 `method-preflight.log`。

本轮结束时 official/DAS 两输入 HEAD 与起点相同、工作区干净。全量迁移总任务保持 in-progress；台账剩余 3,009 个未审核路径不等于 3,009 项缺失实现。

## 7. 2026-10-29 模型服务测试中的 spawn 启动修复

本轮以 cookbook 的 SGLang 0.5.18 单机八卡配方为参考，在已有 HCU 容器中运行本仓插件与固定官方 0.5.22 源码；它不是在容器预装 DAS 0.5.18 上加载插件的兼容声明。用户选定代表模型 Qwen3、MiniMax、GLM/DeepSeek，原始请求、响应、配方和运行记录保存在用户指定测试工作目录。

首次 Qwen3-235B-A22B-Instruct-2507 TP8 服务在加载权重前退出：父进程虽有 AITER shim，`runpy` 将主模块身份换成官方 launcher，spawn worker 在反序列化 scheduler 时先导入 quantization，触发旧 AITER 缺少可选 `gemm.fused` API。此前仅父进程的 Hook apply 不能覆盖此路径。

新增 `hcu_sglang.launch_server` 作为 spawn 可重入入口，在模块导入时仅安装惰性 shim；包 CLI/console 入口将主模块设为该入口，再以不替换主模块身份的方式执行官方 launcher。worker 在解析 scheduler 目标前先恢复相同 bootstrap，不安装全局 sitecustomize/usercustomize，不伪造真实 Quark kernel；真调用缺失 kernel 仍抛出原有 NotImplementedError。

版本更新为 `0.3.2.dev8`。真实 spawn 隔离夹具在旧 dev7 上失败、在修复后通过；71 项 unittest 和 L0（268 个 Python、8 个 Skill）通过。dev8 wheel 含 528 个包文件，逐路径/字节与源码相同，SHA256 `457e064d8d7a57413038ca453330604ef0ede781f79be33f6221eadb8d32f5b5`。同一 Qwen3 TP8 配方复跑已越过原失败点并进入八 rank 权重加载；最终生成结果在后续记录中补充。

测试工作目录内证据：`logs/spawn-regression-original.log`、`logs/spawn-regression-fixed.log`、`logs/repository-tests.log`、`logs/l0.log`、`artifact-dev8.json` 及 `runs/qwen3-235b-instruct-tp8-attempt2/`、`runs/qwen3-235b-instruct-tp8-attempt4/`。前者保留原始启动失败；后者为修复后的独立运行。

## 8. 2026-10-29 量化 CLI 与 forward-context KV 归属修复

GLM-5 INT4 W4A8 的 cookbook 配方首先在参数解析阶段被拒绝：运行时量化 registry 已有 `slimquant_w4a8` 和 `slimquant_w4a8_marlin`，CLI choices 却只加入 `w4a8_int8`。`server_args.py` 现在从同一 registry 常量导出全部三个名称；保持原有选择、幂等性和量化实现的惰性加载。真实 CLI help 已包含这些名称，离线回归检查实际 argparse 接受与不提前导入 Torch/量化实现。

随后用空闲四卡的小型 Qwen3 诊断完整 forward，确认 Torch/HIP 当前设备与张量设备一致，并在 Graph warmup 复现 `ForwardBatch` 不再拥有 `token_to_kv_pool` 的异常。固定官方在 `model_executor.forward_context.get_token_to_kv_pool()` 从当前 backend 解析 KV 池；共享 fused rotary、Qwen2 融合路径和模型公共融合判定均改用此接口，不补写旧字段、不缓存单一全局池。

真实 forward-context 回归在旧 dev9 wheel 上失败、在修复源码上通过：无旧 pool 字段的 batch，在外层/内层/恢复后的 context 中分别写入对应 KV 张量并保留融合 store 标记。昂贵的融合数学 kernel 被替换成可观察的 CPU 写入，官方 context 与实际池选择未 mock。证据为测试工作目录 `logs/kv-context-original.log`、`logs/kv-context-fixed.log` 与 `tests/runtime/fused_kv_context.py`。

当前开发版本为 `0.3.2.dev10`，wheel SHA256 `239e573e98ed5b159d84534a1bccf4ce18d06ef0e550c67bd05034281da71146`。CLI 修复后的 72 项离线 unittest 通过；模型结果另行记录，不将解析或 CPU 写入测试当作整模型精度通过。

## 9. 2026-10-29 四卡代表模型复验与融合门控修复

用户在 GPU 0–3 被其他服务占用后授权先验证四卡服务；使用物理 GPU 4–7、独立 ROCR 可见映射，保留其他服务。Qwen3-30B-A3B-Instruct-2507 BF16 TP4 在 dev10 中完成权重加载，越过 init_cublas 后于 Graph warmup 的模型公共 helper 导入处失败。原始证据为测试工作目录 runs/qwen3-30b-instruct-tp4-attempt1/；不是精度通过。

按失败路径对照固定 DAS models/utils.py，现有迁入 helper 使用已删除的 is_prefill_context_parallel_enabled，且通配导入不会导入私有 _is_cuda/_is_hip 标志。当前改为显式导入和官方 layers.cp.utils.is_cp_active(forward_batch)，保留 DAS 对 HCU 的 CUDA/BF16/非 SWA 门控、当前 CP 与 dcp_kv_mask 限制，其他 HIP 平台沿当前官方门控；池仍由真实 forward context 提供。不运行 DAS 对照服务，不动态导入 DAS 实现。

开发版本 dev11；新增 tests/runtime/model_utils_fusion.py 验证真实 helper 导入、forward-context 池、当前 CP 分支、DCP mask、SWA/dtype 和非 HCU 路径。修复前反例、修复后检查、wheel 与模型复验结果见下文；CPU 和打包检查不代表模型通过。

实际检查：dev10 wheel 在 logs/model-utils-original-attempt2.log 复现已删除 API 的 ImportError；修复后 logs/model-utils-fixed.log 的 3 项真实 helper/forward-context CPU 回归通过。72 项离线 unittest 与 L0 通过（logs/repository-tests-dev11.log、logs/l0-dev11.log）。dev11 wheel SHA256 e7badb052105ea5f06be25f92249427bb2e6fdcfd8931615730352ce4b5e07be，528 个包文件与源码和隔离安装目录逐字节一致，见 artifact-dev11.json。

相同四卡 Qwen3 配方的 dev11 attempt2 在权重和 16K KV 缓存加载后，仍于未修改的 init_cublas 16x16 FP16 矩阵乘法触发 rocBLAS Tensile kernel 缺失；15:53:57 终止，已记录自有进程清理为空、端口无监听。此轮没有请求或模型精度通过；Graph 原失败点的完整模型复验仍 pending。固定 DAS 与官方 attention_backend_setup 文件相同、init_cublas 调用及辅助函数相同，未据此认定环境或插件为最终根因。

## 10. 2026-10-29 按触发路径对齐 DAS 的 AITER 通信逻辑

用户明确要求插件与 DAS 的实现路径对应，不以关闭优化路径代替修复。本轮新增 distributed_aiter.py，迁入固定 DAS custom_all_reduce.py 的三个函数：AITER 注册选项、工作空间大小与完整 dispatcher；现有 HCU dispatch Hook 懒加载该实现，非 HCU 保留传入 original。HCU auto 不再按旧 SGLANG_USE_AITER_AR 默认关闭 AITER，而沿 DAS 的确定性判断与 AITER 导入选择。显式后端参数在 helper 层保留；官方 CLI/GroupCoordinator 的新增后端字段及 strict-Fabric 组合尚未在此轮移植，不宣称该组合已支持。

迁入保留 GraphSafeAiterCustomAllreduce：Graph warmup 运行真实 all-reduce/fused AR RMS、传递 live residual，capture 与 piecewise 路径交还 AITER 父类；IPC 注册受 memory-saver 及 DeepSeek-V3/GLM 特定拓扑约束，Fabric/auto 使用 copy-in。保留 DAS 的环境值校验、显式 aiter 与确定性模式拒绝、auto 下仅 ImportError 回退；不吞 kernel 错误。

来源文件 SHA256 为 2fe4be4c81564948b96d283e66f326f6a37b46ccc13b29fde051c581eb7834f9。只做模块全局名限定、HCU 检测入口及相对导入适配；还原这些适配后三个函数的 AST 与 DAS 完全一致。生产运行不动态导入 DAS。静态核对、修复前后 CPU 回归、wheel 和服务结果分别记录；当前准备的开发版本 dev12 不代表模型或 GPU collective 已通过。

实际执行：dev11 在 logs/aiter-align-original.log 的默认 HCU 后端反例失败；修复后 logs/aiter-align-fixed.log 的 6 项针对性 CPU 回归、72 项仓库 unittest 和 L0（272 个 Python、8 个 Skill）通过。dev12 wheel SHA256 30fbb912c349aed615c8febc8631f4e765b873cec7be40df9e6046162cf8c758，529 个包文件与源码、隔离安装逐字节一致。Qwen3 四卡 attempt4 的四个 rank 均记录 AITER IPC、copy-in，越过 BLAS 初始化后在 Graph 暴露 fused_topk 旧签名不接受 num_fused_shared_experts；没有生成请求或模型精度通过。此过程同时经历节点重启及主机内存恢复，不能据此将先前 rocBLAS 错误单独归因为通信选择。

## 11. 2026-10-29 MoE top-k 当前调用契约与 DAS 分支对齐

Qwen3 TP4 dev12 在 Graph warmup 经真实 select_experts 调用，因旧 REPLACE 的 fused_topk 缺少 num_fused_shared_experts 参数退出。固定 DAS 与官方 fused_topk 函数去除唯一 HCU LightOp 分支后 AST 完全一致；两输入文件哈希已与本次来源核对。插件改为 AROUND，完整保留官方的 11 项参数，非 HCU 及其他分支委托传入 original，保留已有包装。

仅在 HCU/softmax/原有 SGLANG_USE_FUSED_TOPK_SOFTMAX 开关启用、且不优先命中 AITER 分支时执行 DAS 的 LightOp topk_softmax。AITER 优先级、sigmoid、路由缩放、共享专家数、packed 输出与非 padding token 数交还固定官方的相同实现，不吞 kernel 错误，不用忽略新参数规避启动异常。版本 dev13；公共 HookRegistry 分发和 CPU 消费者回归、wheel、模型复验分别记录。

实际回归：dev12 wheel 在 logs/topk-original.log 通过真实 HookRegistry 包装的公共入口复现同一个 num_fused_shared_experts TypeError；修复后 logs/topk-fixed.log 的 6 项公共 Hook/CPU 消费者测试通过。72 项仓库 unittest、L0（273 个 Python、8 个 Skill）通过。dev13 wheel SHA256 8f7f9a0d69e6e96c2d180905f11932a4d9f799022c64d59d3d8491fda2b98b6e，529 个包文件与源码、独立安装一致。来源逻辑核对见测试目录 source-diff/topk-source-alignment.json。

模型复验：runs/qwen3-30b-instruct-tp4-aiter-attempt5/ 使用物理 GPU 4–7、官方 0.5.22、dev13 wheel、DAS 对应 AITER IPC 路径及正常 Graph，完成 batch 1/2/4/8 捕获。六个确定性请求严格 6/6 通过：323、北京、Paris、4、10、重复 323，全部 finish_reason=stop。17:31:25 完成，cleanup_remaining_pids 为空、端口无监听。仅代表该四卡短请求 profile，不外推为 235B 八卡、数据集精度、吞吐或全部 DAS 等价。

## 12. 2026-10-29 GLM/DeepSeek Decoder 退役旧替换并恢复当前 stage

GLM-5 INT4 TP4 在加载权重前被旧 Decoder 构造器拒绝 skip_rope。固定官方与 DAS 的 DeepseekV2DecoderLayer 全部方法已相同；当前实现使用 attn_boundary/ffn_boundary，旧插件仍使用 layer_communicator 且 forward 多出 residual 参数。按实际调用/消费边界成对退役该类的 __init__/forward Hook 和两份不再使用的旧实现，使用当前官方与 DAS 相同的真实类，不复制旧函数后仅添加参数。

当前 Hook 合同去掉这两个目标，变为 362 项；迁入 manifest 及历史 364 项验证保留原义。新增 tests/runtime/deepseek_decoder.py 经真实注册入口验证当前构造、skip_rope/build_stages、NextN NoPE、归约 owner 及 native forward 对 stage 字段的消费。版本 dev14；实际 CPU、wheel 和模型结果在执行后补充。

## 13. 2026-10-29 当前 DeepSeek V2 调用契约与 DAS 数据流同步

GLM-5 INT4 TP4 dev14 仍在加载权重前失败：旧 Attention 构造器给当前 ColumnParallelLinear 传入已删除的 tp_rank。逐项审查后，本轮从固定 DAS 同步现有 24 个注册方法，覆盖 MLP、MoE 与 MLA/MHA/ROCm/CPU attention；保留当前 parallel_group、skip_rope、输入量化载体、非 padding token、共享专家与归约 owner 的完整参数和消费路径。只限定实际 provider 的全局名、显式 super owner 和插件依赖导入；移除文档字符串缩进影响后，24 个方法的规范化 AST 均与 DAS 相同。旧全量同步 profile 仍因 V4 类拆分预检失败，本次没有绕过预检运行它。

DAS 新增但固定官方没有的 fused_rms_quant、GLM5-next 模型类型谓词与 MoE graph 输出桥接放入插件命名空间。运行不导入 DAS 源码。W4A8 线性层同步 supports_prequantized_input 和显式 INT8/FP32 输入检查，显式 q/scale 不再依赖无关 RMS 开关；保持原 GEMM 后端。该谓词不代表完整 GLM5-next attention 后端迁移已完成。

同时核实并修正共享 Qwen/Bailing fused rotary 门控：沿 DAS 的 SGLANG_USE_FUSED_RMS_ROTARY 默认关闭语义，Qwen3-MoE 保留原生融合分支优先级。已有 dev13 Qwen 6/6 属于旧版本结果；当前版本必须另行复验，不能将旧请求记作新版本通过。

开发版本 dev15。tests/runtime/current_v2_dataflow.py 检查量化载体经真实 prepare 到线性层的传递、真实 CPU 数学消费、非法载体拒绝、graph 共享输出所有权与 Qwen 分支门控；仅替换昂贵外部 kernel。实际 CPU/仓库门禁、隔离 wheel 与 GLM/DeepSeek 服务结果记录于测试目录，不以静态同步证明整模型精度。

实际检查：24 项规范化方法 AST 核对通过；5 项当前 V2/W4A8/graph/Qwen CPU 回归、72 项仓库 unittest 与 L0（276 个 Python、8 个 Skill）通过，证据为 current-v2-ast.json、logs/current-v2-dataflow-dev15.log、logs/repository-tests-dev15.log、logs/l0-dev15.log。dev14 GLM attempt2 保留原 ColumnParallelLinear tp_rank 异常；新模型启动与请求结果另记。MiniMax-M2.5 BF16 TP4 dev13 已完成 6/6 严格请求检查，属于该版本四卡短请求结果。

## 14. 2026-10-29 GLM EAGLE NextN 当前 stage 同步

GLM-5 INT4 TP4 dev15 四个 rank 已加载目标模型，在 EAGLE 草稿初始化触发旧 NextN 的 is_dsa_enable_prefill_cp NameError；原实现同时保留已删除的 Decoder CP 参数、residual 参数和三值返回。按固定 DAS 同步现有两项 NextN Hook，使用 layer_stack、skip_rope、当前 residual_batch 边界与两值 Decoder 返回，保持 IndexTopKShareState 更新/发布顺序和 SBO stream 选择，不关闭 EAGLE。

两项规范化方法 AST 与固定 DAS 相同；缺失的 legacy CP helper 复用插件现有 HCU backend provider，CP-v2 可用门控按 DAS HCU 默认关闭语义读取环境值，避免读取官方已废弃的 EnvBool registry 名称。通用 CP/GLM5-next 全路径尚未验证，不能从本轮 attn_cp_size=1 的配方外推。生产运行不导入 DAS 源码。版本 dev16，tests/runtime/nextn_current.py 通过真实公共 Hook 调用核对当前构造器契约、SBO stream 和 forward 的 residual/返回消费顺序；实际门禁、wheel 和模型结果另记。

## 15. 2026-10-29 DSA KV 池与 DSpark 解析路径同步

GLM dev16 目标/草稿加载越过 NextN 后，在 KV 池构造因 index_kpool 参数失败；DeepSeek-V4 dev16 首轮在解析阶段因原始 device=None 被旧包装器读取而失败。本轮分别沿触发边界对照固定 DAS，不运行 DAS 对照服务。

DSA 池迁入当前构造与缓存方法命名空间，34 项方法的规范化 AST 与 DAS 相同。现有 AROUND 转发所有参数，在 HCU 消费完整 index_kpool/compress、tail、request、DCP 和 indexer_layer_ids；非 HCU 委托原函数。缓存 facade 同步紧凑全局层映射，保留当前 FP8/BF16/INT8 模式选择、真实缓存读写/搬移和共享 workspace 所有权。INT8 helpers 保留 gfx936 opt-in 限制，读取环境值以适配官方未提供的新 EnvBool；本次 gfx938 FP8 配方不构成 gfx936 INT8 的整模型验证。DAS 的 _should_allocate_index_layer 在当前来源误带 property，但实际消费者调用该方法；保留官方同一函数体的 callable 绑定。facade 补齐通过现有 GetKAndS 消费的连续 K/scale 接口，避免当前 DAS 方法调用不存在的 facade 名称。新增 host_pool_decls Hook，当前合同变为 363 项，历史 362/364 项结果不改写。

DSpark 同步 DAS 的 _handle_dspark 与 PD/CP 资格判断，读取真实 resolving_view，沿 declare_resolution 写入决策，不再临时改写原始 device。保留 HCU DeepEP target/draft runner、静态 verify 与 CP/PD 限制。其他平台沿传入 original；运行不导入 DAS 源码。完整 PD/CP、INT8 GPU kernel 与长请求仍未验证。

开发版本 dev17。5 项 tests/runtime/dsa_dspark_current.py 通过真实公共 Hook 检查构造参数、page/压缩 tail、紧凑层读写、key/scale 搬移、非 HCU 参数保留和真实解析 overlay；仅替换外部 MLA 分配与昂贵 kernel/配置文件读取。证据为 current-dsa-audit.json、current-dspark-audit.json、logs/dsa-dspark-current-dev17.log。仓库门禁、独立 wheel 与模型复验另记。

已有 dev16 四卡复验：Qwen3-30B-A3B-Instruct-2507 attempt6、MiniMax-M2.5 BF16 attempt2 各 6/6 严格通过，Graph 保持开启，均清理自有进程且端口无监听；当前 GLM/DeepSeek 尚未通过，不将启动进展记作精度结果。

## 16. 2026-10-29 DSA HCU FlashMLA metadata 与 Graph 路径同步

GLM dev17 已加载目标/草稿并分配 KV，在 Graph 初始化误调用 CUDA sgl_kernel.flashmla_ops。固定 DAS 使用 flash_mla.flash_mla_interface 和对象 scheduler metadata。本轮迁入当前完整 DSA backend/multistep 类及 metadata producer/manager/indexer 依赖，HCU 注册两个类替换；其他平台沿原类。六个选择出的定义还原命名空间导入后 AST 与 DAS 相同，实际 package/probe 记录在测试目录。相同的 kpool mixin 与其他依赖复用固定官方，不动态导入 DAS。

保留 HCU metadata wrapper 的创建、刷新、Graph 消费、MLP-sync 实际行裁剪和输出补齐；不将 backend 对象塞入旧 CUDA tensor metadata 结构。补齐两个 EnvBoolWithAlias 字段，类型、默认 false 和旧名称沿 DAS；保留原始用户输入与环境兼容。CP helper 复用现有 HCU provider 和原生 kernel；本轮 TP4 非 CP 不外推 CP/PD 验证。

版本 dev18，当前 Hook 合同变为 365 项。tests/runtime/dsa_flashmla_current.py 的两项 CPU 回归通过：真实 producer 到 attention 消费边界传递新 scheduler 对象/num_splits，真实补齐行还原，HCU 保留 fresh source 对象而 CUDA 保留 tensor 地址；仅替换外部 FlashMLA kernel。证据 logs/dsa-flashmla-current-dev18.log、current-dsa-backend-audit.json；实际 wheel 与服务复验另记。DeepSeek-V4 dev17 尚在旧模型 get_pp_group 构造路径失败，Base/Cuda/Platform 拆分的迁移继续处理中。

## 17. 2026-10-29 DSA backend 实际 entry point 注册修复

dev18 wheel 的 542 个文件通过字节验证，但完整 Hook 合同检查发现两个 DSA 类未注册：新增调用仅放入旧 layers.attention.register_hooks，实际入口 hcu_sglang.register_optimizations 使用显式模块列表。修正在实际入口导入与调用 current_dsa；版本 dev19。保留 dev18 的失败合同证据，不将其 363 项已应用结果称为 365 项通过，也不将 dev18 GLM 复跑当作新后端验证。下一轮先确认完整 wheel 合同通过，再启动服务。

## 18. 2026-10-29 DSA TopK 当前依赖与 paged v2 门控同步

dev19 完整 wheel 的 365/365 Hook 与 3 个 seam 均通过；GLM 实际 Graph 已越过 CUDA FlashMLA 扩展错误，在当前 metadata producer 调用旧 DSATopKBackend.should_use_topk_v2_paged 时失败。同步固定 DAS TopK dependency 模块并让 backend/indexer 使用同一枚举与方法，避免新旧 consumer 混合；补齐 SGLANG_HCU_TOPK_V2_PAGED EnvBool 默认 false，沿 DAS 对 HCU paged v2 的明确 opt-in，保留 ragged v2 的 layout/dtype 条件。版本 dev20，完整服务复验继续；没有请求或模型精度通过。

## 19. 2026-10-29 DSA indexer HCU Hadamard 路径同步

GLM dev20 已越过 FlashMLA metadata 与 TopK 接口，实际 Graph warmup 的 Indexer forward 在 native HIP 旋转导入 fast_hadamard_transform 时失败。固定 DAS 对 HCU 默认使用缓存 dense Hadamard 矩阵及 F.linear，显式 SGLANG_USE_FAST_HADAMARD_TRANSFORM 才选择扩展。本轮迁入该缓存、数学消费和完整 rotate_activation(apply_scale=True) 函数，保留 power-of-two 校验与归一化开关；不以其他旋转或随机矩阵替代。显式 fast 扩展开关缺包仍正常报错。版本 dev21，当前合同 366 项；真实 CPU/GPU 数学与完整服务复验另记。

## 20. 2026-10-30 DSA indexer 当前 HCU producer/consumer 同步

在已触发的 Indexer 路径继续核对，native HIP logits 分支仍断言 page_size=1，而 DAS/HCU 当前池使用 page_size=64。本轮不启动这个已知不匹配的 dev21 全模型组合；dev21 三项 CPU/GPU Hadamard 回归通过，真实 BF16 GPU 最大绝对误差 0.0171714，属于单算子证据。

迁入固定 DAS 的完整 Indexer 类及当前本地 helper，HCU 通过原 module 和 DeepseekV2 的实际构造别名注册同一实现，避免只替换类定义后原 caller 仍持有旧类。保留 HCU FP8/BF16/INT8 cache 读视图、LightOp dense/sparse logits 分发、与 mask TopK 配对的资格检查、head-gate scale 和 logits scratch 预算；既有相同依赖复用官方，Hadamard 复用已验证的 DAS 实现。sparse producer 只有匹配的 consumer/API/布局/门控全部满足才可运行，未命中沿 DAS dense producer。运行不动态导入 DAS。

版本 dev22，合同 368 项；隔离暂存导入已通过，CPU/GPU 消费者、来源 AST、wheel 与完整服务结果继续记录。gfx936 INT8 与 CP/PD 仍未执行整模型验证，不能从 gfx938 TP4 profile 外推。


本轮实际检查：Indexer 与两个 HCU logits helper 规范化 AST 均与固定 DAS 相同；4 项消费者回归通过，涵盖 owner/broadcast 读存储、page64 packed ABI、dense clean-logits 参数、sparse producer/consumer 配对门控以及真实 HCU head-gate Triton 数学。72 项仓库 unittest 通过。依赖字段审计补齐 DAS 保留的 SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB EnvFloat(None)，无全局预算覆盖；证据 current-dsa-indexer-audit.json、logs/dsa-indexer-current-dev22.log、logs/repository-tests-dev22.log。

## 21. 2026-10-30 HCU paged-MQA 后端选择依赖同步

GLM dev22 在资源检查停止两轮后，使用通过准入的物理 GPU 0/1/6/7 越过前述路径，进入新 Indexer 的 paged logits 分发；其残留原生枚举缺少 is_lightop。迁入 DAS 当前 DSAPagedMQALogitsBackend，并让 Indexer 使用该同一枚举。HCU auto/显式 lightop 选择 page64 LightOp；普通 HIP auto 仍选择 AITER；不支持的显式后端正常拒绝，不把 HCU 当普通 HIP。当前配方使用 auto，不宣称已验证所有显式 CLI choices。版本 dev23，合同仍 368 项，模型复验继续。

## 22. 2026-10-30 四卡 W4A8 Triton producer/consumer 路径同步

GLM dev23 已越过 attention/indexer，在真实 MoE warmup 触发已安装 LightOp Marlin 对 E=256,N=512,K=6144 的 unsupported size。当前 DAS auto 同样优先选择可导入的 LightOp，不能把它记作 plugin 独有精度错误；DAS 另提供显式 SGLANG_W4A8_TPMOE_BACKEND=triton，而旧插件仅 auto/lightop/aiter。本轮同步固定 DAS 的完整 SlimQuant Marlin 配置/后端实现和独立后端规范化/context helper，保留 auto 的原选择，四卡新配方明确选择 Triton。

权重生产端必须在载入前选择后端：Triton 保留原 packed INT4 权重，消费同一 channel scale，沿 DAS 的 chunk、block 适配、align、两次 GEMM、激活、路由/共享输出与缩放；不能把已 Marlin repack 的权重直接送入 Triton。版本 dev24。当前仅选择标准 TP4/EP1 路径；完整 DeepEP W4A8 与 DSpark override 尚未执行模型验证。CPU/GPU 数学、完整 wheel 和实际请求结果另记。


Triton GPU 数学夹具首先复现原生 align 的 HCU 320-thread launch 超过 256-thread bounds，随后非法访存。固定 DAS 的同一 align 明确记录并修复该问题。本轮补齐其 HCU Triton count/prefix/sort、padding sentinel 和无效专家映射，公共入口 AROUND 在其他平台委托原实现；当前合同 369 项。原故障证据 logs/w4a8-triton-current-dev24.log 保留，修复后数学测试另记，不把崩溃夹具称为通过。


实际数学检查：修复公共 HCU align 后，随机 INT4 两层 MoE/SiLU/路由/缩放的 GPU 输出与显式参考完全相同，最大绝对误差 0。另以 E=256、N=512、K=6144、top-k=8 的 GLM 四卡真实权重形状执行 warmup、Graph capture 与 replay，零输入/零权重结果正确；共 3 项检查通过。旧夹具的高/低 nibble 顺序错误已修正，失败日志保留；不是通过修改数学容差或算子绕过。证据 logs/w4a8-triton-current-dev24-graph.log 与 w4a8-source-alignment.json。完整模型请求仍待复验。

## 23. 2026-10-30 DeepSeek V4 当前 Base/Platform 执行接入

DeepSeek dev17 的旧模型替换仍调用已删除的 get_pp_group。按当前 DAS 拆分迁入 DecoderLayerBase/Platform 与 NonCudaDecoderMixin、模型/输出包装和所需 mHC/transport provider。模型、causal class 继续保留官方类身份，经原公共入口调用当前方法；显式 super owner 适配保证原类与派生类实例均只执行一次 nn.Module 初始化。当前 Model/causal 共 13 个实际不同方法沿 DAS 同步，旧 Decoder 三个方法替换退役为完整 Platform 类替换；注册同步更新所有已导入的 NextN 构造别名。

14 个选择定义还原 provider 导入、super owner/base 和未启用 mxscale 的惰性导入后 AST 与 DAS 相同。8 个 mHC 数据/状态类与官方 AST 完全相同，复用其类型身份，使 target、attention、DSpark 生产/消费一致；仅把 _mix_stats_impl/post 两项 HCU 差异函数接入同一当前实现。当前合同 376 项。固定 DAS 的 AITER in-tree helper 与版本化 CSV/JSON 原样打包，保留其 upstream-first 能力检查；当前容器缺少三项 gfx950 tuning API，沿 DAS 原逻辑记录 skip，不将其称为实际 tuning 验证。

HCU 不执行的 CUDA row-bucket replay/tail boundary 依赖改为首次实际使用时导入；当前 HCU maybe_create 按 DAS 返回 None。显式 dense-mxscale 仍需基线实际支持的接口，本轮保留默认 false 并仅在原 opt-in 分支导入该接口，不宣称 gfx950/CUDA tail/CP/PD 已迁移验证。没有为缺失算子制造成功结果或替代数学。

版本 dev25；真实原模型类入口、PP group/五 stream、Platform 状态、NextN 已导入别名和 mHC 状态身份的 CPU 回归、完整 wheel 与 GPU 模型复验分别记录。静态/构造通过不代表 DeepSeek 精度通过。证据 v4-execution-source-audit.json、v4-shared-mhc-types.json、v4-execution-env-specs.json 与 logs/v4-execution-current-dev25.log。


GLM dev24 已越过 Triton MoE，在 EAGLE draft forward 的 CP 资格 helper 读取旧 enable_prefill_context_parallel 字段时失败。固定 DAS parallel_hook 为 legacy CP 从 enable_prefill_cp 与当前 attention backend 派生该 alias；官方 bag 没有此字段。共享 HCU helper 现优先使用真实 legacy 字段，缺失时按同一派生条件读取当前 getter，排除 dsa/dsv4 的 MLA legacy alias；不伪造全局 bag 属性。NextN 公共 forward 回归改为真实经过 CP-disabled helper 分支，避免仅在 enable_cp_v2=True 时跳过原失败点。完整 CP/PD 仍未验证。

## 24. 2026-10-30 EAGLE 当前 KV plan 返回与 hidden-state 规则同步

GLM dev25 完成 target/draft Graph 后，在服务 warmup 的 EAGLE draft 调用触发旧包装器按二值解包当前三值返回。prepare_for_draft 已返回 (ForwardBatch, graph_flag, kv_loc_plan)，runner 位于第五个 positional 参数。包装器现只读取第一项补齐 DAS 的 mHC pre-norm 条件，原返回容器、KV plan 和其他项按身份完整返回；普通 GLM 不改 hidden-state 标志。

同时核对固定 DAS eagle_utils.eagle_prepare_for_verify 与官方均明确 return_hidden_states_before_norm=False；旧 HCU wrapper 强行改 True 已不对应当前路径，退役此一项 Hook 与函数。当前合同 375 项；不通过丢弃 plan、关闭 EAGLE 或伪造返回规避异常。版本 dev26，公共入口/调用者三值消费与完整服务复验另记。


DeepSeek dev25 已加载目标权重并进入 mHC Graph，旧 hcu_sglang.kernels.mhc 仍从官方 DSA utils 导入被移除的 round-robin CP 名称。改为同仓已接入的当前 HCU compat provider，保留当前拓扑与 round-robin/interleave 判定；实际模块导入和服务路径继续复验，不设置虚构 CP 状态。

## 25. 2026-10-30 GLM 精度失败与当前 DAS mHC 同步

GLM-5 INT4 dev26 attempt17 已完成 target/draft Graph、warmup 和 HTTP 请求，但算术、中文、英文三个响应均乱码且耗尽 1024 token，不能作为精度通过。主动终止余下请求，控制器清理完成，残余 PID 为空且端口关闭。实际 checkpoint 的三个 dense INT8 投影与 FP32 INT8 matmul/scale 参考误差均为 0；这只排查局部量化投影，未证明模型正确。

DeepSeek V4 dev26 attempt4 在 Graph 捕获进入旧 mHC 的 TileLang Allocate monkey patch 时失败。对照固定 DAS，当前 mHC 已删除该旧补丁并增加 HCU 小 batch split-K、AITER 物理维度压紧等路径。本次整体同步其 mHC 模块，只将 is_hcu 导入改为 plugin 检测；模块内所有顶层函数/类 AST 与固定 DAS 一致。没有通过关闭 Graph、mHC 或 AITER 避开问题。

真实 HCU 运行 8 token、4×7168 hidden 的 mHC pre，对照同源 _mhc_pre_torch：post 最大绝对误差 1.85e-6，comb 8.8e-7，BF16 activation 0.0078125；Graph 捕获及 replay 通过。初次探针误用了另一历史 mhc_pre_torch 辅助函数（其 post 额外乘 2），失败原日志保留，改用当前实际 dispatch 的 _mhc_pre_torch；生产公式未为测试修改。dev27 整体服务复验另记，GLM 精度仍待修复。Qwen3 dev26 attempt7 六请求再次通过；MiniMax dev26 attempt3 正在复验。

## 26. 2026-10-30 MLA INT8 反量化的 None block size 分支同步

继续排查 GLM 精度时发现旧 post_load_weights 通过 hasattr(weight_block_size) 判断 block quantization。当前 QuantizationConfig 即使 channel INT8 也显式带 weight_block_size=None，因此旧分支跳过 channel scale，w_kc/w_vc 留为原始 INT8。固定 DAS 已改为检查实际 block size 是否非 None；plugin 同步其整个 INT8 分支，分支 AST 与 DAS 一致。该修改保留 MLA 路径和 checkpoint scale，不改变注意力/量化后端配置。

新增以显式 None block size、非均匀 channel scale 的实际 post_load_weights 数值回归，检查拆分出的 w_kc/w_vc 是 BF16 且精确对应反量化值；旧 wheel 失败证据与修复后结果均单独保存。该局部问题修复不等同 GLM 整体精度通过，后续完整六请求复验另记。另补充 256 experts、高位 expert IDs 的真实 HCU W4A8 单算子数值参考通过，最大误差 0。

DeepSeek dev27 attempt5 已越过 mHC 并完成 decode Graph，在 verify Graph 遇到 HIP metadata factory 返回 None，而 HCU FlashMLA consumer 要求 FlashMLASchedMeta 对象。沿用当前 DAS HCU backend 的对象 metadata 路径，新增 HIP factory 的 HCU 定向 Hook，通过同源 get_flashmla_op 访问当前 flash_mla_interface provider；非 HCU 仍调用原 factory。合同增至 376 项。

## 27. 2026-10-30 GLM 六请求通过与 RowParallelLinear 融合能力接入

GLM dev28 attempt18-admission-retry2 使用原 EAGLE、FP8 KV、正常 Graph 和显式 DAS Triton W4A8，在四卡上严格 6/6 通过：323、北京、Paris、4、10、重复 323，均 finish_reason=stop。目标 verify batch 1–8、draft decode/extend Graph 已捕获；04:28:33 清理完成，无残余 PID、端口关闭。一次准入资源竞争只造成未启动的 pending_resources，随后控制器自动重试，没有同时启动两个模型。

DeepSeek dev28 attempt6 越过 HCU FlashMLA metadata 后，在共享 MLP 查询 RowParallelLinear.supports_fused_silu_mul_fp8_quant_input 时失败。同步 DAS 两个能力谓词与完整 Row forward 的融合输入消费实现，仅对 HCU 新量化参数的调用进入该实现；普通调用和非 HCU 交还原 forward。保留环境门控、kernel 可用性和真实 consumer trait 的组合判断，不用恒 False 或关闭配方开关代替缺失 API。新增 FP8 EnvBool 的默认 false 与 DAS 相同；DPSKV4 原有独立融合开关保持原值。

三个方法只限定真实官方全局 provider；能力方法挂到当前真实 RowParallelLinear，新增已有 forward 入口的 AROUND Hook，合同 377 项。回归覆盖能力/consumer 配对、FP8 q/scale/输出 dtype 经当前 Row forward 到 quant method、默认及非 HCU 委托。外部 LightOp 数学在该 CPU 数据流夹具中替换为可观察 quant producer，不能作为其 GPU 精度证据；完整 DeepSeek 服务继续复验。

## 28. 2026-10-30 共享 MLP 构造门控同步

进一步核对已失败的共享 MLP 路径，官方 DeepseekV2MLP.__init__ 将所有 HIP 都设为 use_fused_clamp_act_mul=True，而固定 DAS 明确是 HIP 且非 HCU 且 SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL。此前只同步 forward 留下初始化分支差异。本次迁入完整当前构造器；除该赋值外两个输入构造器 AST 完全一致，其他初始化、parallel_group 和权重处理没有省略。按 DAS 添加默认 True 的环境字段，HCU 选择 DAS 的后续 clamp 路径；这是源码中的平台资格判断，不是为跑通测试关闭配方开关。

新增构造器 Hook 后合同为 378 项。真实公共类构造回归保留 nn.Module 初始化，替换昂贵线性分配，验证 HCU 与普通 HIP/环境门控；完整 DeepSeek 服务仍需新版本复验。dev29 的第七轮在资源等待中出现多次准入竞争，实际 run_id 以控制器记录为准；未启动的 pending_resources 不计为模型失败或请求结果。

## 29. 2026-10-30 BW1100 DeepEP ABI、buffer 与量化路径同步

DeepSeek dev29 第七轮实际失败在普通 HIP AITER fused-clamp 包不存在，dev30 已按 DAS HCU 构造门控越过该路径。dev30 attempt8 随后在 Graph 的 low-latency dispatch 被 vendor Buffer 拒绝 use_fp8 参数。固定 DAS 已通过签名检测区分 upstream use_fp8 与 BW1100 quant_type/topk_weight ABI；本次迁入完整 DeepEP dispatcher 模块，保留 normal/low-latency、量化生产/消费、combine、event/overlap，以及 target/draft 共用最大布局 buffer 的管理。

仅替换现有 DeepEPBuffer 和 DeepEPDispatcher 两个公开类，其他 AST 相同的数据/配置类型复用官方身份；合同 380 项。HCU DeepGEMM 能力探测复用 DAS 的真实 deepgemm 导入检查，量化布局另要求实际选择 deep_gemm runner，不能因包已安装就切换 AITER/SlimQuant 的 dispatch。没有安装/替换系统 DeepEP，也不通过关闭 DeepEP、speculative 或 Graph 绕过错误。

CPU 回归覆盖 quant_type=0/2/3 与权重/scale contract、不支持 NVFP4 的明确拒绝、backend/能力/SlimQuant 选择、真实公共 dispatcher 模式选择及 target/draft 最大 buffer 布局保留。通信 kernel 由可观察 vendor stub 替代，不能当作真实多卡通信或模型验收；新 wheel 的四卡完整服务继续验证。


同一失败边界继续核对量化消费，当前官方 DeepEPMoE 没有 DAS 的 HCU channel-FP8 grouped GEMM 分支。因此同步其已验证的 legacy channel-FP8 producer/consumer：完整权重 pack/非持久 buffer、普通及 masked FP8 两层 GEMM、LightOp scatter/gather 资格与 fallback、SiLU clamp/FP8 quant。只在 DAS 同一 channel-FP8/SILU/非 OAI/无 local-shared 条件进入；其他模式委托原实现。旧配置类没有新增 local_shared_experts_without_dispatch 字段时读取其 DAS 默认 False，不声明该 opt-in 功能已支持。构造器同步 legacy flags 和 dispatcher quant config；checkpoint 后按 Source 打包并删除 canonical 权重，保留逻辑 shape。两个现有公开方法新增 AROUND 后合同 382 项；量化后处理复用已有合同目标。运行没有动态导入 DAS 源码。

## 30. 2026-10-30 V4 输出收尾 helper 接到真实模型类

DeepSeek dev31 attempt9 的 target verify Graph 已通过完整 body，包括 DeepEP vendor dispatch、HCU channel-FP8 grouped GEMM 和全部模型层，随后在输出收尾缺少 logits_from_body_output。该 helper 已位于当前 DAS 的 owned causal class，但实际公共 causal 类保留官方身份，原注册只覆盖已有方法交集，遗漏了这个新增方法。

HCU 注册现在把同一已迁入的 helper 接到真实官方 causal 类；不更改数学，不伪造隐藏状态。继续保留既有 forward Hook 及 Native 类身份，合同仍为 382 项；只补齐当前基线缺少的能力方法。公共 forward 回归直接走真实模型类到 helper/logits_processor，检查 pre-hc-head、aux/PD capture 和输出 flatten 的实际消费。Source body/helper 已成对同步，Graph 和完整请求须在新 wheel 复验。

## 31. 2026-10-30 normal DeepEP FP8 全局依赖补齐

DeepSeek dev32 attempt10 进入 normal channel-FP8 分发消费，在已迁入的 contiguous GEMM helper 读 get_offloader 时因遗漏导入失败。该 helper 的源模块额外导入当前官方已有的 get_offloader 与 copy_list_to_gpu_no_ce；本次显式接入这两个真实 provider，保留禁用 copy-engine 时的无 copy-engine GPU 计数搬移分支。

新增 normal 两层 GEMM/quant/scatter/gather 的 CPU 数据流回归，使用真实公共 DTO，检查 count、scale、clamp 与输出传递；外部 GPU kernels 由可观察 stub 代替。同步检查该 owned 模块顶层函数的实际引用全局名均存在，防止只跑 masked 路径而遗漏 normal 依赖。没有改精度容差或关闭 normal dispatch；合同仍 382 项，完整服务由下一轮验证。


全局依赖检查又发现 Native ep_moe module 不再 re-export scatter/gather/m_indices；相同检查在缺失时明确失败。对照 Source，scatter/gather 的现行 HCU-aware 实现亦不同于官方，不能只导入名字相似的旧实现。本次把两个入口及其实际函数依赖闭包和 m_indices fallback 放入 owned kernel provider，保留 Source LightOp 资格与原 Triton fallback；只适配 is_hcu 检测导入，所有选中函数 AST 与 Source 同一实现。未改变其他官方调用方。

## 32. 2026-10-30 V4 target/draft backend 分流同步

继续审查启动失败链，发现固定 DAS attention_registry.create_dsv4_backend 明确用 HIP 且非 HCU 才进入 DeepseekV4HipRadixBackend；当前插件遗漏该分流，前述 DeepSeek 轮次实际仍选普通 HIP 类。现在对现有 registry 函数及 draft decode/prefill 工厂增加 HCU AROUND，按 DAS 同一路径进入 V4 factory；其他平台委托原调用。registry dict 保存 callable，不会仅因模块函数被 Hook 自动更新，因此一并接好实际 ATTENTION_BACKENDS['dsv4'] 字典入口，保留原 callable 供非 HCU 委托。

当前配方 auto/FlashMLA 选择已迁入的 owned HCU Base/Multistep backend；不通过临时篡改 is_hip 或平台全局值选路。显式 TRTLLM/Mega、CP/PD 等模式不因本轮而新增验收。合同增至 385 项。前述普通 HIP 分支没有模型通过记录；正确 HCU backend 的真实服务验证另记。

dev33 已完成 6 项 normal/masked/全局依赖/权重别名/DTO/HCU kernel 回归；11 个 owned EP kernel 定义 AST 与固定 DAS 相同。真实 HCU byte scatter、per-token scale 与路由加权 gather 最大绝对误差 0，m_indices 有效行及 -1 padding 精确匹配。这些属于单算子证据，不能代替完整模型请求。

## 33. 2026-10-30 四卡代表模型验证结果

当前运行时代码提交 `86f3f70ac72e673627c816f546eaf647cb045cf9`，wheel `0.3.2.dev34`，SHA256 `45b88367f9fc3f9462eed414d24279fb7176dacb5f9c588733f0e7e0479d415a`；本地提交，未推送。当前源码/安装文件逐字节核验和 385 项完整 Hook 合同均通过，各次实际版本见下表。

| 模型 | 实际 wheel 目录 | 运行 ID | 物理 GPU | 严格请求 |
|---|---|---|---|---|
| qwen3-30b-instruct-tp4 | `site-dev26` | `qwen3-30b-instruct-tp4-aiter-attempt7` | 0,1,6,7 | 6/6 |
| minimax-m25-bf16-tp4 | `site-dev26` | `minimax-m25-bf16-tp4-attempt3` | 0,1,6,7 | 6/6 |
| glm5-int4-tp4 | `site-dev28` | `glm5-int4-tp4-attempt18-admission-retry2` | 0,1,6,7 | 6/6 |
| deepseekv4-fp8-tp4 | `site-dev34` | `deepseekv4-fp8-tp4-attempt12` | 0,1,6,7 | 6/6 |

六个确定性检查为算术、中文、英文、短阅读、代码结果与重复请求；最终答案及 stop 条件均严格通过。各服务结束后自有进程清理为空，端口关闭。Graph、EAGLE/DSPARK 与 DeepEP 按对应配方记录，具体捕获模式和参数偏移见原始运行配置与日志。

环境仍为固定官方 SGLang 0.5.22 加隔离插件，既有容器提供 DAS 0.5.18 依赖，cookbook 0.5.18 仅为配方参考。用户授权四卡 profile；本表不外推完整八卡配方、数据集精度分数、长上下文或性能基准。全量迁移台账保持独立状态。

仅在启动/精度失败处对照固定 DAS 代码，未启动 DAS 对照服务。原始证据在指定 test-model 目录的 README.md、results.json、runs/、artifact-dev 各版本 JSON 和 wheel-hooks-dev 各版本 JSON；历史失败完整保留。

DeepSeek dev34 attempt12 的日志明确记录 HCU owned DeepseekV4AttnBackend，用于 target/draft 的 backend 选择已与 DAS HCU 分流对应。attempt11 在静态比例 0.8 下因 target/draft 权重合计超过 KV 预算停止，框架测得最低可用比例 0.8175；成功轮使用 0.9，并保留 16K KV、8K context、Graph max batch 8、每卡至少 132000 MiB 空闲的准入检查。该变化为测试 profile 的显式容量调整，未更改运行时代码或隐藏原内存检查。

## 34. 2026-10-30 HCU 安装命名与文档一致性核验

当前 `pyproject.toml` 的 distribution、Python 包和 general plugin 入口分别为 `hcu-sglang-plugin`、`hcu_sglang`、`hcu`。本轮从干净代码提交构建 `hcu_sglang_plugin-0.3.2.dev34-py3-none-any.whl`，检查 wheel metadata/entry_points 和包目录，再隔离安装并在源码目录外检查 import 来源。唯一 general plugin 入口为 `hcu = hcu_sglang:register`，console script 为 `hcu-sglang-serve`；无旧包目录或 `hcu` 入口。该 wheel 的 SHA256 为 `79150616eb8c55e984317554addeed9faad82af42fe811d2f76516e8fc012675`，与前文远程历史构建分别记录，不因版本号相同替换历史文件哈希。

修正安装指南的 wheel/editable、metadata、严格加载合同、服务及卸载示例，当前合同显式指定 `--plugin-package hcu_sglang` 并先初始化 AITER 兼容。同步配置、回溯、排障、README 和相关 Skill 中仍将当前实现称作旧包的说明；旧版本实施记录、来源清单、兼容工具接口和历史文件名保持其来源语义。硬件接口 `is_hcu()` 不属于插件入口改名范围。

本轮仓库 72 项离线测试、L0 的 333 个 Python 文件/8 个 Skill、全部 Skill 元信息及 Markdown 链接/标题锚点检查通过。网页从原始文档重新构建，不直接修改生成 HTML。检查证据保存在本次 ignored artifacts 中。本轮未修改运行时行为、未在本机执行 HCU Hook/模型服务；wheel 名称与入口检查不作为模型精度或设备可用性验收。

