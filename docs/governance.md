# 职责治理与组合兼容契约

> **本文用途**：规定 adapter 与 plugin 的代码归属、跨仓依赖、目标所有权和证据要求。用于需求拆分与方案评审；运行时关系图见 [架构设计](architecture.md)，开发步骤见 [插件指南](plugins.md)。

## 1. 独立仓库与共同部署

adapter 和 plugin 独立维护、独立验证；同一 SGLang 部署可以使用二者。adapter 处理上游既有行为在 HCU 上的兼容与优化，plugin 处理新增能力和自己的注册入口。**职责归属用来拆分实现，不限制部署只能选一个仓库。**

| 问题 | 处理原则 | 常见产物 |
|---|---|---|
| 上游参照平台已有行为，HCU 缺失 | adapter 补齐兼容和行为一致性 | ABI/平台/影子库/算子路由 |
| 上游没有的新融合、新后端或新机制 | plugin 经 seam/注册机制接入 | 新能力实现、Hook、能力门控 |
| 官方已有的 KV 池布局/算子语义 | 默认走官方；自研 page-major 布局与配套 vllm_ 前缀 HCU FA 算子仅作性能对比（`SGLANG_KV_LAYOUT_HCU_FA` 显式开启，2026-11 起默认关闭——2025 年为 pa 性能引入的布局默认给框架带来持续适配负担） | 官方语义绿灯 + 性能 A/B 记录 |
| 一个需求同时需要两类工作 | 拆成两仓任务并定义接口 | 依赖关系、两个变更、组合验证 |
| 只缺 wheel、编译产物或底层库符号 | 交给对应依赖维护方，保留阻塞 | native/package gap 与验收条件 |
| 框架没有安全扩展点 | 记录最小 integration gap，再决定框架接缝方案 | 接口提案与验证，不静默丢行为 |

性能优化可以发生在两仓；使用自研或 vendor kernel 也不单独决定归属。判据
字面以 adapter 仓（`adapter-das`）`docs/governance.md` §0 判别问句为
单源——本节 §1 与该仓 `docs/plugins.md` §0 是其应用视角；三处任一改动
必须同笔同步。两仓同名文档的分工对照见该仓 `docs/reading-guide.md`
「姊妹仓地图」与本仓 [任务与 Skill 使用指南](reading-guide.md) §5。来源
版本见 [抽取地图](extraction-map.md)。

## 2. 目标所有权与加载顺序

两仓可能在同一链路处理不同层次，也可能触达同一函数。组合前逐目标检查，不能把“各自能运行”当作“组合后等价”。

| 交互面 | 必须说明 | 应验证 |
|---|---|---|
| 导入期适配 | 哪些状态须在首次 import 前建立 | 缓存常量、平台判定、实际模块来源 |
| 相同函数/类/registry | 基线提供者、扩展者、装配顺序 | 最终有效对象、包装链、重复/覆盖行为 |
| `from X import Y` 旧引用 | 谁负责同步别名、在何阶段处理 | 调用者是否仍持有旧对象 |
| 环境变量/默认值 | 所有者、类型、覆盖顺序 | 显式值不被另一仓悄悄改写 |
| 能力停用/fallback | 返回哪一条当前有效基线 | adapter 适配是否仍保留 |
| vendor 库 | 使用方、最低版本、符号和形状约束 | 缺失与不支持分支、错误可诊断性 |

`original_fn` 的语义是传给该 Hook 的当前函数，可能已被 adapter 或前序合法 Hook 包装。不要为了回退去重新寻找未经适配的“官方原版”。刻意组合多个 AROUND 时记录顺序及每层原函数传递；不允许无说明的最后一次赋值获胜。

## 3. 组合验证矩阵

adapter 是平台适配底座（HCU/DTK 语义、算子路由、JIT shim、平台合成态）；plugin 按设计为薄业务层，跟随上游升级，**运行时依赖 adapter 的平台底座，不独立于 adapter 运行**。「独立维护、独立验证」（§1）指仓库流程与各自的 L0–L2 测试，不指部署形态；plugin 越薄，跟随上游升级的阻力越小。

| 组合 | 目的 | 结果要求 |
|---|---|---|
| 官方基线（参照环境可用时） | 建立行为与指标参照 | 固定 SHA、配置、输入和依赖 |
| 官方 + adapter | 证明 HCU 平台底座可用 | adapter 职责范围内的契约与服务验证 |
| 官方 + adapter + plugin | 证明目标部署组合可用 | 接线、fallback、数值/graph、服务和适用性能指标；plugin 替换不得破坏 adapter 基线已验证的语义（官方单测按 nodeid 与 adapter 基线 diff，设计声明过的替换登记为 controlled-diff，不许静默红/静默改断言） |

「官方 + plugin（无 adapter）」**不是验收组合**：仅作故障定位的诊断手段（区分问题出自 adapter 还是 plugin），不设通过标准、不进门禁。plugin 的厚度以每次上游升级时「官方 + adapter + plugin」组合的修复量度量，该指标只增不减时必须复盘归属是否漂移。

无法执行的组合写 `pending` 或有理由的 `not-applicable`。版本记录至少包括 official/plugin SHA，使用 adapter 时加 adapter SHA，再记录镜像、挂载、依赖、硬件与参数。不得假定任何两个最新 HEAD 都兼容。

## 4. 处置台账与状态

迁移台账字段及九类处置见 [迁移契约](../skills/sglang-hcu-plugin-migration/references/migration-contract.md)。台账中的“由谁实现”和“以什么组合运行”要分别记录；必要时在风险/证据字段中引用跨仓依赖任务。

| 状态 | 最少依据 | 不能推出的结论 |
|---|---|---|
| pending | 来源、候选目标或待确认行为 | 不能推出已实现 |
| implemented | 实现、注册可达、解析与适用契约测试 | 不能推出模型精度和性能已验收 |
| verified | 适用组合与功能验证通过，有日志与断言 | 不能外推到未测版本/拓扑 |
| blocked | 缺少接口、依赖或资源，记录解除条件 | 不能静默降级成通过 |
| excluded | 来源证明、接受条件与复核依据 | 不能只以没有硬件关键词为由排除 |

AST 候选数、runtime-export 注册数、apply 成功数与验证能力数分别记录。工具生成的默认分类仅是审计起点，人工核实后才更新状态。

## 5. 源码与文档变更边界

- 日常能力开发保留官方工作树，部署时不从 fork 动态导入实现。
- 回溯工具只在新生成树内修改 bootstrap/loader/packaging；来源树保持原样，输出不等于历史 fork 逐 commit 复刻。
- 当前仓库是方法与工具阶段。引入业务包时须先修改 [架构规划](architecture.md#4-未来运行时目录规划) 和 [CI 阶段约束](ci.md)，同时增加运行时测试，不能只移除门禁来获得绿灯。
- 改规则同步 Skill、操作指南与导航；历史来源快照保留原版本语义，不随文案重写冒充新来源。

交付与评审检查清单见 [协作流程](contributing.md)。

## 6. 上游锚点演进三分类（漂移处置与版本基线备注）

plugin 的 hook/seam 锚定上游内部符号（mem_cache、scheduler、attention/
quantization 注册面……），fork 树与官方树并存是常态。上游演进时锚点漂移的
**处置规则单源在 adapter 仓 `docs/governance.md` §5.3**，三分类摘要：
①改名/移位/自愈 → 默认多候选 + fail-loud，**退役旧候选时须注版本号基线**；
②纯新增服务 → 单候选，**首次创建时即注创建版本**；③同名改语义 → **必须
慎重做兼容性处理、按框架版本号选分支**——版本号不可判别（fork 与官方
同号、day-0 树未发版）时，以结构指纹（`inspect.signature` /
`dataclasses.fields` / 布局常量）为版本判别的等价物，按类缓存、fail-loud。
本节只登记 plugin 侧的落地形态与案例：

- **版本基线备注的载体是对应代码文件本身**，紧贴锚点/分支处
  （`# anchor-baseline(①/②/③): <符号> — <树/版本>: <事实>; …` 一行起），
  治理文档不代登；
- 首批范本（PR#17，两处③，均在 `hcu_sglang/mem_cache/common.py`）：
  `_swa_reprefill_tail_tokens` —— fork（0.5.22 基线）签名 `(self, req=None)`，
  官方 v0.5.22 无参（`unified_radix_cache.py:3602`）：签名按类缓存分发；
  `_match_prefix_params` —— fork `MatchPrefixParams` 多 `return_full_match`
  字段，官方仅 key/cow_mamba/req：`dataclasses.fields` 过滤。两处 fork 与
  官方**同号 0.5.22**，版本号分支不可判别——结构指纹即版本判别；
- **release 联动**：①②类修复向下兼容、直接进 main，按需 backport（判据 =
  该锚点在旧 release 树存在）；③类 PR 合并前必须核对 release 线当期组合
  验证记录（§3 版本记录）不被破坏，必要时 release 分支同笔补兼容臂。
