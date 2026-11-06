# 第三方与外部来源清单（THIRD_PARTY_NOTICES）

> 本文件登记仓库内非本仓原创的源码来源（HYGON 开源合规规则
> `hygon-open-source-v1.7` §4 要求）。文档与方法库的抽取来源见
> [docs/extraction-map.md](docs/extraction-map.md)；内部仓库许可证副本按文件名逐个登记在 §3，文件位于 licenses 目录。新增移植/借源代码时同步登记本表。

## 1. 外部开源项目

### sglang（Apache-2.0）

- 项目：SGLang，Copyright 2023-2024 SGLang Team（及文件头标注的 vLLM 间接贡献者）
- 仓库：<https://github.com/sgl-project/sglang>（Apache-2.0）
- 许可证副本：仓库根 [LICENSE](LICENSE)（Apache-2.0），与
  [licenses/sglang-plugin-das.LICENSE](licenses/sglang-plugin-das.LICENSE) 为同一份
  （`sha256 c71d239df91726fc…`，逐字节相同）
- 引入方式：运行时插件以 sglang 官方树为宿主；`hcu_sglang/models/deepseek/`
  等 methods 文件按 pinned SHA 从 sglang / 内部 sglang-das fork 迁入
  （文件头保留原版权声明；逐文件映射见 docs/extraction-map.md 与
  `tools/build_migration_manifest.py` 生成的迁移清单）
- 本地路径：[cpu_cache_methods.py](hcu_sglang/mem_cache/cpu_cache_methods.py)，抽取 Tensorcast、HF3FS、SeaweedFS、token-major、multimodal、StorageAttachment、Mamba 和 UnifiedCacheLinkerWrapper 构造方法；保留 Tensorcast 原 SPDX 与 SGLang 版权行。逐定义来源 SHA 见 [CPU 来源合同](tests/contracts/cpu_cache_sources.json)，目标文件 SHA 与接线归属见 [运行时来源合同](tests/contracts/runtime_owned_sources.json)。
- HYGON 修改：hook 接线与 HCU 适配，见各文件与 docs/migration.md
- 本轮 DAS 增量路径：`hcu_sglang/layers/attention/glm5_next/`、
  `hcu_sglang/layers/attention/index_topk_share.py`、
  `hcu_sglang/layers/quantization/{kv_cache.py,moe_wna16.py,auto_round.py,quark_w8a8_fp8_moe.py}`、
  `hcu_sglang/multimodal/encoder_preprocessing.py`；固定 DAS SHA
  `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`，方法级/导入级差异见
  `tests/contracts/runtime_owned_sources.json`

### vLLM（Apache-2.0）

- 项目：vLLM，Copyright contributors to the vLLM project
- 仓库：<https://github.com/vllm-project/vllm>（Apache-2.0）
- 本地路径：`hcu_sglang/distributed_aiter.py`——锚定 vLLM tag
  `v0.6.4.post1` 的 `custom_all_reduce.py`（文件头 Adapted from 链接）；
  deepseek methods 文件头中标注的 vLLM 声明同源保留；本轮另含
  `hcu_sglang/layers/quantization/kv_cache.py` 与
  `hcu_sglang/layers/quantization/moe_wna16.py` 的 vLLM 来源声明
- HYGON 修改：aiter 适配与 HCU 路径改写

### fast-hadamard-transform（BSD-3-Clause）

- 项目：fast-hadamard-transform，Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file
- 仓库：<https://github.com/Dao-AILab/fast-hadamard-transform>（BSD-3-Clause，全文见
  [licenses/fast-hadamard-transform.LICENSE](licenses/fast-hadamard-transform.LICENSE)；
  本地文件经 sgl-project fork 引入，fork 地址
  <https://github.com/sgl-project/fast-hadamard-transform>）
- 版本口径（主从分明）：
  - **主口径（可复现的 vendor 源）**：`sgl-project/fast-hadamard-transform` commit
    `7fd811c2b47f63b0b08d2582619f939e14dad77c`（2025-10-06）——本地文件头 `static_switch.h:1`
    写明 `Copied from https://github.com/sgl-project/fast-hadamard-transform`，即经该 fork 引入；
    许可证全文与 AUTHORS 均按此 commit 取，取全文日期 2026-11-06
  - **参考**：上游 `Dao-AILab/fast-hadamard-transform` commit
    `e7706faf8d1c3b9f241e36860640ad1dac644ede`（2026-03-10）——其 LICENSE 经 sha256 实测
    与上述完全相同（`8c9ccb96c065e706…`），AUTHORS 内容亦相同；仅作对照，不作为登记口径
- 本地路径：`hcu_sglang/jit_overlay/csrc/fast-hadamard-transform/`
  （fast_hadamard_transform.h / _common.h / hadamard_jit.cuh）
- AUTHORS：BSD-3 正文写 "as shown by the AUTHORS file"，故同目录落
  [licenses/fast-hadamard-transform.AUTHORS](licenses/fast-hadamard-transform.AUTHORS)，
  逐字取自同一 commit（22 字节、无结尾换行，内容 `Tri Dao, tri@tridao.me`），sha256 与上游实测一致
- HYGON 修改：JIT 接线，核体保留原声明

### flash-linear-attention（MIT）

- 项目：flash-linear-attention（FLA），许可证原文版权行 `Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li`
- 仓库：<https://github.com/fla-org/flash-linear-attention>（MIT，全文见 [licenses/fla.LICENSE](licenses/fla.LICENSE)，逐字未改）
- 版本口径：许可证全文逐字取自上游 commit `9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1`（2026-11-04），取全文日期 2026-11-06，落盘后与上游原文 sha256 比对一致。本地文件头 `Adapt(ed) from` 标注的是 `Copyright (c) 2023-2025, Songlin Yang, Yu Zhang`，与许可证原文的年份范围不同，属来源版本与随附文本的既有差异；差异如实保留在此，许可证正文不得为此修改
- 引入方式：Gated DeltaNet / KDA 的 Triton 内核与 L2 归一化按文件迁移，文件头 `Adapt(ed) from` 链接与上游版权行原样保留
- 本地路径：`hcu_sglang/layers/attention/fla/` 下 `chunk_delta_h_source.py`、`chunk_o_source.py`、`fused_recurrent_source.py`、`layernorm_gated_source.py`、`l2norm_source.py`；`hcu_sglang/layers/attention/linear/kda_kernels/` 下 `chunk_delta_h.py`、`chunk_intra.py`、`chunk_intra_token_parallel.py`、`fused_norm_gate.py`、`kda.py`、`l2norm.py`
- HYGON 修改：HCU 门控接线、vendor `sgl_kernel.l2norm` 快车道（`l2norm_source.py` 与 DAS 源逐字节一致）、融合内核适配；逐文件来源 SHA 见 [运行时来源合同](tests/contracts/runtime_owned_sources.json)

### DeepEP（MIT）

- 项目：DeepEP，Copyright (c) 2025 DeepSeek
- 仓库：<https://github.com/deepseek-ai/DeepEP>（MIT，全文见 [licenses/deepep.LICENSE](licenses/deepep.LICENSE)，逐字未改）
- 版本口径：许可证全文逐字取自上游 commit `f6aa38fe06cd4a0489e7730365d4e0114970a2f3`（2026-10-21），取全文日期 2026-11-06，落盘后与上游原文 sha256 比对一致
- 本地路径：`hcu_sglang/jit_overlay/csrc/elementwise/concat_mla.cuh`（代码归属）；`hcu_sglang/layers/moe/deepep_current.py` 仅为注释引用，不作为代码归属
- HYGON 修改：HCU 接线与 JIT 适配

### FlashMLA（MIT）

- 项目：FlashMLA，Copyright (c) 2025 DeepSeek
- 仓库：<https://github.com/deepseek-ai/FlashMLA>（MIT，全文见 [licenses/flashmla.LICENSE](licenses/flashmla.LICENSE)，逐字未改）
- 版本口径：许可证全文逐字取自上游 commit `395cb3afbb331fef7b1b21e9340861b9436b42d8`（2026-10-27），取全文日期 2026-11-06，落盘后与上游原文 sha256 比对一致
- **代码来源与许可证来源不同，必须分开写**：本地两个文件的代码来自 fork 分支
  `endurehero/FlashMLA` 的 `will_fp8_mr`，commit `7fafcd217d9fbaf30ef428b1e2e6c9c46d1ddd89`。
  文件头引用的 **PR #54（"fp8 support"）未合并**：GitHub API 实测 `state: closed`、`merged: false`、
  `merged_at: null`，代码从未进入 `deepseek-ai/FlashMLA` 主干。只写“代码来自 FlashMLA”而不提 PR 状态，
  等于默认它进了主干，本条目不允许这样写
- 两个日期口径并列（来源不同，不互斥）：
  - **代码版本日期 2025-03-01**——该 fork commit 的提交日期，取自 GitHub API（主列，表示代码版本）；
  - **PR 关闭日期 2025-03-11**——PR #54 的 `closed_at`（`2025-03-11T07:29:33Z`），取自 GitHub API
    `repos/deepseek-ai/FlashMLA/pulls/54`；PR 创建于 `2025-02-28T14:37:17Z`，`base` 为
    `deepseek-ai:main@1aef31d163f11c0f2c7329fcda71d2985bfb47a0`
- 许可证全文取自 `deepseek-ai/FlashMLA` 主干
  `395cb3afbb331fef7b1b21e9340861b9436b42d8`（2026-10-27），取全文日期 2026-11-06；
  该主干 LICENSE 与上述 fork 分支的 LICENSE 实测逐字相同（`Copyright (c) 2025 DeepSeek`）
- 本地路径：`hcu_sglang/jit_overlay/csrc/sparse_mla_q8kv8_prefill_sm90/dense_fp8_utils.h`、`dense_fp8_transpose_v.h`；同目录 `kernel.cuh` 文件头为 `Copyright 2025 SGLang Team` 且注明 clean-room re-implementation，不在本条目归属范围内
- HYGON 修改：HCU 接线与 JIT 适配

## 1.1 仅注释/信用引用，不构成再分发（不录许可证副本）

以下位置出现过上游项目的 URL 或名称，但经逐处查看**没有逐字复制代码**，因此不录许可证全文。
按 R11/R13 口径，这类排除必须写清判断依据与置信程度，不得写成“已确认非派生”。

- **pytorch（BSD-3）不录**：`hcu_sglang/layers/quantization/fp8_linear.py:280` 与
  `hcu_sglang/distributed_methods.py:294` 仅在注释中引用上游 PR #144432 / issue #138795，无代码复制；
  `hcu_sglang/jit_overlay/csrc/fast-hadamard-transform/static_switch.h:3-4` 的注释写
  「Inspired by … pytorch/aten/src/ATen/Dispatch.h」，该文件主体来自 fast-hadamard-transform
  （BSD-3 已登记并附全文），pytorch 只是设计思路出处。若后续引入逐字代码须重新评估。
- **NVIDIA/DALI（Apache-2.0）不录**：`hcu_sglang/jit_overlay/csrc/fast-hadamard-transform/static_switch.h:3-4`
  注释写 `Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h`
  与 pytorch `Dispatch.h`；该文件本身是 `Copied from sgl-project/fast-hadamard-transform`（第 1 行，
  BSD-3 已登记并附全文与 AUTHORS），DALI 仅为设计思路出处，无逐字复制，不录许可证副本。
- **flashinfer（Apache-2.0）不录**：`hcu_sglang/jit_overlay/csrc/deepseek_v4/topk_v2.cuh:5-10`
  头注释自述参考 flashinfer 与 vLLM 两个实现、只加 credit，主体为自研 kernel；
  `hcu_sglang/layers/glm5_next_layer_boundary.py:202` 是引用 issue 的 TODO 注释。
  **判定为信用引用，未做逐块比对**（未与 `flashinfer/include/flashinfer/topk.cuh@c2b4db2b1a`
  与 vLLM `csrc/persistent_topk.cuh@a8c6ee9b78` 做函数级/行级对照），因此不写成“已确认非派生”。

## 2. 内部 vendor（DeepInfer / deep_select，HYGON 内部项目）

- 本地路径：`hcu_sglang/jit_overlay/csrc/deep_select/vendor/`（kerutils
  工具头 + cuda_kernels）
- 来源：DeepInfer deep_select 组件的 vendor 快照，内部项目，无外部
  许可证义务；其中 kerutils 使用 CUTLASS/CuTe 原语
  （NVIDIA CUTLASS，BSD-3-Clause）
- 精确 commit 锚定与上游核对：deep_select 仓库固定版本待登记（复扫前
  补全 vendor 快照的引入 commit）

## 3. HYGON 内部仓库血缘（非外部第三方）

- sglang-das / hcu-sglang-adapter / sglang-plugin-das：Apache-2.0，许可证副本逐个列明：
  [licenses/sglang-das.LICENSE](licenses/sglang-das.LICENSE)、
  [licenses/hcu-sglang-adapter.LICENSE](licenses/hcu-sglang-adapter.LICENSE)、
  [licenses/sglang-plugin-das.LICENSE](licenses/sglang-plugin-das.LICENSE)；methods 迁移与改写边界见
  [docs/extraction-map.md](docs/extraction-map.md) 与
  [docs/sglang-das-migration-plan.md](docs/sglang-das-migration-plan.md)。

## 4. 登记规则速记

- 外部开源：保留原声明 + 本表登记，不加 HYGON 头。
- HYGON 原创源码：文件头两行（Copyright + SPDX-Apache-2.0）；存量补齐
  用 `tools/compliance/add_hygon_headers.py --from-list`（按合规报告 §3
  清单驱动；迁移/vendor 文件不机械加水头）。
