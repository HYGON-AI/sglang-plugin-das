# 配置管理：路径、开关与运行时状态

> **本文用途**：区分工具参数、SGLang/adapter/plugin 配置的所有权，说明组合运行时的覆盖与记录规则。当前 `hcu_sglang/config.py` 与 `hcu_sglang/environ.py` 已含插件配置和注册逻辑；所有权及状态规范继续适用于当前实现。

## 1. 当前工具的路径配置

| 名称 | 含义 | 要求 |
|---|---|---|
| `REPO` | 当前方法库 | 在仓库根目录用 `pwd -P` 获取；执行 L0、工具测试和脚本的位置 |
| `OFFICIAL` | 指定 SGLang 官方工作树 | 固定 SHA；与清单工具的 official-ref 匹配 |
| `FORK` | DAS 来源树 | 审计保留 dirty；不能作为正式运行时 import 来源 |
| `PLUGIN` | 含业务实现的 plugin 树；当前实现可与 `REPO` 相同 | 当前包为 `hcu_sglang`；支持旧布局的工具按需显式选择 `--plugin-package` |
| `ARTIFACTS` | 本次审计/日志输出目录 | 建议位于 `REPO/artifacts/<task>`，不要落到来源树内 |
| `OUTPUT` | 源码回溯新输出树 | 不与任一输入相同、互为父目录或子目录 |
| `WHEEL` | 本次安装的 wheel 文件 | 安装前核对交付方 SHA256；不从旧任务目录自动选择 |
| `MODEL` / `SERVED_MODEL_NAME` | 模型目录 / 实际服务模型标识 | 按配方填写；响应校验使用服务返回的模型标识 |
| `CONTAINER` / `CONTAINER_USER` | 执行容器 / 合法的 UID:GID | 仅容器方式需要；镜像挂载内路径与宿主路径分别确认 |
| `SOURCE_ROOT` / `VLLM_SKILL_SOURCE` | 历史抽取来源根 / 原 vLLM Skill 根 | 仅来源溯源需要；JSON 中以变量占位符记录，不是工具默认路径 |

在仓库根目录准备本地配置：新建 `.env`，按上表填写本次任务需要的变量（如 `OFFICIAL`、`PLUGIN`、`ARTIFACTS`）及路径；已有 `.env` 时保留并编辑它。来源、产物和模型路径可位于任意目录，不要求互为同级。外部路径填写本机实际绝对路径，含空格的值使用引号。进入容器后按容器内可见位置配置，不能直接沿用宿主路径。

```bash
# 从当前方法库根目录执行；.env 只保存本机配置，已被 Git 忽略。
export REPO="$(pwd -P)"
test -e .env || touch .env
# 编辑 .env 后再加载；仅 source 自己维护、可信的配置文件。
set -a
. ./.env
set +a
# 按任务检查必需项，例如插件能力开发：
: "${OFFICIAL:?请在 .env 中设置官方源码路径}"
: "${PLUGIN:?请在 .env 中设置外部插件源码路径}"
: "${ARTIFACTS:?请设置本次证据目录}"
```

这些是手册中的 shell 变量，不是本库自动读取的环境变量接口。加载 `.env` 是显式操作；实际 CLI 参数以工具 `--help` 为准，设置 `PLUGIN` 不会自动启用运行时插件。只阅读文档或运行离线工具测试时无需配置外部路径。首次初始化插件只需现有官方源码与外部新工程目录；按对应指南检查所需输入。

临时安装或示例工程使用 `mktemp -d`，由系统及 `TMPDIR` 决定位置，不固定临时文件名。需要持久部署时显式指定可写安装目录。历史记录中的 `${REPO}`、`${SOURCE_ROOT}` 等是脱敏位置标识；来源清单仍保留原始版本及文件哈希，不自动解析占位符，也不把当前路径冒充历史路径。原始 ignored 运行日志保持原样，按任务单独保管。

共享文档、Skill、模板与代码不得包含个人目录、私人节点名称或固定任务临时目录。目录结构示意使用相对路径；外部输入用显式变量或 CLI 参数。操作系统接口（如 `/proc`）和标准 shebang 不属于用户目录配置。

## 2. 运行时配置所有权

| 配置来源 | 所有者 | plugin 接入原则 |
|---|---|---|
| `SGLANG_PLUGINS` 与框架参数 | pinned SGLang | 按该版本 loader/CLI 语义使用 |
| SGLang 自有 `Envs` / typed descriptor | pinned SGLang | 有对应 API 时使用 `.get()` / `.override()`，不在 Hook 中散落重复解析 |
| adapter 的配置/快关 | adapter | 引用其固定版本手册；不复制同名注册表或改写它的默认值 |
| plugin 自有能力开关 | plugin 实现 | 集中定义、记录类型/默认/支持条件，独立命名并可诊断 |
| NCCL / driver / launcher / OS 变量 | 外部依赖或部署 | 按所有者的接口消费，不擅自改变语义 |

本仓 `hcu_sglang` 已包含来源插件的配置开关和 `hcu` 入口；具体以当前代码为准，不假定存在统一的 `HCU_DISABLE_OPS` 或额外配置服务。

### 当前 Qwen4 运行时配置

以下字段由 `hcu_sglang/environ.py` 注册到官方 typed `Envs`，在服务启动前设置并继承给 worker；更改后重启服务。默认关闭的性能路径需按支持条件显式启用，checkpoint/INT8 PLE 支持不依赖这些性能开关。

| 名称 | 类型 / 默认 | 行为与支持条件 |
|---|---|---|
| `SGLANG_QWEN4_COMPACT_PREFILL` | bool / false | 纯 EXTEND/IDLE 使用 SUM_LEN，idle rank 保留一行；TP=DP=EP、attention TP/CP=1、Gatherv、无 A2A/TBO、disabled prefill Graph |
| `SGLANG_QWEN4_LOCAL_ROUTING` | bool / false | compact prefill 在量化前的本 rank BF16 输入上选专家；要求非融合 shared expert |
| `SGLANG_QWEN4_ROUTED_INT8` | bool / false | 局部 routing 后传输 INT8 codes/scales，直接交给支持预量化的 AITER moe_c 一次 |
| `SGLANG_ENABLE_DP_GATHER_INT8` | bool / false | Qwen4 FFN 的独立 INT8 transport 入口；SUM_LEN/Gatherv、支持预量化的 quant method |
| `SGLANG_MOE_C_NATIVE_SORT` | bool / false | E129/top-k10、HIP、支持的 block size 使用保留 drop-route 的 native sort；依赖 AITER sorting/conversion API |
| `SGLANG_QWEN4_SPLIT_HC_MIX` | bool / false | gfx936/BF16、1–16 行、HC4/hidden2560/lowrank320 的分阶段 HC mix；保留确定性模式守卫 |
| `SGLANG_QWEN4_FLAT_STATE_SCATTER` | bool / false | gfx936、36 层 FP32 48×128×128、1–16 条 metadata 的连续状态回写；保留 scratch 源行映射 |
| `SGLANG_QWEN4_PREFILL_DIAGNOSTICS` | bool / false | 首次 compact 布局输出 rank/counts，和 routing 日志一起诊断实际路径 |
| `SGLANG_PLE_ETP_REDUCE_SCATTER` | bool / true | 支持的 PLE 分片布局采用 reduce-scatter；其他布局保留 all-reduce + scatter |
| `SGLANG_PLE_ETP_REDUCE_SCATTER_MIN_TOKENS` | int / 2048 | 小于阈值时使用 all-reduce + scatter |

关闭相应性能开关或超出窄形状门控时调用当前有效基线，不绕过 adapter。显式启用 INT8 transport 而缺少必要量化/通信合同，或启用 native sort 却缺少 vendor API 时明确报错。`SGLANG_DP_USE_GATHERV`、Graph、并行与 MoE 后端属于官方配置，按其接口设置；adapter 的 `qsa_mqa` 快关参见 `adapter-das` 仓 `docs/env-vars.md`，不由插件重复定义。完整支持范围与已测组合见 [开发记录](hcu-sglang-development.md#2026-11-04-qwen4-exp-w4a8-与-int8-ple-适配)。

## 3. 新增配置的设计检查

| 检查项 | 要给出的答案 |
|---|---|
| 名称与类型 | 谁拥有；布尔/枚举/数值/字符串，取值范围是什么 |
| 默认与覆盖 | 未设置、显式设置和不支持环境下分别采用什么；显式用户值优先，非法或不支持值给出可诊断结果 |
| 生效阶段 | import 前、进程初始化、请求级还是 forward 级；修改后是否需重启 |
| 依赖与组合 | 需要哪个 adapter/vendor/official 版本；另一个仓是否影响同一状态 |
| 观察与回退 | 日志如何显示最终值、来源和选中路径；停用后回到哪条有效基线 |
| 测试 | 默认、显式覆盖、不支持/缺依赖、进程传播与恢复是否覆盖 |

优先用一个含义明确的配置，不增加彼此冲突的多个布尔项。使用目标版本实际存在的 API；不能因为 DAS Skill 讲了 RuntimeContext 就假定旧 official 也具备同名接口。

## 4. 配置与状态分层

```mermaid
flowchart LR
    E[启动参数与环境] --> C[解析后的进程配置]
    C --> R[资源/stream/buffer 生命周期]
    R --> F[请求与 per-forward 状态]
```

配置、进程资源和每步状态分开。融合 KV 写入标记属于 batch/layer 的生命周期，不放模块全局跨请求复用；Graph 的动态输入不能只在 Python capture 时赋值。若 pinned 版本有 publish/override 机制，按其生命周期写入并记录来源。

平台判定也有所有权：adapter 可能向 SGLang 暴露 CUDA 语义，而真实 torch/编译器仍是 HIP。plugin 的硬件能力判定应使用足够直接且被验证的设备/依赖信息，不能只凭某个 SGLang `is_cuda()` 推断物理设备与 vendor 符号。

V3 的外部运行时输出实际引入 `HCU_PLUGIN_MINIMAX_OPT`，由 `--minimax-opt` 在构造配置前设置并继承给 worker；在 ServerArgs 上仅提供兼容读取，不写冻结记录。该接口最初属于外部 runtime 0.3.0，当前迁入实现见 `hcu_sglang/config.py`；历史行为与验收边界见 [V3 手册](v3-runtime-validation.md)。

官方配置记录可能是 msgspec Struct，不要仅靠 dataclasses.is_dataclass 判断；使用 pinned 版本的 `arg_utils.is_record`/record_fields。CLI 解析后还要 resolve_once 才算完成配置解析。

### 当前 custom all-reduce 参数

当前 HCU 插件提供 `--custom-all-reduce-backend {auto,native,aiter,off}`，默认 `auto`。官方0.5.22的原始 ServerArgs/comm记录没有该字段，插件在 CLI 构造原始记录之前消费参数，并用内部 `HCU_PLUGIN_CUSTOM_ALL_REDUCE_BACKEND` 继承到 worker；不向冻结输入或 `get_exec().comm` 补写属性。直接构造官方 `ServerArgs(custom_all_reduce_backend=...)` 不是已支持接口；程序化调用应在初始化 engine 前使用 `hcu_sglang.config.configure_custom_all_reduce_backend(value)`。

| 规则 | 实际行为 |
|---|---|
| 优先级 | 官方解析后的 `disable_custom_all_reduce=True` 强制 `off`；否则保持显式后端；HCU上 `auto` 且 legacy `SGLANG_USE_AITER_AR=1` 按DAS提升为 `aiter` |
| 生效阶段 | 每个 worker 的官方 distributed bootstrap 先设置原有 flags，再将插件解析结果交给组工厂；一个进程对应一个活动 engine 配置 |
| 组覆盖 | world组始终 `off`；model-parallel组默认取解析结果；显式组级参数优先，不修改其他组成员定义 |
| strict-Fabric | `aiter` + `AITER_AR_TRANSPORT=fabric` 时构造异常原样抛出，disabled communicator 明确拒绝；不静默回落成 RCCL |
| QuickAllReduce | 保留DAS的独立 `SGLANG_USE_QUICK_CUSTOM_ALLREDUCE` 门控；Fabric/auto transport 不并建 IPC QuickAllReduce |
| 原配方别名 | `--nsa-prefill-backend` / `--nsa-decode-backend` 使用原 deprecated alias action，写入当前 DSA 字段，未指定时不覆盖 canonical 默认值 |

后端初始化日志记录 requested/selected transport、disabled、ranks与world size。参数解析通过不代表 communicator 初始化或模型正确；IPC、Fabric、Graph及各组拓扑分别验收，见[开发记录](hcu-sglang-development.md)。

## 5. 变更与排障记录

每次配置改动同步默认值说明、支持范围、测试与启动日志示例。发生冲突时先列出“原始值 → 解析值 → 最终值 → 实际分支命中”，再修改实现。验证模板见 [validation-record](../templates/validation-record.md)，详细状态约束见开发 Skill 的 [runtime-and-env](../skills/hcu-plugin-dev/references/runtime-and-env.md)。

## 6. GLM CT W4A8 的原生 BF16 RoPE 选择

插件开关 `SGLANG_HCU_USE_LIGHTOP_ROPE` 为布尔型，默认 `False`；在 GLM-5.3 CT W4A8 配方中显式设置为 `1`。它在 HCU 的 BF16 `RotaryEmbedding` 构造结束后准备镜像内公共 LightOp `rotary_embedding_fuse`，由正式 `forward_cuda` AROUND Hook 消费；修改后需重新构造服务。默认关闭时保留既有分发，非 HCU、非 BF16 或融合 KV-store 调用保留当前有效原实现。

该路径使用已有 `HcuRotaryFallback` 的 typed cache memo，为 LightOp 提供与 query 同 dtype/device 的缓存视图；框架持有的 FP32 `cos_sin_cache` 及其他 indexer 消费者保持原所有权。LightOp 的缺失或执行错误直接暴露，不在数值失败后静默切换其他算法。此选择不改变 CT W4 权重、动态 INT8 激活或 GEMM 分发。

LightOp 自行选择 launch，不套用旧 sgl_kernel 的 256-thread head 分块；真实 64-head 输入完整调用与拆成 8-head 调用的数值并不相同。offsets 在调用前按原语义加到 positions，返回原地更新的 query/key。

适用证据和待验证范围见[开发记录](hcu-sglang-development.md#2026-11-05-glm-ct-w4a8-的-bf16-rope-选择)。两组八 rank 的真实 Q/K 及更换输入后的组件 Graph 已逐元素对齐原生，完整服务仍待复测。

## 7. GLM CT W4A8 的公共 W8A8 GEMM 分发

`SGLANG_HCU_W8A8_USE_LIGHTOP` 为插件布尔开关，默认 `False`。GLM CT W4A8 配方显式设置为 `1`，使既有 `triton_scaled_mm` 接口使用镜像内公共 LightOp 分发，跳过插件自行选择的 smooth ASM 快路径；公共接口内部仍可能使用 ASM。开关不改变输入 INT8 量化值、token scale、权重或输出 dtype；默认关闭及非 HCU 的行为保持原样。应在服务启动前设置，Graph 捕获后不要切换后端。

真实首层 Q-b 的两组八 rank 输入、权重和 scale 对照中，两套镜像的公共接口均逐元素重建原生输出；插件的 ASM 快路径有四个元素不同。该有限组件证据不证明所有 shape 或完整模型通过，后续分发和模型复测见[开发记录](hcu-sglang-development.md#2026-11-05-glm-ct-w4a8-的-w8a8-gemm-分发)。

## DeepSeek V4.1 的逐通道 FP8 特殊投影

`SGLANG_DSV41_WO_A_QUANT_BEFORE_TRANSPOSE=1` 使连续的分组 wo_a 输入先按行量化，再移动 FP8 行和 scale。设为 0 使用原始 BF16 转置顺序。

`SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME` 默认为 `auto`。插件根据 HCU、量化方案和本地 checkpoint 的 safetensors 元数据选择路径。目标与独立草稿各自检查自己的 checkpoint。符合条件的 wo_a、Markov 输出投影和 lm_head 保留 E4M3 权重及每输出行的 FP32 scale，Markov 的 BF16 embedding 沿用原加载器。要求 FP32 logits 时，lm_head 直接输出 FP32。

`0` 关闭此路径。`1` 用于显式选择兼容量化方案的自定义加载器，权重类型、scale 形状和加载完整性仍由 loader 校验。修改环境变量后重新启动服务。FP8 lm_head 与 lm_head LoRA 的组合会在初始化时报告不支持。

For HCU LP expert dispatch, `SGLANG_LPLB_HIP_SOLVER=device` selects the on-device water-filling solver. The default `host` solver also uses the compact replica layout. `SGLANG_LPLB_HOST_PASSES` (4) and `SGLANG_LPLB_HOST_MIN_PAIRS` (6144) control the iteration count and the threshold below which replicas are split evenly. Rank-local empty inputs still contribute to the EP count reduction. `SGLANG_HCU_ROCBLAS_NN` (default true) prepares contiguous transposes for eligible BF16 target projections after loading or IPC attachment. Set it to 0 before launch to use the original weight layout.

### DeepSeek V4.1 的 HCU CP prefill 和 Engram 共享表

| 环境变量 | 默认值 | 生效条件 |
|---|---:|---|
| `SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA` | 自动 | HCU V4.1 的 FP8 routed experts、584 字节 V4 FP8 KV 启用 BF16 FlashMLA 读取。普通 decode、target verify、draft extend v2 在读取时反量化；常规 prefill 使用现有路径。置 0 恢复 FP8 FlashMLA。 |
| `SGLANG_DSV4_HCU_USE_LIGHTOP_BF16_GATHER` | 自动 | 上述默认路径使用 LightOp 的单池或双池 gather。置 0 使用 Triton。自动选择时缺少任一基础导出会使用 Triton；显式置 1 时缺少导出或禁用 BF16 FlashMLA 会报错。 |
| `SGLANG_USE_FUSED_SILU_MUL_FP8_QUANT` | DSV4.1 独立共享专家自动 | HCU 上单 P 的 replicated BF16 shared MLP 在 down_proj 支持动态 token／逐通道 FP8 时，使用 LightOp 的 SiLU、clamp 和 FP8 量化。显式设为 0 后继续按原有配置分发。修改后重启服务。 |
| `SGLANG_USE_FUSED_RMS_QUANT` | DSV4.1 WQ-B 自动 | gfx938 的 DSV4.1 动态 token／逐通道 W8A8-FP8 WQ-B 使用 LightOp 合并 QNorm 和激活量化。设为 0 后，原 QNorm 和 CT GEMM 分别处理 BF16 输入。其他量化方案和模型继续使用各自的默认设置。 |
| `SGLANG_HCU_OPT_QNORM_STRIDED` | 1 | LightOp v2 可以直接读取 packed Q。设为 0 时先复制成连续 Q 行，再执行同一融合。该开关在构造 attention 时读取。 |
| `SGLANG_HCU_OPT_DSV41_CP_POOL_NORM` | 1 | C2 的 CP prefill 在 KV 宽 512、FP32 输入以及连续 BF16/FP32 norm 权重时使用 LightOp 合并成对池化与归一化。关闭后由 compressor 分步完成。 |
| `SGLANG_HCU_OPT_INDEXER_PREP` | 1 | V4.1 稠密 prefill indexer 对连续 BF16 [T,H,128] 查询直接消费 complex64 频率表和 int32/int64 位置，在一次 LightOp 调用中完成 RoPE64 与每 32 通道 UE8M0/E2M1 fake quant。关闭后先 gather 频率并沿用原算子。 |
| `SGLANG_HCU_INDEX_Q_ROPE_PACK` | 1 | single P 的 decode graph 在 HCU 缺少 AOT RoPE 算子且有 32/64 个 index heads 时，沿用 torch complex 旋转，再用 Triton 同时计算两阶段 FP4 pack 和 head weights。关闭后分别执行查询、量化和权重缩放。 |
| `SGLANG_HCU_OPT_DSV41_ROPE_FQ_STORE` | 1 | 支持融合写入的分页 V4 FP8 池在连续 BF16 latent（宽 512、RoPE 宽 64）上合并 RoPE、量化与写回。关闭后由原 KV 写入流程完成，V4.1 FP4 的路径独立。 |
| `SGLANG_DSV4_FUSE_MHC_REPEAT_CP_SPLIT` | 1 | 旧版 round-robin CP 在非 speculative extend 的首个 PP rank 直接复制本 rank 的 mHC 输入行，TBO 保留完整行。 |
| `SGLANG_DSV41_CP_PRUNE_FINISHING_PREFILL` | 1 | DeepSeek-V4.1 CP 单 P 的 overlap 调度在 prefill 准入前移除已完成或最后一个输出已排队的请求。0 沿用当前框架调度。 |
| `SGLANG_DSV4_CP_OUTPUT_INPLACE` | 1 | 已完成的等长 round-robin all-gather 在独占输出上重排。带显式 gather index 的 tail 和 graph capture 保留原路径。 |
| `SGLANG_DSV4_CP_OUTPUT_INPLACE_MAX_CYCLE` | 128 | 置 0 取消排列周期上限。正值限制 LightOp 就地重排的最长周期，超限的大规模二维模型输出可分块还原。 |
| `SGLANG_DSV4_HC_BF16X3_MIN_ROWS` | 128 | 连续 BF16 mHC 投影在此行数至 65536 行间可使用 BF16x3，同时要求已加载拆分权重并关闭 batch invariant。 |
| `SGLANG_OPT_DSV41_CP_LOCAL_COMPRESSOR_PROJECTION` | 0 | 本地完整 4096 行、全局长度恰为本地长度乘 CP 宽度、无 encoder replay 过滤时，可先投影再 gather。 |
| `SGLANG_HCU_DSV41_BF16_MAIN_PROJ_PREPACK` | 0 | PD prefill 的 DSpark 草稿在加载完成后重排连续 BF16 main_proj。single P 的 DSPARK=0 不创建该层。 |
| `SGLANG_HCU_ENABLE_DSV41_ENGRAM_HOST_TABLE` | 1 | HCU 可使用 Engram 主机表，也可由官方 `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` 请求。 |
| `SGLANG_HCU_DSV41_ENGRAM_HOST_TABLE_LAYOUT` | shared | `shared` 每个主机组共享一份 HSA IPC 表，要求在 HIP 初始化前设置 `HSA_USERPTR_FOR_PAGED_MEM=0`。未满足此条件时使用设备分片。`per_rank` 使用各 rank 的主机分片，weight-cache IPC 需要 shared。 |

`SGLANG_HCU_MEGA_MHC_OPT`（默认 true）控制 DeepSeek V4.1 attention→FFN 的 DeepGEMM Mega mHC 打包与融合。在 CP 配置下仅用于非 speculative extend；设置为 0 可按相同输入对照普通 mHC 边界。进程需重新加载模型以应用配置。

### DeepSeek V4.1 single-P 稀疏注意力

- `SGLANG_DSV41_PREFILL_TOPK_KV_WORKSPACE=1` 默认在 HCU eager EXTEND 的 C1/C2 使用当前层 TopK 所选 KV。全容量 V4 在 gfx936/gfx938 按所选位置反量化，其他可压缩情况按 GPU 有效行计数解包。设为 `0` 可对照逐行 gather。Graph、TBO、request-window replay 及 decode 保留原路径。
- `SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION` 沿用已有 MQA 显存预算；触发切块时将各请求的有效 Q 行合并填充每块，并保留各自的 K 偏移和 candidate 数据。
- `SGLANG_HCU_OPT_DSV41_PREFILL_CONSUMER_40352=1` 默认跳过覆盖全部历史的候选筛选，按 late-layer tail 实际需要的行发布候选。设为 `0` 可对照完整候选发布。gfx938 默认使用 dense 候选消费者；显式设置 `SGLANG_HCU_PREFILL_PAGED_SPARSE` 可覆盖此选择。

#### DeepSeek V4.1 Engram

| Environment variable | Default | Scope |
|---|---|---|
| `SGLANG_HCU_OPT_ENGRAM_HASH_IDS` | `1` | Enable LightOp hashing for the supported EXTEND and fixed-block TARGET_VERIFY layouts. Set to `0` to call the active official provider. |
| `SGLANG_HCU_OPT_ENGRAM_GATE` | `1` | Enable the additional HCU fused-gate route where the official platform check selects torch. Set to `0` to follow the official gate dispatch. |

The official `SGLANG_OPT_ENGRAM_EXTEND_HASH` selects a separate fused
extend-and-commit kernel and takes precedence for eligible batches.

### DSV4.1 CP prefill 的 mHC residual 写回

gfx938 上，使用前一子层 pre 系数的 HC4、hidden 5120 CP 路径对连续 BF16 激活和 FP32 混合系数调用 AITER mHC post，与参考 fork 使用相同的 BF16 写回舍入。设置 `SGLANG_OPT_USE_AITER_MHC_POST=0`，或显式选择 `SGLANG_OPT_USE_TILELANG_MHC_POST=1`，可沿用既有分发。修改配置后重启 client。

V4.1 的 gfx938 HC4 / hidden 5120 CP 路径在 mHC combine 后对 BF16 行使用 HCU RMSNorm。显式设置 `SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=0` 可恢复原有 RMSNorm 分发。此处仅处理无额外量化、普通方差和乘权重约定的输入。
