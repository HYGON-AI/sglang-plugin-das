# DeepSeek V4.1 W4A8 P/D 部署示例

本目录整理当前 `deepseek_v4.1_w4a8_cp8ep8` 分支的模型22配方：P 为 CP8/EP8，D 为 DP8/EP8，DSPARK block size 5，实际 verify window 6。适用于 SGLang 0.5.22、plugin 0.3.2.dev77 与配套 HCU adapter/provider。所有八卡由本服务独占后才能启动。

## 源码与职责

- adapter 提供既有框架行为的 HCU 兼容；模型新增能力和专项回退位于本仓 `hcu_sglang/`。
- `source-lock.json` 保存本次已运行的21个运行文件、36个选定源文件及外部 wheel 哈希。它是一次源码快照，不能当成未来版本的完整性清单。
- `prefill.sh`、`decode.sh` 从实际模型22配方整理，只将节点、调优数据和专家位置路径改为环境输入。模型参数、量化、预算、Graph 和推测窗口保持原值。
- `env.sh` 固定当前已验证的组合策略：公共 NCCL/RCCL all-reduce、现有 Torch FlashMLA 回退、HCU provider 与私有 Mooncake SDK。没有把 fork 的新增功能放进 adapter。
- 原生 FlashMLA code-object、对称内存路径和 custom AR 不在本配方的通过范围内。Torch 回退只保证其已验证的功能范围，不代表原生性能。

## 配置

在每个节点容器中设置实际路径；`PROVIDER_PYTHONPATH` 可包含多个用冒号分隔的 provider site：

```bash
export ADAPTER_ROOT=/path/to/adapter-das
export PLUGIN_ROOT=/path/to/sglang-plugin-das
export PROVIDER_PYTHONPATH=/path/to/megamoe/site:/path/to/deepgemm/site
export MOONCAKE_ROOT=/path/to/mooncake-shca-runtime
export BLAS_TUNING_ROOT=/path/to/gemm_asm_tuning
export PREFILL_HOST=<prefill-IB-IP>
export DECODE_HOST=<decode-IB-IP>
export PREFILL_EXPERT_LOCATION=/path/to/prefill/static_load.pt
export DECODE_EXPERT_LOCATION=/path/to/decode/static_load.pt
```

`MOONCAKE_ROOT` 须提供 `sdk/include`、`sdk/lib/libibverbs.so.1`、`sdk/lib/libshca-rdmav34.so` 和 `site/mooncake`。它是外部部署依赖，不是 adapter/plugin 的 Python 修复。SHCA SDK 头文件与加载的 host/provider ABI 必须配套；不能只复制一个 wheel 或修改 ELF 头冒充兼容。

容器需要匹配的 Torch/DTK、LightOp、DeepEP/HIPC、MegaMoE、DeepGEMM 与原始 tuning/EPLB 数据。不要在已有 DTK 环境中使用普通 `pip install .` 自动替换 Torch、Triton 或 CUDA 依赖；现有镜像中按已核版本使用 `--no-deps --no-build-isolation` 安装两仓。

## 启动

在 P 节点执行：

```bash
bash "$PLUGIN_ROOT/examples/deepseek_v41_w4a8/prefill.sh" 30000 /path/to/model
```

在 D 节点执行：

```bash
bash "$PLUGIN_ROOT/examples/deepseek_v41_w4a8/decode.sh" 30000 /path/to/model
```

两端各完成 target/draft 加载、Graph 初始化和 ready，且普通健康检查通过后，在 P 节点执行路由器：

```bash
export ROUTER_PYTHON=/path/to/router-venv/bin/python
bash "$PLUGIN_ROOT/examples/deepseek_v41_w4a8/router.sh"
```

路由器为0.3.2，API默认30003、metrics默认29003，显式 `request-timeout-secs=120000` 和 `disable-retries`。该 HTTP 超时不修改服务内部的 KV transfer 等待上限；本次仍保留过300秒等待失败的负证据。此脚本只配置启动，不自动停止旧服务、修改全局库、创建控制器或补发评测请求。

## 验证与当前结果

先执行确定性完整答案和正常停止检查，再使用同一模型、数据集、prompt、seed、输出预算和评测器比较。验收为 HumanEval164一次、GSM8K1319两轮（每侧2802题），MATH500按用户要求取消，候选相对原可运行 DAS 基线的精度降幅不超过5%。性能只作同条件复测，不要求一定追平；prof 数据不计入正式性能。

2026-11-15本次已完整核验：HumanEval158/164=96.34%，GSM8K第一轮1285/1319=97.42%。GSM第一轮三题达到输出上限仍计入分母。第二轮一次502后保留336个成功结果，采用具名恢复继续缺失题；MATH500已取消，新DAS对照待完成。因此**完整精度对齐和性能尚未验收**。

回归源码位于 `tests/runtime/`，包括 CT 权重加载、INT8投影、CP状态/传输、Engram、masked KV、logits gather 和 overlap 契约。CPU夹具、真实算子/Graph、真实模型和最终数据集各自保留验证边界。GPU回归需要另行独占资源和显式测试开关，不能与当前整模型评测同时执行。

完整说明见 [适配与验证文档](ADAPTATION.md)。维护本模型新增能力使用 [hcu-plugin-dev](../../skills/hcu-plugin-dev/SKILL.md)，排障使用 [hcu-plugin-triage](../../skills/hcu-plugin-triage/SKILL.md)。
