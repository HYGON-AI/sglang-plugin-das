# GLM5.2 INT4 单实例适配与 0518 对照

> **本文用途**：提供可审阅的 TP8/EP8/CP8 INT4/EAGLE 配方、适配问题与验证边界。用于复现和 PR 评审，不代表任意两个最新 HEAD 的组合已通过 GPU 验收。

## 1. 范围与依赖

- 模型：GLM-5.2-Channel-INT4-w4a8，量化入口 `slimquant_w4a8_marlin`。
- 单机八卡，TP8/EP8/CP8、dense TP=1、prefill CP interleave；无 PD/DCP。
- BF16 计算、FP8 e4m3 KV、page64、chunk16384、EAGLE steps/topk/draft-tokens=3/1/4、开启 Graph。
- 历史硬件：gfx938，八卡各 96 GiB；目标框架 SGLang 0.5.22、Torch 2.13、Triton 3.7。
- 原生 0518 对照使用供应方 SGLang 0.5.18 镜像，不加载 adapter/plugin；其 Torch/Triton 与目标镜像不同。这是版本适配对照，不是只改变 plugin 的单变量实验。
- 安装 adapter 与 plugin 遵循各仓安装说明；源码路径覆盖不能替代依赖和 entry point 的正确安装。

历史验证的 adapter 基点为 `1c9448e1b09514a8d1424ad0d50d70cb88a6d078`，plugin 基点为 `15c09173aeacc24216e330e9a52aa7eb8db042c6` 加本轮补丁。PR 准备时 plugin main 已前进到 `b8ea99304d1f513996491a77f28ac0cb50c09557`，adapter main 到 `fd1b7f5447b930ae1d843a9d123e48ab7741607e`；本轮固定 adapter `fd1b7f5`、plugin `b8ea993` 加补丁进行组合验证；不采用后续 adapter `01b9826`，不自动追随最新 HEAD。

## 2. 启动与配置所有权

示例入口：[launch.sh](../examples/glm52-int4/launch.sh)。先安装匹配的框架、adapter、plugin 和 native 依赖，再执行：

```bash
export MODEL_PATH=/path/to/GLM-5.2-Channel-INT4-w4a8
DRY_RUN=1 bash examples/glm52-int4/launch.sh
bash examples/glm52-int4/launch.sh
```

`HOST`、`PORT`、`LOG_DIR` 和 `PYTHON_BIN` 可覆盖。使用源码时可提供 `SGLANG_PYTHON_ROOT`、`ADAPTER_ROOT`、`PLUGIN_ROOT`；需要经过验证的私有 native 构建时可通过 `NATIVE_ENV_FILE` 设置 `AITER_JIT_DIR`、code-object 版本及必要路径，或用 `AITER_SITE` / `FLASH_MLA_SITE` 前置包目录。仓库不包含这些二进制，不自动下载、重编译或改写 ELF。

脚本保留当时实测的 CLI 别名；`--nsa-*` 在部分新树中已弃用，但这里不把未复验的参数迁移混入性能修改。更新框架时核对解析后的有效配置。

| 配置 | 所有权与用途 |
|---|---|
| `HCU_SLIMQUANT=0` | adapter 已有开关；避免旧 shard/dequant shim 改写新量化后端所需的原始 INT8 权重和 scale。本轮没有修改 adapter 源码默认值 |
| `SGLANG_HCU_GLM_CT_BF16_ROUTER=1` | plugin 的 GLM 数值策略；需配套 SlimQuant 类型匹配修复 |
| `SGLANG_ENABLE_CP_V2=1`、dense TP=1 | 对齐实际 CP/MLP 配方；旧 env 在新框架中的支持范围需复核 |
| LightOp RoPE、融合 DSA TopK、TopK ID 后处理 | 已有能力的显式选择，不是本轮新增算子实现 |

固定 token 输入示例见 [request.json](../examples/glm52-int4/request.json) 和 [tokenizer 指纹](../examples/glm52-int4/request-manifest.json)。先核对 tokenizer 指纹；不匹配时从同一聊天模板重新生成，两边必须复用同一组输入 ID。

## 3. 适配问题与处置

| 问题或修改 | 处置及边界 |
|---|---|
| 权重与量化入口 | 保持原始 checkpoint 权重/scale 传递，关闭旧 dequant shim；检查过首个 QKV 的八 rank 实际加载参数，不宣称检查了每个模型参数 |
| NextN EH norm | SlimQuant 未命中原生独立 norm 策略，补齐量化类型；其余模型/量化保留原路径 |
| Router 输出精度 | 首次可比 draft 的 gate 出现 BF16/FP32 差异；保持 BF16 后，正式 Graph 的回答与接受统计对齐 0518 |
| 专家 ID 映射 | 一个 Triton kernel 合并 lookup、越界和 drop sentinel 处理，保留 CPU/布局回退 |
| 首次 draft TopK seed | 纳入整套组合验证；未做独立模型 A/B，不宣称单项收益 |
| Aiter/FlashMLA | 属于 native 构建和交付兼容性；见下一节，不把二进制补丁混入模型 PR |
| DSA CPU mirror | 候选未完成模型验证，运行源码已回退；不进入本轮已验证修复 |

## 4. Native 兼容性与复现限制

本次节点无法加载镜像中部分 COV6 对象。Aiter 使用同源 native 源码重编译 COV5，覆盖实际所需的 C++/MoE/SiLU/all-reduce 模块；这不是 plugin Python 修改。

历史性能测量中的 FlashMLA 使用原生 `f58f935` 构建的私有 ABI 兼容重打包：保留 kernel 指令，仅修改经过检查的 device ELF ABI 标记，并通过了选定形状及变输入 Graph 对拍。它不是源码重编译，也不是可推广的供应方交付方案。本示例不包含或自动执行该二进制操作。公开复现需要供应方提供与驱动、GPU、Torch ABI 匹配的构建；缺少此条件时，应记录 native dependency gap，不能宣称换个镜像即可复现数值和性能。

## 5. 已验证结果

历史组合使用相同提示词“请用中文简要介绍中科曙光，150字左右。”，temperature=0、关闭 thinking、max_tokens=128；性能测量不带 profiler，清缓存，预热一次后测三次取中位数。

| 开启 Graph 的单请求指标 | 原生 0518 | 历史修复后组合 |
|---|---:|---:|
| 解码速度 | 38.5198 token/s | 39.0293 token/s |
| 输出 token 数 | 104 | 104 |
| verify 轮次 | 44 | 44 |
| 正确 draft token 数 | 61 | 61 |
| 平均接受长度 | 2.363636 | 2.363636 |

固定 19 个输入 token 的 `/generate` 请求接受数分布均为 `[13,12,8,11]`，完整输出一致；北京、17×19、中科曙光介绍三项简单对话内容也一致。39.0293 是整个历史组合的结果，不能归给任意单个 PR。

阶段证据也需保留原配置：专家映射融合在 Graph 节点展开配置中从 29.6209 提升到 30.7601 token/s（约3.85%），对应回答一致；TopK ID 后处理让关闭 Graph 的速度从5.6877提高到5.9667 token/s（约4.9%），但开启 Graph 未测到明确独立收益。各阶段收益不能相加。

最终 BF16 router 修复后未重新采集节点展开 Graph trace，因此没有最终版本的 GEMM、FlashMLA、MoE、间隙等逐项毫秒表。关闭 Graph 的修复后模型与性能也未完成复验。本轮验收限定单请求，数据集精度由使用者另行评测。

## 6. 检查与待验收矩阵

```bash
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s tests -p test_glm52_recipe.py -v
bash -n examples/glm52-int4/launch.sh
python3 tools/build_guide.py
python3 tools/build_guide.py --check
```

| lane | 状态与接受条件 |
|---|---|
| 配方 CPU 检查 | 参数引用、开关、路径、失败传播均使用 fake executable，不启动模型 |
| 最新 main CPU 回归 | 与同一 main 的干净基线按失败条目比较；基线已有失败不可记成全绿 |
| 固定组合 TP8 Graph | 三项对话、单请求测速、固定输入接受统计已复验；最终 eager 待验证 |
| IndexShare | 使用者要求只验整套组合，不做单项模型 A/B；不得据此宣称独立收益 |
| 最新节点展开 trace | 待验证，保持 Graph 开启但展开节点；与 0518 同一采集设置比较 |
| 包/镜像复现 | 私有 native 构建不随 PR 分发，供应方兼容产物到位后另验 |

每轮改动须保存 source SHA、实际 import/native 路径、有效参数、原始请求和响应。冷编译与稳定测量分开。测试后只清理自己的服务，核对资源已释放。GPU 待验收项不能被 CPU scope 测试或历史组合结果替代。

## 7. 本轮固定版本组合复验

- 固定 adapter `fd1b7f5447b930ae1d843a9d123e48ab7741607e`、plugin `b8ea99304d1f513996491a77f28ac0cb50c09557` 加本 PR 运行时补丁；不自动更新代码。
- 三项简单对话通过，文本与既有 0518 对照一致；固定 19-token 请求输出 104 token、44 轮 verify、61 个正确 draft token，分布 `[13,12,8,11]`。
- 整套组合 Graph 单请求三次为 39.0118、39.0214、39.0284 token/s，中位数 39.0214；收益只归属于整套组合。
- 专家映射本轮两项真实 GPU 测试通过，包含整数类型/布局与 Graph 变输入重放。15 项针对性 CPU 测试通过，L0 和导览生成检查通过。
- 合并分支全量测试 360 项，92 failures、9 errors、20 skips；干净同基点 356 项具有完全相同失败条目，没有新增失败。这不是全量测试通过。
- 最终版本 eager 和节点展开 trace 尚未复验；本轮模型验收范围为 Graph 配方。导览实际浏览器检查因工具不能可靠确认当前 URL 而受阻，构建检查不替代截图验收。


本 PR 的性能验收仅限上述单请求 decode 结果，不作其他负载的性能结论。
