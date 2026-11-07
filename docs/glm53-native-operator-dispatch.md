# GLM5.3 原生算子分发接入（2026-11-06）

> 2026-11-07 更新：下文保存 11 月 6 日候选接入记录，不代表最终部署。DTK26.10 启动发现 `torch.ops.sgl_kernel.rotary_embedding` 未注册，最终版本保留已运行的 LightOp RoPE 分发，只将 GLM canonical cache 改为 BF16。MLA 拼接同样保留本轮运行的 JIT 路径，不纳入未部署的 raw torch.ops 替换。`expect_m=-1`、recv-hook 与 stream 修改已有后续运行记录；最新状态见开发记录。

## 范围与基线

用户要求对照原生 profile 直接接入尚未对齐的调用，本轮明确跳过对拍、GPU、graph 和整模型运行验证；只运行项目 lint、语法与 diff 检查。不更改环境变量、驱动、库、W4A8 量化策略，不切换运行服务。

插件修改基线为 `0dcde9bf89689515e7a0ccb967d9652aaf659d83`；配套个人 adapter 快照为 `42296d823a9d8e292e74f33e5ba5b62f2c63e587`。异常 profile 来自 adapter `7ad53f6` 与 plugin `ae3160f` 运行时，后续文档提交不代表新的运行证据。参照为原生镜像实际安装的 SGLang `2610271044.ge37bbf.glm53w4a8`，读取路径 `/usr/local/lib/python3.10/dist-packages/sglang`，保留该镜像的 sgl_kernel、LightOp、DeepGEMM、DeepEP 二进制及 ABI。参照文件从 decode node23 原生容器只读提取，哈希列于下文；不能视为对四节点安装状态的新验收。

profile 证据位置使用验收目录下的相对路径：`native-decode-curl-profile-20261106-v1/profile.tar` 与 `adapter-native-decode-live-20261106-1425/traces.tar`。各有 16 rank、每 rank 5 组 draft/verify/draft-extend graph。原生主要是 DP6 单请求，其余 rank 参与 EP；adapter 同时处理评测请求，bs 为 1–3，且随后发生 kernel fault。两组不能直接作为等负载吞吐 benchmark。

## 调用差异与接入

| 项目 | 修改前 → 原生及本次接入 | 状态 |
|---|---|---|
| DeepEP 低延迟调度 | 新框架 HCU 默认 `return_recv_hook=False`，dispatch/combine 为 `async_finish=True`，消费者等待 event；原生 dispatcher 显式 `return_recv_hook=True`。CT W4A8 的 `hcu_w4a8_int8_dispatch` 配置现在将低延迟 dispatcher 接为 recv hook，使两次库调用使用 `async_finish=False`，dispatch_b/combine_b 执行 hook。 | 存在差异；代码已接入，运行未验证 |
| MoE 激活量化 | `forward_masked → lightop.fuse_silu_mul_quant_ep` 之前额外传入估计 `expect_m`；原生 wrapper 默认 `-1`，caller 只传激活与 masked_m。本次去掉该额外参数。两次 W4A8 GEMM 的 expected_m 不变，INT8 输出、FP32 scale 和权重布局不变。 | 存在差异；代码已接入，运行未验证 |
| GLM 主注意力/NextN RoPE | 从 LightOp 路由接到实际安装 `torch.ops.sgl_kernel.rotary_embedding.default`，沿原生 256-thread 限制拆分 query heads，key 按匹配布局旋转。复用既有位置、head_size、interleave/NeoX、offset 参数；fused KV-store 分支继续交给当前原函数。 | 存在差异；代码已接入，运行未验证 |
| GLM Indexer RoPE cache | adapter 的 CUDA 兼容分支保留 FP32 canonical cache，原生 HIP 构造器按 layer dtype 转换，实际 trace 为 BF16 specialization。本次 GLM Indexer 和主注意力构造后按 layer.dtype 转换 cache；fused LN/RoPE 直接读取该缓存，不增加逐 token 转换。 | 存在差异；代码已接入，运行未验证 |
| DSA MLA Q 拼接 | GLM DSA backend 原来调用新框架 JIT concat；现在对所有该 backend 的调用点选择镜像 `torch.ops.sgl_kernel.concat_mla_absorb_q`。保持原生 BF16、三维、512+64、末维 stride=1、外维 stride 为 8 的倍数等完整条件；不符合时按原生 torch.cat。 | 存在差异；代码已接入，运行未验证 |

RoPE 和 MLA 拼接在 HCU + GLM MoE DSA 架构下启用，其他架构保留既有路由。CT W4A8 调整只影响既有专用 INT8 dispatcher/compute 路径；未调整 FP8、W4A16、普通 BF16 通信。保留原有 combine-input producer event 和 overlap-stream 依赖，不删除此前正确性修复。DeepEP Buffer 生命周期、路由权重应用和 EP 组不变。

## 接入归属与加载依赖

plugin 修改其自有 GLM/DSA/MoE 消费者，不改社区框架源码，不新增全局平台伪装或环境变量。实际底层仍是镜像配套库。原生 sgl_kernel 导入会加载 common_ops；adapter 的同名影子包通过已有 `sgl_kernel.weak_ref` 加载镜像 common_ops。本次调用该扩展注册的 torch.ops，不经影子包的 Triton rotary 实现。镜像缺少符号或 ABI 不匹配时保留明确异常，不默默回退到另一实现；本轮未重新加载二进制或进行 ABI 运行验证。

## 耗时结论的边界

在此次 profile，native DP6 的 DeepEP dispatch/combine 各 800 个 kernel，adapter 每 rank 各 400 个；recv-hook 与 event 路径的源代码差异能够解释调用结构不同，是本轮优先修正项。adapter 部分 RMSNorm、FlashMLA combine、激活量化也显示明显长耗时，但 kernel 名称相同不能证明输入、流依赖和等待相同，也不能证明内核自身变慢。保留原始异常记录，不把这些数字直接当成需要替换每个算子的证据。

RMSNorm、FlashMLA、其他 MTP/metadata 内核的耗时异常以及 kernel fault 根因仍为未验证；本次未声称这些问题已解决。没有执行用户明确跳过的运行验证，因此接入状态不能写作“已对齐”或“性能恢复”。既往对拍报告只作为历史证据。

## 原生文件哈希

| 实际安装包中的相对文件 | SHA256 |
|---|---|
| `layers/moe/fused_moe_triton/layer.py` | `97d0d13e9314ca5b74911d48219632aced38dca2b3463f01c83c0ea65494244e` |
| `layers/moe/token_dispatcher/deepep.py` | `30397487113bdddc1e741c50dffa06d15f246db9db4bd40e22b0b53ae7c4790d` |
| `layers/moe/ep_moe/layer.py` | `8d79f68e2562828dc0639c54a6c8fc9f021e0400dc28e880b3278919a0bd3677` |
| `layers/rotary_embedding/base.py` | `99225d71a941674a5f4385ddf1330fc5d57172e808e3e0c2ce7438579862fb52` |
| `layers/attention/dsa/dsa_indexer.py` | `873f570a14d02a4ece10e7405cc65b629717fc708c70ef3e762b84753597134c` |
| `packages/sglang/kernels/ops/attention/utils.py` | `4d56760533c5820102f4a2bd47844f9e2a06adb9aa76a1d042f488565347fff1` |
| `packages/sgl_kernel/elementwise.py` | `3c3d447f5c881182910b2a38cca2361cb1d571cb50c94212ff452b95179b851e` |

## 本地检查

按 `.workflow/l0-gate.yml` 执行 migration plan、guide snapshot 和 L0；按既有 CI 版本执行 Ruff 0.12.1（E9/F63/F7/F82，ignore-noqa），另做修改 Python 文件语法解析与 git diff --check。用户未要求单测，本轮不新增、不执行运行测试。结果：Ruff 与 7 个修改/新增 Python 文件语法解析通过；L0 通过（566 个 Python 文件、8 个 skills、0 error）；migration/guide 快照与 git diff --check 通过。
