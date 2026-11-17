# FlashMLA gfx936 原生稀疏 Prefill：容器修复与实测

当前容器已经可以实际调用原生 FlashMLA 稀疏 Prefill。实际整模型的8个scheduler均映射私有兼容SO；32K输入/1输出的5-step trace出现 **105次原生Sparse Prefill kernel**，不再使用这一阶段的Torch/Triton回退。paged decode仍保持Torch，不在本次原生资格范围。

本轮是固定环境内的实验兼容部署：仅改28个gfx936 code object的ELF ABI标记，ISA及其它字节完全不变。**不是源码重编、供应方正式wheel或通用V6→V5转换方案，也没有发布新Docker镜像。** 原始安装包保留；当前P服务通过PYTHONPATH选择容器内私有包，adapter源码没有改动。

## 根因与修改

原FlashMLA wheel含gfx936以及本次BF16/D512所需kernel，并非未编译该架构。其28个gfx936对象标记为CodeObject V6；当前libgalaxyhip对原对象moduleLoad返回0，但同一kernel的getFunction返回500。私有副本仅将ELF byte8从4改3后，函数解析成功，继而完成真实kernel执行和下述验证。运行库内部缺陷的源码位置尚未取得。

- 原SO SHA256：`7ad05bfbb2fa67148b3e6bd6cf288f54ea1baa074438dc457d9d69b90bb92301`。
- 当前私有SO SHA256：`d0d4018a4e6d63573e0b55b9cd435320415c41f4ac94b37bc8bdd1403cb7bd7b`。
- 私有site：`/opt/dsv41-redevelop-20261114-switch/providers01/flashmla-gfx936-v5compat01/site`。
- plugin增加独立的`SGLANG_HCU_SPARSE_PREFILL_BACKEND=kernel`。公开调用宽度不是128倍数时，仅补无效索引-1；length外索引先置-1，保证不引入有效候选。
- 全局`SGLANG_HACK_FLASHMLA_BACKEND=torch`保留，避免把未验证的原生paged decode一起启用。

## 验证结果与边界

| 验证 | 实际结果 | 范围 |
|---|---|---|
| 原生数值/Graph | 17项通过；原rtol=.02/atol=.004不变 | finite、ragged、mask/NaN、sink、stride、Graph三次更新和32K形状，topk128/640/2048 |
| plugin公开入口 | 64项通过，52次实际Python native调用 | H8/H64、width2/65、NaN/长度mask及Graph；Graph replay不重复进入Python计数器 |
| 整模型加载 | ready，8个scheduler均映射私有SO | P为TP8/EP8/CP8；D/router未重启 |
| 整模型快速正确性 | 20/20 GSM8K | 相同20个prompt与评分，不能替代完整数据集验收 |
| trace负载 | 128/128请求完成，8rank各5步，gzip完整 | 与上一版完全相同128组32768 input IDs、1输出 |
| 无profiler性能 | 三轮384/384请求完成 | SSE完整，逐请求prompt/completion token数与耗时检查通过 |

本轮没有重新运行完整HumanEval/GSM8K，不能继承旧版本得分称新全量精度通过。只验证当前BF16稀疏Prefill调用形状；未授予FlashMLA所有dtype、decode/FP8或其他硬件/DTK版本支持。

## 32K性能对比

| 指标 | 上一版Triton Prefill | 当前原生FlashMLA | 变化 |
|---|---:|---:|---:|
| TP4五步GPU包络中位数(ms) | 1797.297 | 1668.576 | 缩短7.16% |
| 三轮无profiler输入吞吐中位数(token/s) | 23,582.28 | 25,855.67 | +9.64% |
| Sparse Prefill kernel累计(ms/5步) | 611.063 | 263.223 | 缩短56.92% |

当前全部8rank的GPU包络中位数范围为1668.372–1668.685ms。这里的GPU包络是匹配External id后的最早开始到最晚结束，不是跨stream时长之和或纯计算时间；kernel累计也可能受排队/同步影响。

同名原生Sparse Prefill kernel的当前累计263.223ms，相比历史DAS的265.154ms差-0.73%。这一选定kernel已接近历史耗时，但不能据此推定整模型性能对齐。当前W4A8主kernel为200次、累计2724.687ms，历史为200次/1038.115ms；剩余差异应优先检查其计算与通信等待范围，不能把全部device elapsed当纯GEMM计算时间。

用户历史DAS trace的TP4包络为506.012ms，当前相对它为3.298倍。两侧HIP版本60326255/70226365不同，历史输入IDs未知；历史42864.75token/s采样被profiler扰动。因此不把历史数值当本轮双端无profiler正式基线，也不将全部差异归因于plugin。

| 无profiler轮次 | 输入吞吐(token/s) | 128请求总耗时(s) | 平均TTFT(s) | P99 TTFT(s) |
|---|---:|---:|---:|---:|
| 1 | 25,810.85 | 166.434 | 84.630405 | 161.918620 |
| 2 | 25,855.67 | 166.158 | 84.472501 | 161.813224 |
| 3 | 25,946.62 | 165.602 | 84.218041 | 161.146963 |

独立单请求32K→1 TTFT三次为1.317911s, 1.291658s, 1.320863s。所有请求IDs、控制器和生成参数与上一版逐字节核同；effective server配置相同，原自动seed55965328在新启动中显式固定。预算mem=.85/KV262144/running32、chunk32768、TP8/EP8/CP8、MegaMoE/DeepGEMM、新BLAS、DSPARK gamma5/verify6均保持。

## 使用和回退

当前P启动记录在`/home/plugin_adapt/redevelop-20261114-node-switch/prefill32k-megamoe-compare01/flashmla-model06`。激活脚本为私有site上一级的`activate.sh`，它只设置当前shell的PYTHONPATH和两项Prefill/paged后端开关，不替换全局安装包。

回退需从启动环境移除私有provider路径，设Prefill后端为`triton`，再按实际PID/PGID/start_ticks核归属后重启P；不能对运行进程热换已加载SO。不要把该私有包无条件推广到其它Torch/DTK/gfx版本。

本次新增运行行为在节点`/home/sglang/sglang-plugin-das/hcu_sglang/layers/attention/debug_flash_mla_adapter.py`与对应测试中，未改adapter。CPU dispatch7、L0=630Python/8skills、导览构建与freshness检查通过。此前导览截图因浏览器安全策略未完成，不作为运行验证证据。

## 证据

- 原生trace：`work/deepseek-v41-w4a8/20261114-node-switch/prefill32k-megamoe-compare01/flashmla-model06-trace-local01/1794801469.1026533-TP-4-EP-4.trace.json.gz`，SHA256 `31df9c648674e057836576c33a6f011a5b62e466362d05584dc98cebf33de0f3`。
- 整模型证据ZIP SHA256：`45e13e703b723c82888b3f5b9493667a1e4295a90f1d7ee21c9264994138d71f`，138成员；本地逐成员/20原始回答/384计时与完整IDs/8rank来源核验通过。
- 本地核验：`work/deepseek-v41-w4a8/20261114-node-switch/prefill32k-megamoe-compare01/flashmla-model06-local01/local-proof01.json`；原生kernel摘要：`work/deepseek-v41-w4a8/20261114-node-switch/prefill32k-megamoe-compare01/flashmla-model06-local01/trace-summary01.json`。
- 原生17项和公开64项证据父包：`118c49014d97b70df3c224eaca24c403f857fb3d5561fd8d00ee1fc9c1800fbc`。
- 先前Triton报告和失败用例保留。原公开width2/65断言、首启动端口占用拒绝等不被后续通过覆盖。
