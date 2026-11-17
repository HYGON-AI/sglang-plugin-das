# DeepSeek V4.1：P 端 FP8 paged 原生路径修复与新 prof

**修改和重抓已完成：plugin现在与DAS一样，在5个32K Prefill步骤中实际执行105次BF16 Sparse Prefill和95次FP8 paged原生kernel；8个rank均无Torch paged回退。** 本轮只改P端后端选择，不改adapter、attention数学、KV布局或D服务。

## 修改

节点`/home/sglang/sglang-plugin-das/examples/deepseek_v41_w4a8/prefill.sh`在source共享env之后仅增加：

```bash
export SGLANG_HACK_FLASHMLA_BACKEND=kernel
```

共享`env.sh`保持原Torch设置供D使用。实际P启动脚本`/home/plugin_adapt/redevelop-20261114-node-switch/prefill32k-megamoe-compare01/flashmla-model07/p_megamoe_plugin.sh`同步使用这一选择，并保留`SGLANG_HCU_SPARSE_PREFILL_BACKEND=kernel`。相对P06，完整CLI和其它环境逐项不变；seed55965328、TP8/EP8/CP8、LayerSplit/bounded replay、mem=.85/KV262144/running32、chunk32768、MegaMoE/DeepGEMM/BLAS/DSPARK均保持。

原问题是Prefill复用了名为`_forward_flash_mla_decode`的paged接口。只切换独立BF16 Prefill开关仍会让另一支走Torch；修复后这个helper名称仍存在，实际执行已是原生FP8 kernel，不能再按名称判断请求阶段。

## 新 prof 路径命中

| TP4同5步 | 修改前plugin | 修改后plugin | 原DAS |
|---|---:|---:|---:|
| BF16 Sparse Prefill native | 105 | 105 | 105 |
| FP8 paged native | 0 | 95 | 95 |
| Torch paged函数 | 95 | 0 | 0 |
| cache accumulator | 190 | 0 | 0 |
| Torch KV unpack | 950 | 0 | 0 |
| GPU step包络中位数(ms) | 1688.308 | 863.002 | 506.012 |

TP4包络相对修改前缩短48.88%。 当前863.002ms仍高于DAS历史506.012ms约70.55%；attention路径一致不代表整步性能已经对齐。FP8 paged kernel五步累计10.122ms，原DAS 9.203ms；BF16 kernel当前累计263.237ms。

| rank | 修改前GPU step中位数(ms) | 修改后(ms) | 当前BF16 / FP8 / Torch调用 |
|---|---:|---:|---:|
| TP0 | 1688.148 | 863.244 | 105 / 95 / 0 |
| TP1 | 1688.642 | 863.394 | 105 / 95 / 0 |
| TP2 | 1688.202 | 863.251 | 105 / 95 / 0 |
| TP3 | 1688.136 | 863.140 | 105 / 95 / 0 |
| TP4 | 1688.308 | 863.002 | 105 / 95 / 0 |
| TP5 | 1688.132 | 863.236 | 105 / 95 / 0 |
| TP6 | 1688.200 | 862.746 | 105 / 95 / 0 |
| TP7 | 1688.228 | 863.297 | 105 / 95 / 0 |

全部rank均固定5个32K EXTEND步骤。GPU包络按匹配External id跨stream取最早开始至最晚结束，不是kernel时间求和。TP4另用完整JSON流解析器独立核实每步21 BF16+19 FP8共40次provider调用均在CPU EXTEND窗口中，避免仅依赖快速扫描器。

## 数值与请求验证

- 原生FP8 paged独立13项通过：H64/D512、584B KV、SWA+extra、ragged、负索引、masked/length NaN、无效候选、stride、不同query数，以及3次Graph KV/query更新；保持rtol=.02/atol=.004，最大绝对误差0.001953125。
- 实际trace里的两份KV形状为`[596,256,1,584]`和`[1025,256,1,584]`，页大小/布局与测试相同；整模型请求同时覆盖实际大cache页数。
- 新P模型8scheduler映射同一私有兼容SO，server effective配置与P06无差异，router连续健康90秒后才进入请求。
- 相同20条GSM8K快速检查为**19/20**；128条固定完整32768 input IDs/1输出全部完成。此快测不替代完整数据集精度，不继承旧全量分数。 修改前同组快测为20/20；本轮sample12标准答案13、实际回答12，请求payload逐项相同，原始响应已保全。这个小样本相对下降为5%，不能代替完整精度结论，也未据此确定错误根因。
- bash语法/L0/guide build/check/default git diff check均通过。旧P06精确owned-stop/no survivors；D/router未重启。

## 采集参数与性能边界

```json
{
  "start_step": 2,
  "num_steps": 5,
  "activities": [
    "CPU",
    "GPU"
  ],
  "with_stack": true,
  "record_shapes": true
}
```

输入32768、输出1、128请求，沿用原`prefill_throughput.py`及前版完全相同input IDs。profiled吞吐/耗时只作为采集记录，**本轮没有新增无profiler正式性能测试**，不能把旧P06的25855.67token/s继承给新版本。DAS历史trace的输入IDs未知且Torch/HIP版本不同，整步仍有其它计算/通信差异，不据此声称完整性能已对齐。

私有SO仍为`d0d4018a4e6d63573e0b55b9cd435320415c41f4ac94b37bc8bdd1403cb7bd7b`，原全局SO保持`7ad05bfbb2fa67148b3e6bd6cf288f54ea1baa074438dc457d9d69b90bb92301`。它是固定gfx936/当前Torch与DTK的ABI-tag实验兼容包，不是供应方源码重编。资格仅授予本轮实际P端路径；D端和其它模型/硬件未扩展。

## 文件和回退

- 新TP4 trace：`doc/prof/20261116-flashmla-paged-kernel-prefill32k/1794816376.5560822-TP-4-EP-4.trace.json.gz`，SHA256 `66c951b1869d7d7cbd91723733ea02e749a2bd7a2c43e1ee521217f8be6a907e`，可直接拖入Perfetto。
- 全部8rank位于节点`/home/plugin_adapt/redevelop-20261114-node-switch/prefill32k-megamoe-compare01/flashmla-model07-profile01/traces`；本地`capture-metadata.json`记录完整清单。
- 本地完整核验：`doc/prof/20261116-flashmla-paged-kernel-prefill32k/local-proof01.json`。
- 原始证据ZIP：`flashmla-model07-evidence01.zip`，SHA256 `648667350225c4c570c2de904ee8cc1c724776e9a37811a43c972e4e66ba4bf0`，124成员。
- P recipe前后SHA：`391e8a19b354f06f83fb523d2b99e1f7797aaaa089f1cae388c1b47686d72029`→`d512c681dfae0656bb03fb0c4b8aa203a37b376f180341099c45cb19abaa6809`。Python runtime源与测试前相同。
- 若需回退，仅P端改回`SGLANG_HACK_FLASHMLA_BACKEND=torch`，按实际owner确认后重启P；不对已加载SO热替换，不动D。旧P06脚本和prof保持完整。
