# DeepSeek V4.1 W4A8 Prefill：step 间与 LayerSplit 空泡修复（P16）

采集：2026-11-16 UTC；整理：2026-11-17。任务是复现用户提供的 P09 Perfetto 截图，在节点 plugin 源码修复已核实的依赖，保留真实失败，再对照历史 DAS trace。

## 1. 本轮结果

P16 同组快测 **20/20**（固定门槛19/20），无 profiler 三轮共384请求错误0。输入吞吐中位数 **55,155.501 token/s**，相对P09顺序复测 **+5.21%**；32K→1单请求TTFT中位数 **653.603ms**。这些是不开profiler的数据。完整GSM8K/HumanEval等没有在本轮重新运行，不能把20题快测当完整精度验收。

**稳态两类目标空泡已恢复到原生量级；突发首次采集的492MiB冷分配长等待尚未修复，不将所有场景声明完全正常。** 详见第4节。

已消除截图中确认的主线程 SWA 索引阻塞回读、Full 索引分裂的大块 prefix clone 分配和固定哨兵8B阻塞上传；LayerSplit 后半段 Extra 重复选页/广播准备已复用。新增 CPU 共识复用的实际调用见第5节。保留必要协议等待，未靠降低负载、清cache或隐藏慢rank使trace变好看。

## 2. 固定条件与比较边界

- P：a12r4n16；D：a14r4n13；容器 `sglang_deepseek_V4.1_LC_adapter`。仅P节点 `/home/sglang/sglang-plugin-das` 变化，adapter、D、router保持。
- 原 `p_megamoe` 配方脚本SHA256 `2c4655a103679d40d3f5b8bc1575fa2f297b0bd83d185731da9a3466379f3dc9`；完整argv和env见归档preparation。TP8/CP8/EP8、LayerSplit、bounded replay、DSPARK、MegaMoE保持；seed55965328、chunk/max-prefill32768、mem0.85、KV262144、max-running32；Prefill Graph disabled。
- FlashMLA沿用此前单独验证的私有兼容SO，SHA256 `d0d4018a4e6d63573e0b55b9cd435320415c41f4ac94b37bc8bdd1403cb7bd7b`；这是既有实验兼容产物，不是本轮供应方重编或全平台资格，本轮未换库。
- 同128个完整32768输入IDs、输出1、客户端一次并发128（调度batch以实际trace为准）；原bench实现、三轮无prof输入和采样条件不变，不flush cache。用户实际原流程是在bench运行到32K时调用start_profile，故另抓稳态窗口，不替换原突发窗口。
- profiler活动CPU+GPU，start_step2/num_steps5，全部8rank。文件名TP4/EP4是rank4，不能误读为TP4配置。
- 历史DAS原始traceSHA256 `a72bd48e9e176834e519bb9bb59c71324feab2b06f3b44682b489a1c27dbcf1c`。它用于路径/间隙对照；历史启动版本、库和输入身份与当前不是重新锁定的交替A/B，因此不宣称严格的DAS端到端性能对齐。
- 各阶段为同条件顺序复测，存在运行波动；阶段增益不机械累加，不把单次profile包络缩短换算为正式吞吐收益。

## 3. 原因与落地修改

| 位置 | 已核实原因 | plugin最小修改与语义边界 | 新trace核验 |
| --- | --- | --- | --- |
| 后半层Extra/LayerSplit | 同forward、同source和ratio的完整历史payload被重复选页和广播准备 | P10复用有效完整历史payload；每层remap保留，KV写入、新forward/source/ratio和staging覆盖均失效；不将SWA或Graph数据混用 | 5step每rank active-pages调用200→30，compact-broadcast390→220，与历史DAS调用数一致 |
| step结束的 `_swa_payload` | `kv_to_page_indices.cpu()`的8B回读挡住主线程，CPU API可等待此前GPU工作 | P11自有pinned快照+当前流异步拷贝+event；既有后台sender在原依赖之后resolve，保留abort/source-drain和生命周期 | 主调度线程该位置阻塞DtoH为0；拷贝仍存在，未宣称删除必要传输 |
| FullComponent节点分裂 | `clone`大prefix索引引发hipMalloc；P09例为2MiB/30.276ms | P12只对HCU CP、连续1D GPU整数Full索引采用长prefix view+独立suffix；限制backing<=2×prefix，小prefix/大backing/其他pool回原路径 | Full split目标位置大prefix clone/hipMalloc为0；不是KV payload共享视图 |
| `prepare_for_extend` | 常量 `[-1]`由pageable host构造，8B HtoD阻塞 | P13设备端 `torch.full((1,),-1,int64)`；动态输入和其他prepare逻辑保持 | 该位置8B阻塞HtoD为0，独立eager/Graph/stream等价已过 |
| `forward_batch.init_new` | CPU Gloo每步object gather重复收集未变化的capture层；P13对应约5.3ms控制开销 | P16每步仍做全rank CPU changed-flag MAX；任一输入/group/model变化走原完整gather，冲突不缓存，结果返回独立list；非HCU/非Gloo/default group回原路径 | 真实CPU双rank14项及本次完整模型/trace验证；没有改GPU完成或通信流依赖 |

Mooncake传输线程的queue/socket wait在截图中处于后台，不能仅凭长条归因主流阻塞。同名RCCL Generic kernel无法从名字区分全部collective类型，故以实际Python callsite、correlation和rank一致性核实。

## 4. step间隙：突发与稳态分开报告

计算口径：相邻GPU step包络之间，扣除**所有stream的kernel/memcpy/memset区间并集**后剩余时间。CPU inclusive时间、单stream空白或重复step标签不能直接当全GPU空泡。 相邻GPU step包络若重叠，该包络间区间为空，间隙记0，并另存重叠量；不能报负空泡，也不能仅凭包络重叠宣称内部所有GPU时间都被kernel覆盖。

| rank4窗口 | 四个真实未覆盖间隙/ms |
| --- | --- |
| 历史DAS | 0.300 / 0.297 / 0.296 / 0.313 |
| P09 突发 | 101.892 / 128.190 / 40.362 / 43.772 |
| P13 突发 | 88.050 / 89.122 / 46.459 / 39.861 |
| P16 突发 | 193.195 / 0.000 / 33.678 / 0.317 |
| P13 稳态 | 17.724 / 16.603 / 0.313 / 0.307 |
| P16 稳态 | 0.311 / 0.309 / 0.317 / 0.322 |

P16稳态全部8rank共32个间隙范围 **0.304–0.330ms**。突发窗口仍包含请求接入、分页分配和控制提交开销；不能以稳态窗口替代突发窗口后声称是代码收益。未移除后端必要的GPU完成边界；长HtoD CPU API可能等待之前的GPU工作，不能把其整段时间视为可回收带宽开销。

### 突发首步异常：单独保留，未声称已消除

P16突发窗口第一步GPU包络为 **44,155.136ms**；其余四步为593.824/611.050/593.561/592.893ms。不是以中位数覆盖首步异常：rank0 correlation21829显示layer20的 `_hcu_dense_fp4_mqa_logits → lightop.fp8_fp4_mqa_logits → aten::full → aten::empty → hipMalloc` 申请 **515,899,392B（492MiB）**，CPU阻塞43,556.558ms；同关联的设备memset记录43,555.800ms。同时所有rank的MegaMoE kernel记录约43,565ms等待；rank3/4 CPU forward约518ms、其他rank约44,100ms，存在明显到达偏斜。

这把问题定位到大logits冷分配与在途MegaMoE执行交叠，**尚未单独证明底层同步/等待机制或完成冷路径修复**，不能无证据归因后台socket wait或FlashMLA kernel。原始突发trace和rank0调用链保全。实际LightOp公开Python API未提供out参数（gemmopt.py SHA256 `6ff95887ca85348c30afdf64fb9f34747ebd00c17e5572bcf4d390b7320271b9`）；本轮没有伪造参数、强行热补丁native或改allocator/缓存政策。

对应用户在已运行bench中抓prof的**稳态窗口**，全部8rank共40步为约593–604ms，没有超过1秒的GPU事件；四个step间隙均约0.3ms。稳态可交付，首次突发采集的这项异常仍是明确限制。若继续处理冷路径，应先提供真实out/workspace能力或在持久通信kernel启动前规划有界workspace，再独立数值和新冷进程验证，不能只为截图增加假请求或清cache。

## 5. 后半层与CPU控制路径

用户所说“后20层”在当前实际路径中以21..39这19个attention层统计；层20的source/owner准备单独保留，未硬编码跳过数学。GPU跨度用相关launch覆盖的首末时间，包含重叠/等待，不是各kernel耗时求和。

| 窗口 | 后半层attention单层CPU中位/ms | 19层attention GPU包络中位/ms |
| --- | ---: | ---: |
| 历史DAS | 4.366 | 36.800 |
| P09 | 6.230 | 46.941 |
| P13 | 4.756 | 38.381 |
| P16 | 4.644 | 38.513 |
| P16-steady | 4.589 | 38.287 |

P16五步19层GPU包络原值/ms：`38.167 / 38.946 / 38.473 / 38.513 / 38.831`；稳态：`38.575 / 38.274 / 38.005 / 38.620 / 38.287`。保留首步慢值；P12曾出现278.606ms首步outlier，未删除或以挑rank规避。

| rank4窗口 | init_new内object gather次数 | CPU flag all_reduce次数 | init_new中位/ms |
| --- | ---: | ---: | ---: |
| P13 | 5 | 0 | 6.233 |
| P13-steady | 5 | 0 | 5.966 |
| P16 | 0 | 5 | 3.549 |
| P16-steady | 0 | 5 | 2.885 |

## 6. 无profiler性能与精度边界

| 版本 | 同组快测 | 输入吞吐中位/token/s | 32K→1单请求TTFT中位/ms |
| --- | ---: | ---: | ---: |
| P09 | 20/20 | 52,422.368 | 657.535 |
| P10 | 20/20 | 53,351.989 | 643.885 |
| P11 | 19/20 | 54,306.371 | 648.644 |
| P12 | 20/20 | 54,325.111 | 656.330 |
| P13 | 20/20 | 55,212.892 | 643.539 |
| P16 | 20/20 | 55,155.501 | 653.603 |

P16相对P13的无profiler吞吐变化为 **-0.104%**。这是顺序复测，不能从小差异证明CPU共识复用带来额外端到端收益；其直接证据是减少重复object通信及profile中的控制开销，累计收益与各阶段结果分别保留。

P16三轮吞吐原值：`[55097.063354184735, 55155.50139211615, 55173.63266059849]`；single TTFT原值/秒：`[0.6597906444221735, 0.639918489381671, 0.6536031924188137]`。各轮128请求，三轮384个输入逐SHA与P09相同，失败0。P11的19/20按原结果保留；没有补测挑分。

## 7. 未采用试验与回退

P14/P15尝试在CPU计算bounded-tail/CP metadata并异步上传。独立metadata数值通过并不足以证明整模型的分布式/stream消费安全：P15快测19/20后真实prof挂起，watchdog终止模型，native栈在HSA BusyWait/hipLaunch/MaskKernel，具体根因未被单独证明。该轮**没有有效prof或性能分数**。负证据ZIP SHA256 `2b71be0945139e4a415c50d4133602a1bdaace44a5e02ca589348ee9c0adf445`已本地逐成员、源码和20响应重算通过。

P16 backend已恢复P13原字节SHA256 `eeb26eab14922603162fb48b38db4a53df612f7ff93460dc9b15cbe3c5901abf`，未交付未通过的异步GPU metadata试验。当前唯一新增项为CPU Gloo共识复用。P16之前的FileStore CPU测试超时、CRLF检查和prelaunch SMI空表门禁失败均保留，各自恢复未冒充模型通过。

如需回退P16 CPU共识：恢复 `prefill-opt-change16/before/hcu_sglang/forward_batch.py` 并移除本轮新增helper，保留P10-P13四项修复，重新启动P服务并复验；不要回退到已拒绝的P15 backend。所有原运行/测试记录和source policy保留，未创建commit或push。

## 8. 产物与复现

- burst: `doc/prof/20261116-prefill-opt16/1794855027.0207932-TP-4-EP-4.trace.json.gz`，SHA256 `0c07bd30a2bc8c38d9791571588b9dad4b165fc325b6a8d2185eef7117d81185`。
- steady: `doc/prof/20261116-prefill-opt16-steady/1794855568.091571-TP-4-EP-4.trace.json.gz`，SHA256 `06ff18b3796c9b497ac373693cabe0316cf65ef5540760bb3bec1f12ec1a9c4b`。

- 其余rank位于对应 `all-rank-traces`；两窗口共16份raw全部本地SHA/size/完整JSON验证。
- 节点源码：`/home/sglang/sglang-plugin-das`；报告：`examples/deepseek_v41_w4a8/PREFILL_BUBBLE_FIX_P16.md`；优化流水：`examples/deepseek_v41_w4a8/PERFORMANCE_OPTIMIZATION.md`。
- P16运行source policy：`prefill-opt-change16/policy.json` SHA256 `61bd6ec60daf24deb7acfe965098210747eab08042e726755f5baf7af3fc6ae1`。最终交付policy额外绑定文档/持久化测试/658个目标目录清单，不把清单数量当整模型覆盖数量。
- 关键检查包括实际HCU的async indices/Full index视图/原HookRegistry回归、设备full的eager/Graph/stream数值、真实CPU双rank共识与同负载模型复验。未恢复旧hostabi/Engram联调，未改变全局平台/adapter或默认flags。
+
+CPU共识测试可在仓库根目录独立复现（不启动模型）：
+
+```bash
+out=$(mktemp -d)
+HCU_ADAPTER_HOST=none SGLANG_PLUGINS=__ifb_no_general_plugins__ +CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= +GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 PYTHONPATH= +python tests/runtime/pd_hidden_capture_consensus.py "$out" +  hcu_sglang/disagg/hidden_capture_consensus.py
+```
+

结论范围是已确认空泡位置的源码修复和本次真实复验。仍有CPU分页/调度、必要同步及历史DAS版本差异，未声称所有空白都可删除或DAS整套性能已经完全一致。
