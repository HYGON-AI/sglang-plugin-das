# DeepSeek V4.1 W4A8 Prefill：空泡修复与 DAS 对照（P18）

采集：2026-11-16 UTC；整理：2026-11-17。对照用户 P09 截图及历史 DAS trace，修改落在 P 节点 `/home/sglang/sglang-plugin-das`。adapter、D 服务、router及原启动预算均未改变。

## 1. 结果与交付边界

**七类已定位的冗余准备、阻塞拷贝和分配问题已修复；不能将当前结果称为全部 step 间隙均已对齐 DAS。** P18 固定同组快测 **20/20**，不开 profiler 的三轮共 **384 请求、0 错误**。输入吞吐中位数 **55,114.928 token/s**，相对 P09 同条件顺序复测 **+5.14%**；32K→1 单请求 TTFT 中位数 **653.386 ms**。

LayerSplit 后半段 19 个 attention 层的 GPU 包络中位数 **46.941→38.365 ms（-18.27%）**；历史 DAS 为 **36.800 ms**。调用数和原生路径已恢复，但整模型仍有版本、实现和控制路径差异，不能声称严格端到端性能相等。

原先 492 MiB logits 冷分配引发的约 **44.155 秒 GPU step** 已在全新 P17/P18 模型进程中消失。P18 三个窗口全部 8 rank、共120个GPU step包络均无超过1秒的GPU事件；rank4各窗口五步约 **594–642 ms**。

**仍有 CPU 控制平面空档。** 首次稳态 rank4 四个真实未覆盖间隙为 **0.322 / 0.324 / 0.321 / 31.945 ms**；独立复核为 **33.868 / 9.793 / 13.476 / 0.330 ms**。慢值均保留。它们对应请求准入、真实跨 rank 共识、分页和元数据准备，不能删掉必要屏障或换窗口后说全部消失。具体到达偏斜见第4节；OS/GIL/profiler分别贡献多少尚未单独证明。

本轮没有重跑完整 HumanEval/GSM8K/MATH 数据集；20题快测只用于当前性能修改回归，不授予新的完整数据集精度分数。

## 2. 固定条件与采集口径

- P：a12r4n16；D：a14r4n13；容器 `sglang_deepseek_V4.1_LC_adapter`。只重启自有 P 服务；D/router保持原 PID/start_ticks。
- 原启动脚本 SHA256：`2c4655a103679d40d3f5b8bc1575fa2f297b0bd83d185731da9a3466379f3dc9`；实际 bench 源 SHA256：`f288a12ec4ede07fc81ac50c09fc856ed5825bfc4be30cc9b1567846c4615986`。
- TP8/CP8/EP8、LayerSplit、bounded replay、DSPARK、MegaMoE不变；seed55965328、chunk/max-prefill32768、mem0.85、KV262144、max-running32；Prefill Graph disabled。
- 同128条完整32768输入IDs→1输出，客户端并发128；调度batch以trace为准。三轮无prof输入逐SHA与P09一致；单请求另测三次。原脚本已有 logits free-memory fraction `.05`，没有新增或调大预算。
- CPU+GPU profiler，start_step2/num_steps5、with_stack/record_shapes=True，全部8rank。burst为先arm再发128；steady及repeat均为同128先完成16条后arm，不flush cache。三窗口都保留，repeat用于核查残余慢值，没有替换第一次结果。
- 真空档为相邻 GPU step 包络之间扣除所有stream的kernel/memcpy/memset区间并集。CPU inclusive时间、单流空白不等于全GPU空泡；包络重叠时区间为空，间隙计0，重叠量单独保存。
- 文件名 TP4/EP4 表示rank4，不是TP4配置。后台Mooncake queue/socket等待本身不证明主流阻塞。
- 历史DAS原始trace SHA256：`a72bd48e9e176834e519bb9bb59c71324feab2b06f3b44682b489a1c27dbcf1c`。它是路径/时序参考，没有本轮重启锁定为交替A/B；阶段间小差异不机械累加为收益。

## 3. 原因、修改与验证

| 位置 | 原因 | plugin修改及边界 | 实际命中 |
| --- | --- | --- | --- |
| LayerSplit后半层Extra准备 | 同forward/source/ratio的完整历史payload重复选页和广播 | 复用已有效的完整历史payload；每层remap保留。新forward、KV写入、source/ratio变化和staging覆盖失效；不复用SWA/Graph | 每rank五步active-pages **200→30**，compact-broadcast **390→220**，与DAS调用数相同 |
| step结束SWA索引 | `.cpu()`回读在主线程等待先前GPU工作 | 自有pinned快照+当前流异步copy/event；原后台sender在既有依赖后resolve，保留abort/source-drain | 目标主线程阻塞DtoH为0；必要传输仍存在 |
| FullComponent分裂 | 大prefix索引clone触发hipMalloc | HCU CP连续1D GPU整数索引才用长prefix view+独立suffix；限制backing<=2×prefix，其他情况原路径 | 目标Full split大prefix clone/hipMalloc为0；没有给KV payload做错误别名 |
| `prepare_for_extend`固定哨兵 | host `[-1]`构造触发8B阻塞HtoD | 设备`torch.full((1,),-1,int64)`；动态输入不动 | 该位置8B阻塞HtoD为0，eager/Graph/stream等价已核 |
| capture层CPU共识 | 不变的layer IDs每步重复object gather | 每步仍做全rank changed-flag MAX；变化回完整gather，冲突不缓存，group/model绑定，独立返回list | P18每五步object gather **0**、flag归约 **5**；init_new中位burst2.834/steady2.392ms |
| dense MQA冷分配 | layer20在持久MegaMoE执行期间首次申请492MiB，P16记录43.557秒CPU阻塞 | 私有native out入口+构造期有界FP32工作区；原device数学不变，当前流复用，其他流/不支持配置回原路径，无forward扩容 | P17/P18全部rank每五步新out **25**、旧dense入口 **0**；冷44秒长事件消失 |
| bootstrap单rankCPU空归约 | ready/prefix分别在TP1和CP多rank组都归约 | 仅HCU/CPU/Gloo单rank轴且peer轴为多rankGloo时省略identity；真实多rank归约、MIN-ready→读prefix→MAX-prefix顺序保留 | 每完整scope实际all_reduce **4→2**；完整模型及真实Gloo四进程32case/rank通过 |

### native out来源与数值

实际供应源取自 `/home/sglang/lightop` commit `c4811f65e9724d978cfd5cb779faff8a2d7a1307` 的四个自包含文件；保留Apache/SPDX/LICENSE及ORIGIN。已安装LightOp为 `0.6.0+dtk2610.torch2130.2611122137.gf4f79d`，不把源码commit冒称安装binary身份。仅隔离namespace、新增必需out入口、host侧验证和fill；device数学/launch逆变换回原字节。

实际Torch distribution：`2.13.0+dtk2610.2611091606.ge8aa84`；runtime版本单独核验。使用实际Torch COMMON_HIP_FLAGS/HIPCC_FLAGS、torch_hip/c10_hip、gfx936编译，未用假的兼容头。

- 原生27合同：14形状、边界/candidate/padding/空及全invalid/canary/alias/3次Graph/跨流，与实际已安装入口输出 **rtol=0、atol=0**；独立reference沿用原容差3e-3/3e-2。
- 工作区9合同含实际4096×32768、width31488；原前后预算行数相同，工作区无热路径增长；CPU dispatch15及实际packaged native5通过。
- 热算子trace有6个kernel，目标域无aten::empty/hipMalloc。构造期logical workspace为 **515,270,244B/rank**，分配block492MiB，已计入原logits预算。不能在workspace存在后重复扣减预算。
- 当前加载SO SHA256：`2f8daa3dac837e0337de497863797c276d60a9967ed7e10517ff6d9f209fc426`。8个实际scheduler均核maps；FlashMLA既有兼容SO `d0d4018a4e6d63573e0b55b9cd435320415c41f4ac94b37bc8bdd1403cb7bd7b` 未换库。
- P18 CPU wheel `21471f16c8bab4124c0fdc225b601f03af07caac2dd685226a18ca64fbcf0c0c` 已逐RECORD/当前运行源核验，只授予该构建范围；最终补充文档另外按交付source policy绑定。

ORIGIN中的未构建字段是准备时历史快照，不回写已消费来源。实际资格以独立native证据和当前加载hash为准。

## 4. 全窗口间隙及剩余原因

| rank4窗口 | 四个真实未覆盖间隙/ms |
| --- | --- |
| 历史DAS | 0.300 / 0.297 / 0.296 / 0.313 |
| P09 burst | 101.892 / 128.190 / 40.362 / 43.772 |
| P17 steady | 31.751 / 39.645 / 8.855 / 0.315 |
| P18 burst | 104.652 / 97.478 / 37.002 / 0.000 |
| P18 steady | 0.322 / 0.324 / 0.321 / 31.945 |
| P18 steady独立复核 | 33.868 / 9.793 / 13.476 / 0.330 |

P18 steady全部8rank的前3个间隙为 **0.301–0.327ms**，第四 **31.661–32.184ms**；复核全部间隙范围 **0.307–33.973ms**。不能只引用0.3ms，也不能把阶段性差异直接归为bootstrap改动收益。

首次steady第四个包络间CPU区间约33.368ms：rank4的waiting-queue CP poll约8.495ms、hidden-layer flag共识约8.415ms、bootstrap ready/prefix约5.539ms，另有分页/组batch及元数据准备。以上是嵌套inclusive scope，**不能直接相加**。8rank共同时间轴显示hidden-layer共识rank2约30.974ms才提交，其他rank约23–24ms提交；rank2在此前`prepare_mlp_sync_batch_raw`有约6.663ms Python self区间。明确有到达偏斜，尚不能从这些scope单独区分OS抢占、GIL或profiler记录成本。

独立复核的慢值换到前3个间隙；调用链仍为准入/组batch/真实CPU共识，最长约33.868ms。最后一步即使有27.316ms的late-tail HtoD CPU API，真正全GPU未覆盖间隙仅0.330ms：长API可能在等待此前GPU工作，不能据CPU条长推算可以回收27ms。

已只读核实实际官方 `_all_reduce_polls` 自带`world_size==1`返回分支，因此没有再添加无效hook。bootstrap的两阶段屏障和hidden-layer changed vote都保留；若继续减少这部分开销，需要独立设计控制平面重叠/协议，并证明动态状态、空rank、group/owner及异常语义，不能通过关闭检查或改变默认配置来美化trace。**“所有step始终保持DAS约0.3ms间隔”尚未达成。**

burst前两个空档还包含ingest与大请求对象的跨rank广播/反序列化；这是另一个控制路径，当前没有为此修改adapter。没有把新请求接入空档误算成后半层attention kernel空泡。

## 5. LayerSplit后半层对照

当前按实际21..39共19个attention consumer层统计，层20的source/owner准备另保留。GPU跨度取相关launch的首末区间，包含重叠及等待，不是逐kernel耗时求和。

| 窗口 | 单层CPU中位/ms | 19层GPU包络中位/ms |
| --- | ---: | ---: |
| 历史DAS | 4.366 | 36.800 |
| P09 | 6.230 | 46.941 |
| P18 burst | 4.709 | 38.373 |
| P18 steady | 4.624 | 38.365 |
| P18 steady复核 | 4.964 | 38.440 |

P18 steady五步19层GPU包络/ms：`38.633 / 38.575 / 38.268 / 38.365 / 38.211`。全部24份raw每rank每窗口都有95个后半层scope；BF16稀疏attention105、FP8稀疏attention95、新dense out25、旧dense数学入口0。原LightOp namespace下35个是gather/query-lens/prefill-metadata，不能误数为旧dense路径回退。

## 6. 无profiler性能与失败保全

| 阶段 | 同组快测 | 输入吞吐中位/token/s | 单请求TTFT中位/ms |
| --- | ---: | ---: | ---: |
| P09 | 20/20 | 52,422.368 | 657.535 |
| P13 | 20/20 | 55,212.892 | 643.539 |
| P16 | 20/20 | 55,155.501 | 653.603 |
| P17 | 20/20 | 55,182.362 | 642.326 |
| P18 | 20/20 | 55,114.928 | 653.386 |

P18三轮吞吐原值：`55064.53539249981 / 55182.90234578713 / 55114.92846243156`；single秒：`0.6583578223362565 / 0.6533858152106404 / 0.647642576135695`。各轮128请求、0错误；相对P17约-0.12%，没有证据宣称最后的单rank空归约优化额外提高吞吐。**累计+5.14%相对P09，不能把局部trace加速当端到端收益。**

P14在ready前取消、0请求；P15异步tail元数据试验虽过独立数值与19/20快测，实际prof watchdog失败、无有效prof/perf，未交付该路径。当前保留P13 GPU完成语义，P17仅加入有界输出能力。旧负包及P16 cold44秒记录保持，不用后一个成功覆盖失败。

P17 standalone build01缺少真实Torch HIP编译flags，失败保留；build02才按实际flags成功。P18真实Gloo四进程32case/rank及repo packaged同32复验均通过。所有已消费的模型/测试入口未重发；复核capture是新具名诊断任务，原完整窗口保留。

## 7. 源码、回退与产物

- 节点源码：`/home/sglang/sglang-plugin-das`。新native入口：`hcu_sglang/models/deepseek/current_v4/hcu_backend/dense_mqa_out.py`；native源位于`hcu_sglang/jit_overlay/csrc/dsv41_mqa_out/`。
- CPU共识：`hcu_sglang/disagg/hidden_capture_consensus.py`、`bootstrap_vote.py`；持久化回归位于`tests/runtime/pd_hidden_capture_consensus.py`、`bootstrap_vote.py`、`dense_mqa_output_{dispatch,native}.py`以及既有async/view测试。
- P18运行policy：`prefill-opt-change18/policy.json`，SHA256 `fb58e6b398fe02c9bde725896c19149999a607c113d6a5289682ccb7b277325d`，53源。最终交付policy另绑定报告/guide更新，运行源和测量前文档的原始字节另存，不混写身份。
- 回退bootstrap省略：按`prefill-opt-change18/before`恢复该轮prefill方法，保留之前修复并重启P复验。回退native out须成套恢复P17前的backend/预算helper和模块引用，退回P16已验证路径，**会恢复cold大分配风险**；不能退到P15失败试验。不修改在运行进程的模块对象。
- 未stage/commit/push。没有改变全局平台、LightOp安装包、adapter或D/router，没有调整cache/flags/负载掩盖问题。

raw rank4（其余rank在对应`all-rank-traces`）：

| 窗口 | 文件 | SHA256 |
| --- | --- | --- |
| burst | `doc/prof/20261116-prefill-opt18/1794866956.723114-TP-4-EP-4.trace.json.gz` | `784a44631d4a30f49acbd48d5ffb0a693e4a813dc02537f32ab0ef0475e0ec36` |
| steady | `doc/prof/20261116-prefill-opt18-steady/1794867597.2528102-TP-4-EP-4.trace.json.gz` | `61bee165dddd0e42f5817d4feabb5e184e3eef7d964addfda6158b82d873c997` |
| steady复核 | `doc/prof/20261116-prefill-opt18-repeat/1794869109.4044926-TP-4-EP-4.trace.json.gz` | `a117e0d815e1dd83ddce18f80e7b9b2ae434f26c993d9a9ad32bb520dd0da762` |

三窗口全部24raw已逐SHA/size、完整JSON及gzip校验；源码/精确20输入响应重评分/384输入SHA/native父证据/最终检查均由独立本地verifier复核。正式性能不取profiler扰动数据。尚未得到的是全控制平面空档消失、完整新数据集验收和严格历史DAS端到端A/B结论。
