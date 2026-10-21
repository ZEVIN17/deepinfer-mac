# DS设计与Mac实现对应

本轮目标为生产执行路径中的明确DS机制和同质量竞争优势。下列代码已实现，不代表性能目标已达到；主展示Qwen3.5-9B为混合状态，不能直接使用可trim Qwen3的草稿回退。4B是控制模型，14B为后续控制候选，其结果不能替代9B证据。

## 固定来源与归属

依据[DeepInfer MTPVarlen原始源码](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp)，commit `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`。未修改头保存于 `tests/reference/vendor/mtp_varlen.hpp`，git blob `09af680aa42b724fe86315007a160d5012a1fde9`，SHA256 `164f45db4cccd6d6909dd2846e7dfffb57754b108af679443bf4bd71ec33785a`。Apache-2.0归属见NOTICE和upstream.lock.json。

| 分类 | DS具体方法 | Mac执行对应 | 保留不变量与差异 | 证据 |
|---|---|---|---|---|
| 依据DS设计重新实现 | SamplerRuntimeEstimator，20–42行：9样本滑动中位数、5样本门槛 | `ds/mtp_varlen.py:SamplerRuntimeEstimator` | 窗口和门槛完全对应；Python deque替代C++环数组 | C++ oracle逐次对照含奇偶中位数、窗口覆盖 |
| 依据DS设计重新实现 | ForwardRuntimeEstimator，48–131行：5000样本全局FIFO，按graph残差中位数，再对达10样本的graph中位数等权聚合 | 同模块ForwardRuntimeEstimator | duplicate淘汰、每graph等权、启动0保持；Python列表重算代替平衡multiset，开销独立测 | 5110次oracle输入，含负残差、重复值、不同graph、淘汰 |
| 依据DS设计重新实现 | MTPVarlen，133–275行：expected yield / (启动曲线+残差斜率×graph+sampler)最大化 | 同模块MTPVarlen.schedule / update_runtime | float32累计、lower_bound graph、负斜率截0、严格大于更新、rate相同时seq index优先均保留 | 250组固定种子随机C++ oracle，batch1/2/4、空曲线、无草稿、不同长度 |
| 依据DS设计重新实现 | 两代接受率隔离，N−2选total_q、N−1分配每请求seqlen_q；schedule/process差0或1 | MTPState及schedule两排序队列 | paired时total用prev，不配对时用accept；分配始终accept。不读取本轮尚未验证的token来决定本轮预算 | oracle及明确反例测试；每个DS选择可保存curve/state/budget用于回放 |
| 源码复用 | 上述原始C++头 | tests/reference/mtp_oracle.cpp + shim/sequence.hpp | 头字节未改；shim只替代Sequence/TORCH_CHECK宿主边界，不实现调度。生产不链接C++ | pytest编译C++17并通过stdin协议调用原实现，无GPU/下载依赖 |
| 平台适配 | 为核心机制提供启动性能曲线/接受率/运行反馈 | `policy.DSAdaptivePolicy` | 无Confidence Head，使用仅已完成周期的接受生存频率加Beta(1,1)平滑。两代历史仍隔离；设备执行前choose，输出/rewind后observe推进process | Metal确定性模型检查state计数配对、accept/reject/EOS式close、diagnostics开关 |
| 平台适配 | DS曲线假设MTP前向成本；Mac独立draft有额外成本 | DSAdaptivePolicy初始逐预算实测2次；curve为完整draft+verify+host wall成本 | 不是原DS启动校准器。每个graph第一次校准后固定，只有扩展短请求曲线才重建并清空旧残差。sampler项传0避免host成本重复相加；正残差修正仍用原算法。catch-up额外重放周期不进入稳态curve，单独保留，不从端到端耗时中扣除 | 同循环fixed/legacy/DS消融与完整周期记录；性能结论待真实报告 |
| 独立基础优化 | MLX-LM 0.32.2 generate.py的device-array / async_eval草稿生成 | speculative.py defer_draft_sync | MLX来源，不计DS贡献；combined interval不能拆成伪GPU draft/verify时间；旧同步路径保留测试 | 历史phase2-draft-sync报告、双路径Metal控制 |
| 独立基础优化 | 原生MLX-LM LRUPromptCache的trie/trim | cache.NativePromptCacheStore，engine默认native，exact保留历史消融 | 可trim取公共前缀；hybrid仅精确snapshot+suffix replay。不能宣传成DS缓存移植 | test_native_cache，原生cache对照单列 |

## 生产入口与边界

```sh
.venv/bin/deepinfer-mac run --profile qwen3-4b --mode adaptive --adaptive-scheduler ds --no-cache --max-tokens 128 --json '解释统一内存并给出一个实际例子。'
.venv/bin/python -m pytest -q tests/test_ds_scheduler.py
.venv/bin/python -m pytest -q --run-metal -m metal tests/test_speculative_metal.py
```

等价Python入口为 `InferenceEngine(adaptive_scheduler="ds", round_diagnostics=True)` 和普通 `GenerationRequest(mode="adaptive", ...)`。legacy开关使用同一deferred执行循环、完整周期观测，避免把同步修复算作DS调度贡献。legacy目标-only bucket在60秒后重新校准；DS每64个周期重探到期budget，无永久禁用。

完整周期以主机wall time记录，包含choose、host转换、验证、同步消费者detokenize/emit、draft状态更新和rewind；combined draft+verify只测整体，非分离GPU kernel计时。generator.close/EOS式中断在finally中结算已输出token，不把未输出的预测token当完成工作。诊断关闭仍保留轻量budget/accepted/produced/fallback计数，关闭state/curve回放与逐轮时间字段；正式性能和profiling必须分开。

名称明确为 **DeepInfer MTPVarlen-derived scheduler with empirical acceptance**。没有官方Confidence Head、完整DSpark、连续批处理、多请求GPU批执行；Python核心支持N>1的离线预算验证，当前生产仅N=1。启动探索、经验接受率、独立draft成本属于Mac适配，不能声称与DS完整原生系统相同。

## 性能验收约定

`benchmarks/experiments/ds_budget_ablation.py`固定开发/保留题，使用相同模型revision、草稿、prefill形状、greedy、输出上限、缓存状态和执行循环比较fixed0/1/2/4、legacy、DS；再附native greedy/fixed对照。先用实际native0/1/2/4开发集锁定native fixed预算，保留集不得反向选参数。输出不一致全部保留，不纳入精确等价速度结论；另保留全量任务输出与首次分歧，禁止删失败样本。重复测量按prompt聚类bootstrap，不能把一次prompt的多轮重复当独立题。

当前没有该机制优于两个主要框架的证据。原生持平、相对旧循环改善、模型选择质量提升均不满足竞争优势标准。最终9B需另有真实DS对应执行路径和同模型消融；此文档不提前宣称9B已加速。

## 本阶段实际验证（2026-10-21）

- Portable验证命令：`.venv/bin/python -m pytest -q`。C++ oracle在普通CPU测试中编译/执行，无GPU或下载依赖。
- 实际Metal控制命令：`.venv/bin/python -m pytest -q --run-metal -m metal tests/test_speculative_metal.py`。设备初始化需要允许Metal访问的运行环境；portable通过不能代替Metal证据。
- `benchmarks/reports/phase2-ds-engine-smoke-qwen3-4b.json`：短请求只做6轮探索；长请求96输出tokens、42周期、31次MTPVarlen curve/state/budget实际决策，附完整回放输入。不是只实例化scheduler。
- `benchmarks/reports/phase2-ds-budget-smoke-qwen3-4b.json`：同loop和native控制，保留warmup/development中真实token分歧及首个位置；单heldout中文题全部相同。运行与下载I/O重叠且只有单heldout题，因此不支持性能或竞争优势结论，单cluster interval已设null。
- 旧4B分歧的same-history single/batch logits定位仍待继续。9B已完成下述真实hybrid执行控制；正式开发集调优/保留集统计与两个主要框架的竞争优势尚未完成。

## 主展示9B混合状态适配（首轮硬件正确性通过，正式性能待验收）

新增 `hybrid_speculative.py`，由同一MTPVarlen经验接受率policy决定每轮草稿预算。恢复机制属于**平台适配**：DS原预算算法没有提供当前MLX Qwen3.5 recurrent状态的精确回退实现，不能把此恢复算法宣传成DS源码复用。使用锁定MLX-LM 0.32.2的 `ArraysCache.state`（recurrent tensors/left_padding/lengths）和 `KVCache.state`（keys/values/offset）契约，验证前快照、恢复时给原对象state赋值，保留engine combined-cache列表中对象identity。未调用混合state.trim，也未假设任意删除最后n个token可逆。

状态流：

1. 零预算仅target消费当前gold token；draft网络不执行，已消费输入累积为pending。
2. 正预算快照target/draft容器；首个draft forward合并pending与当前gold，之后顺序提出k个token；target批验证gold+drafts。
3. 全接受且输出完整k+1时保留target已消费状态，draft最后一个accepted输入作为下一轮pending，与下一次首forward合并，不额外执行一个draft网络步骤。
4. 拒绝、输出预算截断或EOS式close时，恢复快照并重放 **gold + 已实际yield的outputs[:-1]**；最后一个输出保留未消费。draft恢复另补原pending。精确满足prefix时保留已有draft状态。
5. 首yield前网络失败恢复可用快照并abort尚未完成的policy周期；不会发明acceptance反馈。当前请求失败后由engine丢弃局部cache。

每轮真实成本必须覆盖snapshot、draft、target验证、必要的target/draft重放、host输出与恢复。仅从操作计数可知：低接受率时常需额外target replay，可能比普通解码更慢；这只是成本方向分析，不是性能测量。DS startup curve用完整周期实测，不能扣掉恢复或只报告验证forward速度。`budget0`可回退并定期再探索。

候选小草稿已完成只读官方metadata与完整tokenizer核验：

| 草稿 | revision | 核验结果 |
|---|---|---|
| mlx-community/Qwen3.5-0.8B-4bit | da28692b5f139cb0ec58a356b437486b7dac7462 | 输出词表248320；与9B8bit整个tokenizer.json完全一致，SHA256 87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4 |
| mlx-community/Qwen3.5-2B-4bit | 674aaa7240b91e8012fcad5d791b7dfe5ba90207 | 同上；没有为本实验下载其权重 |

证据在 `artifacts/phase2-model-audit/hybrid-draft/compatibility.json` 与对应官方API/config快照。完整词表、added_tokens、tokenizer_config、EOS/BOS全部比对。独立chat_template.jinja文件有差异：9B默认thinking，0.8B默认non-thinking；两者显式enable_thinking=True/False渲染tokens均相同。引擎使用target tokens同时送入draft，禁止二次模板。2B尚未审计独立jinja，不把tokenizer_config相同外推全部模板相同。通过这些静态检查不代表真实模型输出一致或取得速度收益。0.8B权重下载/manifest由统一模型审计维护，不在适配器里隐式下载。

实验生产入口为 `InferenceEngine(adaptive_scheduler="ds", experimental_hybrid_speculation=True)`；要求profile明确有experimental_speculation元数据与真实锁定draft。公开API/UI默认仍只允许已经验证的supported_modes，不绕过为可用功能。已实际执行的硬件正确性控制可用下列命令重现（正式性能仍需独占硬件和冻结source）：

```sh
.venv/bin/python -m pytest -q tests/test_hybrid_speculative.py
.venv/bin/python -m pytest -q --run-metal -m metal tests/test_speculative_metal.py -k hybrid
.venv/bin/python benchmarks/experiments/hybrid_ds_probe.py --profile qwen3.5-9b-8bit --max-tokens 96
```

首次分歧诊断是独立profiling pass：固定k=2，找到首个与普通greedy不同的实际输出，保留验证前snapshot、原始batch logits及输入；在完全相同snapshot上逐token执行相同历史，记录top-2/token ID/margin、最大logit误差、KV offsets。若相同prestate的单步与批argmax也不同，才有该步形状效应证据；若两者相同而baseline不同，则继续隔离更早的state轨迹，不能直接归咎浮点。诊断开销不混入正式性能样本。

当前portable hybrid状态用例覆盖真实已消费历史、状态依赖输出、拒绝/中途close、零预算不调draft、pending追赶、网络异常、cache对象identity与分支隔离；全部通过。6项真实ArraysCache/KVCache tinyMetal控制已实际通过。9B8bit+0.8B4bit生产probe也已完成：baseline/fixed1/fixed2/DS各96输出tokens逐token相同，DS35周期中24次实际MTPVarlen选择；独立固定k=2诊断pass也未发现输出分歧，故logit分歧字段为null，不能捏造数值归因。报告为 `benchmarks/reports/phase2-hybrid-ds-qwen35-9b8-probe.json`。本轮单prompt、不同编译热身，只有机制/正确性证据，没有竞争速度结论。raw DS2473ms、固定k2 2363ms，尚未证明DS优于最佳fixed；baseline首请求4554ms含编译影响，禁止相除宣称收益。

主9B正式消融使用同一 `ds_budget_ablation.py --profile qwen3.5-9b-8bit`，会自动选择hybrid adapter。原生MLX-LM 0.32.2的speculative_generate_step显式拒绝非trimmable target cache，因此hybrid不运行native fixed、不伪造其耗时零分；原生greedy保留target-only预填充能力。同循环fixed0/1/2/4开发集上只从逐token与greedy相同的候选中选tunedfixed；这是自家消融控制，不算第二个竞争框架。正式运行尚未执行，最终需冻结模型、source与空闲I/O，并交第二完整服务引擎对照。

## 显式实验成本适配：N−2 lagged_mean

原始DS/MTPVarlen核心及默认`hybrid_cost_adapter="original"`均保留。新增 `ds/recovery_cost.py:LaggedMeanRecoveryPolicy` 是**Mac平台适配**，不是官方DS的启动校准实现。入口为 `InferenceEngine(adaptive_scheduler="ds", experimental_hybrid_speculation=True, hybrid_cost_adapter="lagged_mean")`，仅接受明确hybrid_cache元数据。

实际9B开发probe的恢复耗时是混合分布：固定k2的41个周期中27次全接受，中位fullcycle43.78ms但算术均值55.54ms；原DS adapter k4的25周期中10次全接受、15次拒绝，中位fullcycle85.52ms但均值72.86ms。首轮仅2样本曲线和中位数不能可靠表达这种随机恢复成本。不同模式的状态/顺序/编译不同，以上只描述旧测量，不能作竞争加速结论。

最小适配直接为原DS吞吐目标提供E[cost]估计：每预算至少5个已完成完整周期，使用最近9个完整周期的算术均值。**不从真实端到端耗时中扣除snapshot、draft、验证、replay、catch-up或host工作。** 输出预算截断/EOS式中途中止的实际成本仍保留在报告，只不作为完整稳态周期的校准样本。缺少样本时明确unknown并继续探索，不插值、不编造性能点。阈值5/window9是预先固定的实验参数，尚未通过保留集证明最优。

整个最新N−1 observation（包含与acceptance相关的runtime）延后一轮发布；当前N轮曲线与yield均只能使用N−2或更早信息。测试专门令N−1的acceptance和cost同时极端改变，验证当前预算与曲线不变。原core仍按N−2算total、N−1分配。每个estimated curve点记录真实cycle索引、原始完整耗时和stage分项，严格区分estimated与measured。

此适配直接观测完整cost曲线，所以**不向ForwardRuntimeEstimator提交残差**、不额外叠加旧baseline的slope、也不在更新曲线时清空历史掩盖漂移。core的residual/sampler输入明确为0；测量漂移已在滚动完整成本样本里。全部样本历史与反馈年龄可回放，原始C++ oracle继续核对每次最终budget。此模式不声称保留官方启动曲线+残差拟合的整套运行策略，只保留其expected-yield/cost预算算法。

```sh
.venv/bin/python -m pytest -q tests/test_recovery_cost_policy.py tests/test_ds_scheduler.py tests/test_hybrid_speculative.py
.venv/bin/python benchmarks/experiments/recovery_cost_replay.py
```

CPU分析产物：`benchmarks/reports/phase2-hybrid-recovery-cost-analysis.json`。35个旧实际DS周期对graphs1–4都不足5个可用样本，故没有完整新curve、没有新的core估计选择，也没有速度预测。报告中的hypothetical exploration不是实际执行请求，不能据此计算任何counterfactual耗时或收益。未运行新adapter的GPU性能/保留集；原报告字节未改。

### 中途关闭的反馈修复（RECOVERY-P2-001）

独立审查发现：旧观察器虽在lagged_mean排除了partial的cost，却仍把未输出的draft位置计为拒绝。这会污染跨请求的接受率；N−2滞后并不能修复错误样本。本次同时修复lagged_mean、原DS adapter和legacy中的同类风险，**只更改不完整周期反馈**，完整周期的统计/选择方法保持原样，原MTPVarlen核心与历史报告不修改。

实际observer收到的是已yield的accepted/produced计数。完整greedy验证周期必须在accepted输出之后再输出一个target/golden token，因此 `produced > accepted` 才是完整周期。EOS、max_tokens或消费者异常在accepted块内关闭时，后续draft位置是未知，不能当拒绝；整个partial周期同时退出shared cost与yield统计，实际wall time和已输出计数保留在原round记录及policy.partial_cycles诊断中。

DS的schedule/process轮次仍正常推进，prev_accept_rates移为先前accept_rates，当前accept_rates保持原估计，不制造新拒绝；lagged_mean保持整个observation一轮滞后，partial标记不进入后续cost曲线。之前已完成周期仍按年龄正常发布，不能为隐藏partial而重置历史。

portable回归通过真实hybrid与trimmable生成器分别验证：legacy/DS/lagged三策略在首次k4输出一个accepted token后close或消费者抛错，shared cost/positions完全不变；下次仍探索缺证据的k4，后续完整周期只贡献一次真实接受信息，N−2状态正确。原始C++ oracle回放同时插入partial周期验证预算规则。证据与源码hash见 `docs/evidence/phase2-recovery-partial-feedback-fix.json`。此修复没有新增GPU性能证据。

## 可复现实验服务与lagged_mean消融入口（CPU接线，未新增硬件成绩）

公开registry保持9B `supported_modes=["baseline"]`。`create_app(experimental_hybrid_speculation=True)`只在当前服务的局部metadata副本中，为已声明experimental_speculation、hybrid_cache且有真实draft pin的profile开放fixed/adaptive；`/v1/models`同时返回experimental、experimental_modes、verified_supported_modes及speculative_verified=False。默认服务仍在提交前422。engine自身仍需独立的实验flag，采样继续严格greedy；14B等未声明的候选不会随服务flag被放开。

独立实验server命令（需在获得GPU使用权后执行；当前仅完成CPU验证）：

```sh
.venv/bin/python benchmarks/experiments/deepinfer_service_server.py --port 8082 --experimental-hybrid-speculation --adaptive-scheduler ds --hybrid-cost-adapter original --prefill-step-size 512
# 对照进程使用同配置，仅改 --hybrid-cost-adapter lagged_mean；请求mode=baseline为DS off、adaptive为DS on。
```

`--prefill-step-size`仅是实验进程配置，通过dataclasses.replace作用于请求副本，不新增公共API字段。所有竞品的chunk设置/可调限制、额外625,229,487 bytes草稿权重和512MiB逻辑cache预算需进入实验manifest；不能假设驻留内存与target-only对照相同。

`ds_budget_ablation.py`在hybrid上明确包括独立state的original DS、ds_lagged_mean、legacy、fixed0/1/2/4与开发集锁定tunedfixed；原生只有其真实支持的greedy。新adapter保持预定min5/window9，无保留集反向调参；在线状态仅按已经完成的测量更新。失败保留error、实际partial输出和已产生rounds，不以零耗时代替错误；自家tunedfixed不是第二个竞争框架。

为新阶段提供独立冻结工具，不覆盖既有quality档案：

```sh
.venv/bin/python scripts/freeze_ds_source.py --output benchmarks/reports/phase2-ds-source-v1/source.tar.gz --extract /tmp/deepinfer-phase2-ds-source-v1
```

工具对已有archive/extract拒绝覆盖；冻结runtime、locks、具体harness、独立未修改C++ oracle与许可证，并写出`ds-source-manifest.json`。运行时必须从提取目录加载代码，以原仓库venv Python执行；`--freeze-manifest ds-source-manifest.json`验证真实tar的完整成员集、内容hash、内部manifest及canonical tree digest，并绑定执行中的`__file__`与imported runtime。提取目录的本地manifest另带archive_binding定位原档案（不进入tar或报告）；旧的无binding提取manifest可显式指定`--freeze-archive <archive.tar.gz>`。缺档案、缺文件、imported runtime不匹配或执行期间变化都禁止正式等价速度声明。源码冻结不代表已做性能测量。

短开发控制准备为 `hybrid_lagged_probe.py --profile qwen3.5-9b-8bit --max-tokens 128 --max-requests 3 --freeze-manifest ds-source-manifest.json --output <fresh-report.json>`。最多三个预定开发请求，保持原min5，出现真实成熟core选择后按预定规则停止，与输出是否一致无关；baseline/候选完整输出、失败、成熟与否全部保存。不会自动运行GPU。

独立 `verify_lagged_trace.py --input <report.json>`不导入Python scheduler：从实际rounds重建N−2完整cost样本/窗口/mean及两代acceptance，核对estimated curve每个输入都对应原测量，再把curve/state送未修改原始C++头。没有成熟curve明确报告insufficient_calibration，不能记作验证通过。CPU测试使用清楚标记的stateful fake-device fixture验证整条trace校验及篡改拒绝；这些fixture不是模型性能数据。

新服务接线与freeze已通过独立CPU审查。实际9B8 lagged成熟probe已执行，结果与补充核验边界见下节；真实API/SSE与正式消融另行记录，不能把先前original DS的9B probe视为新adapter成绩。

实验wrapper的性能默认是`round_diagnostics=False`，只保留轻量round计数；`--round-diagnostics`必须明确启用且属于独立profiling pass。公开engine的历史默认诊断设置没有改变。短成熟probe显式启用诊断以取得原C++回放输入，不能将这次profiling耗时混入性能服务结果。

混合成本适配的作用域也显式限定：`lagged_mean`只作用于hybrid profile的adaptive请求；baseline/fixed/native不构造未使用的lagged策略，非hybrid adaptive继续使用original DS。每次结果metrics记录hybrid_cost_adapter_requested、applicable、effective，避免实验server仍公开4B能力却被无关hybrid检查阻断。此项由实际InferenceEngine.generate控制流的设备边界测试覆盖，默认公共能力守卫不变。


### 实验来源与非空验收修复（独立审查四项P2）

旧审查证据 `docs/evidence/phase2-experimental-service-review.json` 保留不改。修复后，loose文件hash即使与checkout相同，也不能替代真实冻结档案；外部修改过的harness不能借cwd下同名文件取得正式有效标记。开始与结束必须核对同一manifest、archive、内部manifest、tree、执行harness与runtime。发布报告只输出hash及档案文件名，绝不输出本地archive_binding或home路径。

`--repeats`和`--max-tokens`在加载MLX前拒绝非正数。正式等价标记要求预定的每个held-out prompt×repeat恰好一次、每个variant均有真实非空输出与有效耗时、无失败且与原生greedy逐token相同；缺失、重复、空集合或陈旧的equal_to_greedy标志都不能通过。此标记表示比较有效性，仍不等于已证明优于竞争服务。

短probe保留raw `correctness_and_maturity_pass`与`maturity_and_oracle_pass`；只有结束后全部来源核验通过才可给出`accepted_frozen_probe_pass=True`。源码、manifest、archive或执行harness变化时raw功能事实仍保留，accepted标记为false。当前只有portable正确性验证，没有新增硬件成熟/性能成绩。可复现CPU回归：

```sh
.venv/bin/python -m pytest -q -m 'not metal'
.venv/bin/python -m pytest -q tests/test_ds_experiment_harness.py
.venv/bin/ruff check benchmarks/experiments/ds_budget_ablation.py benchmarks/experiments/hybrid_lagged_probe.py scripts/freeze_ds_source.py tests/test_ds_experiment_harness.py
```


### 真实9B8成熟曲线与核验容器修复

`benchmarks/reports/phase2-hybrid-lagged-dev-qwen35-9b8.json`来自archive `f48d430d2433194f4680896e699760a8e0a9602505a320833c02fbbefabe60ab`：baseline与lagged各128个输出token完全相同，58周期中有28次成熟core决策，min5/window9保持原值。原probe的独立核验在内存中失败，因为scheduler curve使用tuple pair，而核验重建使用list pair；落盘JSON会将tuple转成list，所以原始未修改verifier对同一JSON已能完成28/28原C++匹配。

本次只规范化pair的容器类型，不重新舍入、不增加浮点容差。真实报告分别恢复成runtime tuple与保留JSON list后均28/28匹配；任一cost改动一个Python浮点ULP仍被拒绝。原报告SHA为 `51c4b6e7744129fcfe55c69644d6d6129f021f255987058a8427223161be2060`，原始档案和`accepted_frozen_probe_pass=False`保持原样。独立补充结果另存 `benchmarks/reports/phase2-hybrid-lagged-dev-qwen35-9b8-supplemental-verification.json`，不把补充通过改写成原probe通过；此开发控制仍无正式速度结论，不需要重复GPU请求来修复序列化核验错误。

```sh
.venv/bin/python benchmarks/experiments/verify_lagged_trace.py --input benchmarks/reports/phase2-hybrid-lagged-dev-qwen35-9b8.json
.venv/bin/python -m pytest -q tests/test_ds_experiment_harness.py
```

同档案的API入口已冻结，可从对应提取目录、以原venv解释器及`PYTHONPATH=src`运行 `benchmarks/experiments/deepinfer_service_server.py --port 8082 --experimental-hybrid-speculation --adaptive-scheduler ds --hybrid-cost-adapter lagged_mean --prefill-step-size 512`，默认diagnostics关闭，公共registry门槛不变。该服务runtime不依赖此次离线verifier修复；未来内存核验probe须用包含修复的新档案。硬件执行仍由root统一分配。


## Phase3 section 2: accepted model capability interfaces

The config-driven `ModelSpec`, `ModelRegistry`, `ModelAdapter`, `CacheAdapter` and
`DraftProvider` contracts are **Mac platform capability/integration layers**, not
new DeepInfer algorithms or upstream MLX APIs. MLX-LM remains responsible for
model execution, tokenization, sampling and native cache machinery. Local file
identity, exact repo registration, metadata redaction, resource guards and explicit
baseline fallback are framework engineering. No trained DS ConfidenceHead, batch
rank/remap allocator, true variable-length verifier or ZOS overlap is added here.

The adapter separates chat/vocabulary/geometry, cache trim versus audited exact
snapshot/restore, and draft compatibility from existing DS scheduling. The
DeepInfer-derived expected-yield/cost core and its two-generation history semantics
remain unchanged; the new cache interface guards existing Mac hybrid recovery.
Probe method availability never enables an unverified optimized mode.

Accepted real execution is limited to SmolLM2-135M-Instruct 8bit dense baseline:
9 integration steps cover native full-token equality, cache branches, cancellation,
exact KV snapshot/restore logits and consistent Python/CLI/API/SSE identity.
273 portable checks and 41 tiny real-Metal regressions pass; historical pretrained
hybrid reports are not relabeled current-source optimized-path acceptance.
See `benchmarks/reports/phase3-model-adapter-integration.json` and
`docs/evidence/phase3-adapter-checkpoint.json`. This validates a replaceable model
boundary; it provides no new speed comparison or broad architecture-support claim.

## Phase3 section 3: request ticks and platform work admission

`continuous.py` and `server/continuous_worker.py` connect the API/CLI opt-in
backend to locked MLX-LM 0.32.2 `BatchGenerator.next`, including pure prefill
ticks. This is upstream greedy baseline batching. MLX-LM owns decode-first
ordering, cache merge/filter/extract, and batched forwards; these are not new DS
methods. Single-row dense KVCache is checked through its actual merge into a
batch cache, rather than incorrectly requiring batch methods on single-row KV.

The new Mac worker separates stable request/generation lifecycle, bounded work
queue, total active UID/context limits, future KV and longest-row padding,
extraction/scratch reserve, cancellation cleanup, and group fairness. Its
interactive policy adapts prefill chunk size to measured whole-tick latency and
ages waiting work. These platform resource/work policies follow the DS separation
of scheduling and cache capacity. They do not yet invoke DS cross-request
confidence ranking or variable-length verification. Existing single-request DS
MTPVarlen, two-generation history and exact hybrid recovery remain on the guarded
exclusive serial path.

One prefill row currently means an existing decode progresses alongside a long
prefill, while a newly arriving short prefill waits for that prompt slot. A tick
returns to the scheduler; decode output still waits for that tick's prefill
synchronization. No prefill preemption, vector verification cap, zero-sync ZOS
or performance advantage is asserted. Completed cache boundaries come from real
upstream Responses and are discarded, with continuous prefix reuse explicitly
cold. Model/cache protocol checks do not promote public capabilities.

See [CONTINUOUS_EXECUTION.md](CONTINUOUS_EXECUTION.md) for exact controls, ownership,
resource accounting and source-attributed ablations. Portable tests cover the
state/control contracts only. The opt-in actual small-model probe
`benchmarks/experiments/phase3_verify_continuous.py --execute` preserves native
output mismatches and both arrival directions.

Actual frozen Smol runs now observe real decode/prefill interleaving, both arrival
directions and prefill/decode cancellation with recovery, but **do not fully pass
section 3**. Both interactive and upstream policies match serial-native full
tokens on 12/13 completed outputs; the same Chinese B4 request diverges at index
39, token 27830 versus 179. Direct locked native BatchGenerator B4 reproduces all
four wrapper outputs exactly, while Chinese B1 matches the serial 48 tokens. This
is an observed upstream batch/shape-path boundary, not proof of a floating-point
root cause. The strict serial gate stays false; no case/tolerance is changed.
See `benchmarks/reports/phase3-continuous-{interactive,upstream}.json` and
`phase3-continuous-upstream-diagnosis.json`. Both policy probes have zero system
swap growth, but supply no stable speed claim. Batch-native agreement is separate
diagnostic evidence, never a replacement for the original strict control.
The direct control inserts four rows together at fixed chunk32 and is not an
arrival/shape-trajectory replay. Independent actual API/SSE cancellation/recovery
also passes (four completed streams, 50 events each); verification and all report
hashes are in `docs/evidence/phase3-tick-verification.json`.

### Experimental joint service boundary

`joint_service.MLXJointBatchSession` connects the actual separate target/draft
executor to `JointInferenceWorker`, reusing bounded request ownership, admission,
slot generations and terminal-state handling. The explicit service opt-in keeps
public default serial and baseline continuous behavior unchanged. It is limited
to custom exact-identity dense pairs and greedy requests. Both original cache
states, snapshots, copies, replay/catch-up and padded verification are reserved.
New-request prefill cancellation is isolated; verification-round failures roll
back and invalidate that affected group. SSE only receives verified output tokens.
Public round metadata is an allowlist of aggregate counters, never another row's
output or confidence observations. Native serial token equality, trained
confidence, automatic cost calibration and competitive performance are not
accepted by this integration. Without an independently measured exact-key curve,
the executor uses full budget. See `JOINT_VERIFICATION.md` for entry commands,
non-preemptible prefill and experimental shape-dependent output boundaries.
