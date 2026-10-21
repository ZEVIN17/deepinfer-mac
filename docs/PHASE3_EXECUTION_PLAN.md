# 第三阶段：面向 Apple Silicon 的通用自适应推理框架

执行位置：现有 deepinfer-mac 仓库。Codex持续执行，不设日历排期，不逐项等待确认，关键节点汇报实测结果后继续。第二阶段实现、质量报告和冻结产物保留；第三阶段承接尚未完成的竞争性能验收。本文是新阶段方案，不表示其功能已经实现。

## 产品目标与完成标准

产品定位：**以 MLX 为执行后端、以 DeepInfer 设计方法驱动调度的通用本地推理框架。** 模型是可替换的执行对象，Qwen3.5-9B是首要演示与性能验证载体，不是产品定义。当前不重建MLX张量/自动微分系统；模型加载、算子与通用缓存尽量复用后端，核心投入在预算分配、请求生命周期、资源准入和执行重叠。

第三阶段交付应同时成立：

1. 用户可通过模型路径/repo与配置接入受支持架构，不修改核心调度代码；优化能力由适配器声明与探测决定。
2. DS设计 → Mac实现方法 → 正确性 → 真实执行 → 消融有完整对应，不能把调用MLX现成功能当作DS创新。
3. 在预先定义的代表场景中，相对原生MLX-LM和oMLX同时取得稳定、显著收益。目标：单请求任务总耗时降低至少15%，或并发交互p95 TTFT/ITL降低至少20%且吞吐不恶化；目标是待测指标，不是承诺。至少跨3个独立进程轮次、多个保留任务验证，区间支持正收益。不能宣称每个模型/负载均更快。
4. 主页、文档和展示站形成完整产品体验：漂亮、易上手、价值清晰，公开性能有准确场景标签与复现入口。

模型能力优先要求继续有效：资源/适配条件接近时优先能力与可信跑分。主展示9B8bit、同模型4bit对照、14B工程控制保持；不因更容易制造加速比而换成明显更弱的展示模型。硬件依然为24GiB M4 Pro，支持范围不等于已验证范围。

## 0. 已完成的源码核查与基线

用户指定DeepInfer commit `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`已读取。其 `mtp_varlen.hpp` blob仍为`09af680aa42b724fe86315007a160d5012a1fde9`，与二阶段参考一致，无需重写已通过oracle的核心。源码清单及blob见 `phase3-source-audit.json`，它是资料记录，不是依赖升级指令。

直接来源：

- [MTPVarlen](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp)：schedule/update_runtime，期望收益/成本、两代历史隔离。
- [设备预算分配](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py)：条件置信度累乘、跨请求rank与flat-index tie、fresh remap。
- [DFlash ConfidenceHead](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/mtp/dflash.py)：预测接受概率，有模型相关训练权重与逐位置温度。
- [worker](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py)、[zos.cpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/zos.cpp)：当前forward、上一轮postprocess、当前sample，以及跨轮remap和位置修正。
- [scheduler.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp)、[block_manager.hpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/block_manager.hpp)：prefill/decode工作预算、缓存容量、块引用及分支复制。
- [YoCo](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/yoco.py)、[cache_engine](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/cache_engine.py)、[EPLB](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/eplb/policy.py)：模型结构和多设备专用方法，保留为扩展依据。

同场景生态核查：`ARahim3/mlx-dspark@d6042f38f0aa7dd3e03a2a406111d725056d45a3`，`batch_engine.py:635–684`的SpecSlots.step共享cap和`[B,cap+1]`；`generate.py:1386–1415`的置信度截断存在物化路径；`calibrate.py`已有成本校准、深度修正等。以上仅适用于核查版本，不是对未来版本的缺陷断言。正式比较前重新固定合理稳定版本。

可复用的本地接口（MLX-LM 0.32.2）：

- `generate.py:1541` BatchGenerator；`insert(prompts, max_tokens, caches, all_tokens, samplers, logits_processors, stop_sequences)`、`insert_segments(...)`、`remove(uids, return_prompt_caches=False)`、`next()`。已有分块prefill和连续批处理，不从零重复造基础能力。
- `models/cache.py:596` ArraysCache，具有filter/extend/extract/prepare/merge等批操作；KVCache可trim。**不可trim不等于不可batch**，但这些接口也不自动提供变长推测逐行回滚。
- 现有 `ds/mtp_varlen.py`支持states列表；`policy.py`生产目前仅`schedule([state])`。`server/worker.py`按完整请求串行调用engine；`hybrid_speculative.py`是单请求快照恢复路径。
- 复用现有 `tests/test_ds_scheduler.py`、`test_hybrid_speculative.py`、`test_native_cache.py` 与 `benchmarks/experiments` 服务/预算比较脚本。先核对实际签名与锁定版本，再调用，不给上游API添加假参数。

执行时先读PROGRESS及当前diff，确认新增工作和资源条件；不要把冻结历史哈希改为当前版本。原有大型GPU比较暂缓条件延续，直到资源实际变化；可先做CPU算法、状态机、适配器和小模型验证，不重复要求用户关闭应用，也不通过取消内存保护强跑大模型。

## 1. 六个方向的判断与优先级

| 方向 | 判断 | 当前基础与真正新增 | 执行顺序 |
|---|---|---|---|
| 置信度+成本验证预算 | 项目核心，跨请求分配比单请求选长度更有辨识度 | 核心算法、经验接受率和单请求已存在；新增置信度契约、批次总预算、逐请求长度及实际节省计算的验证器 | 主线；通用接口后实现 |
| ZOS状态流水 | 高潜力，但收益依赖实际同步占比 | 已有async_eval借鉴和两代历史；缺请求动态进出时的跨轮映射与完整异步状态机 | 有时间线后做，预算执行正确后接入 |
| 缓存/工作量准入 | 本地交互的重要产品能力，也是多请求核心的地基 | 目前FIFO与内存检查；需请求可暂停的分块执行、工作预算与无饥饿优先级 | 先于真实跨请求预算 |
| 前缀共享/分支缓存 | 保留并针对瓶颈改进 | 已复用原生LRU、精确快照与消息检查点；重点是target/draft状态一致性和复制开销 | 随批处理集成，不作为首要卖点 |
| 模型/缓存共设计 | 重要但强模型依赖 | 设计原则可用于adapter；普通Qwen没有V4.1的CED/YoCo共享结构 | 本阶段不迁移模型特定优化 |
| 专家并行/EPLB | 不适合当前单机核心目标 | 多GPU冗余与通信不可直接转为24GiB单Mac收益 | 本阶段不实现；仅profile证明时补Metal热点 |

难度修正：第3项“策略”较简单，但从整请求串行引擎接入可抢占prefill与连续批处理是中等工程量；第1项真正变长计算、第2项动态状态流水均为高难度。采用先正确、再减少实际工作、最后重叠的执行顺序。

## 2. 通用模型、草稿和缓存能力接口

实现：从当前models.py固定profile拆出可配置ModelSpec/ModelAdapter/CacheAdapter/DraftProvider；这些是拟新增项目接口，不是已存在的MLX API。模型注册接受本地目录或repo+revision，演示profile只作为预设。加载仍走MLX-LM；能力查询提供baseline、batch、trim、snapshot/replay、varlen_verify、confidence、sampling等，具体模式按经过验证的能力组合启用，缺优化能力时透明baseline回退。

参考：当前engine._load/_tokens、models.verify_pair、native cache adapter，以及上游BatchGenerator。模型差异放adapter：模板/词表、offset/mask、recurrent状态、hidden-state tap与draft对齐，不在调度器里按模型名字分支。缓存身份纳入模型/量化/模板/位置和draft版本；置信度权重及校准revision也入实验身份。

验证：现有9B/14B/4B路径保持；新增至少一个不同架构家族的实际baseline集成（先选有MLX适配、许可清晰且能驻留本机的小模型，精确ID核验后锁定）。第三方模型不要求都有推测加速，但能力查询及回退真实可用。至少dense与hybrid两种状态结构验证优化，不能把同架构不同参数量称为全部架构覆盖。

验收：接入新模型无需编辑调度核心；Python/CLI/API模型身份一致；不支持的能力有明确可用替代；权重、tokenizer不兼容在执行前识别。baseline通用兼容与optimized验证范围分别列出。

## 3. 请求生命周期与交互优先的工作预算

实现：复用上游批生成容器，将worker从整请求执行转为scheduler tick；请求具备queued/prefill/decode/draining/finished/cancelled状态，稳定request_id和slot_generation。同模型/兼容状态分组批执行，不把不同权重混在一个batch。每轮维护token预算、上下文工作量、KV/恢复临时空间和活跃请求容量。

参考：DS scheduler._schedule/_schedule_prefill/_schedule_decode的联合决策；MLX BatchGenerator.insert_segments/next/remove。prefill_step_size参数本身不能保证让其他请求插入，需要验证tick确实返回、可让正在decode的请求取得执行机会。按实测prefill时间与内存调整chunk，优先保持流式响应，同时用aging/配额保证后台任务不饿死；单用户默认低延迟，多请求按需求提高吞吐。

实现内存准入时将target/draft、KV、共享前缀、recurrent快照、验证buffer共同计入；预留与实际占用反馈校正，避免RSS/Metal重复计数。排队有界、取消可清理、过载及时返回明确状态。优先级只影响调度，不应改变同请求的token历史。避免为了塞进更多请求默默缩短用户输入。

验证：短聊天穿插长文档、取消prefill/verify、请求进出及slot复用、资源不足恢复。并发1/2/4分别测TTFT/ITL p50/p95、queue等待、完成率、总吞吐与swap。该项和上游连续批处理独立开关消融，证明增加的是DS对应策略而非首次启用上游batch。

验收：真实交错执行且无串话、重复/丢失输出；长prefill不再独占整个请求周期；批量任务不会永久等待。性能目标取决于同负载对照。

## 4. 置信度与成本驱动的真实变长验证（核心）

### 4.1 置信度输入与成本模型

将当前经验接受率与模型置信头放在统一provider契约下：明确输出是条件接受概率还是累计前缀存活概率，禁止对累计值重复cumprod。已有经验统计作为可靠后备；支持模型匹配且实际含训练权重的ConfidenceHead时再开启逐位置预测。不能将草稿softmax最大值直接当接受概率，也不能把DeepSeek训练头的权重直接装给Qwen。

若现成兼容draft没有训练头，先用已完成验证数据估计/校准，称经验预测；可在冻结特征和独立训练/校准数据后试轻量头，但不能把训练实验当首发依赖。未验证的后缀是删失数据，不直接计为拒绝。报告校准误差、位置覆盖、前缀接受预测与实际值；只有完整运行收益才决定采用，预测指标变好不等于加速。

复用DS两代历史隔离；将batch规模、上下文区间、量化、draft及执行策略纳入性能曲线键。成本包括全部draft/verify/recovery/catch-up/host工作。保留原成本适配与lagged_mean，按独立开发集选用；扩展批成本不能直接乘单请求成本。成本漂移重新校准，短请求计入冷启动损耗。

### 4.2 总预算与逐请求分配

MTPVarlen输入真实活跃states，先选择total_q，再按连续前缀收益分配seqlens_q；对照DS设备实现的cumprod/rank/tie/fresh remap，保留稳定request-id映射。优先实现MLX数组版本，必要时对小batch排序/选择热点写Metal实现。新CPU、MLX、Metal版本分别与独立参考对照，不以同一实现自证。

验证空/新请求、全零/全一、相同概率、预算边界、EOS、被取消/换slot、两代滞后与请求顺序变化。输入概率需校验finite与范围；单请求退化行为必须可解释。

### 4.3 让验证长度真正改变计算量

先建立共享cap和padding版正确性控制，再实现按验证长度分桶的microbatch，优先复用已有dense计算；依据GPU时间/有效token数判断小桶启动开销是否抵消节省。若分桶不足，再设计packed/ragged路径及匹配的attention mask、position与KV写入。不能假定上游BatchGenerator接受向量cap；新增验证器必须明确实现接口。

报告每轮requested_q、有效verified_q、padded/executed_q、kernel/提交次数和完整耗时。只做mask或把每行补至max长度不计“减少计算”成果。per-request预算与shared-cap最优基线必须同batch、同draft、同恢复策略比较。

hybrid必须逐行快照/恢复或有证明正确的专用状态方法；padding token不能污染recurrent state。快照成本、批内拒绝、EOS/输出预算截断、异常、slot复用、target/draft共同前缀均要覆盖。不要将DS近似SWA bounded replay用于要求精确等价的通用recurrent恢复。

验收：dense/hybrid实际执行、独立请求隔离、greedy输出控制，以及异质接受率并发负载上优于shared-cap的完整成本证据。greedy先交付；采样推测另实现正确的拒绝采样和随机数状态隔离后开启，不能直接沿用greedy接受规则。

## 5. ZOS式跨轮状态流水

先在原生MLX、当前引擎、变长验证新路径各抓一次执行时间线，区分GPU计算、host物化、调度、tokenizer/SSE、编译与恢复，确认等待占比。若没有可消除的等待，不为了“异步”增加复杂度。

参考DS worker.execute_model_step/_correct_seqlens_qk与zos.prepare_zos_inputs：建立StepState，包括request_id/generation、remap、已提交/已消费长度、pending token、验证结果、CPU可见历史代次与设备依赖。当前模型计算提交后处理上一轮输出，当前采样/接收结果延后消费；尽量把前缀概率、长度选择和接受长度更新留在设备端。

复用MLX async_eval及已核查的stream依赖语义；CUDA事件不能直接照搬为不存在的MLX API。CPU调度器控制shape必须同步时，先保留一个明确边界，实测分桶/固定容量是否值得；device scalar不能魔法般驱动任意Python动态shape。对EOS、取消、异常和服务关闭执行最终drain，禁止发出未验证token或重用未完成slot。

验收：相同输出与请求顺序语义、单请求/动态批进出均正确；时间线证明减少了目标等待；同步/流水开关下完整请求收益。不能把基础mx.async_eval当新ZOS成果，也不宣传“完全零同步”。

## 6. 缓存协作与可选热点

复用native LRU、现有消息检查点及精确混合快照，先profile复制的字节和时间。将target/draft的已消费前缀、待消费token及恢复锚点作为一致状态管理；target命中但draft缺失允许只重建draft并计成本。共享只读前缀，分支可变状态隔离；验证取消后引用释放与长会话内存稳定。

只有复制确实主导时，才做写时复制尾部/页管理，保留上游后端对照。上游原有prefix cache/KV量化不标成独创。量化KV可能改变输出，必须单列质量条件。YoCo/CED/SWA近似恢复、专家副本与集群通信不在本阶段主线；未来由模型能力明确声明后接入。

## 7. 竞争性能与泛化验证

复用二阶段服务输入审计、权重锁和质量协议；先修复时间/内存覆盖缺口，再冻结本轮源版本。主要对照是原生MLX-LM和oMLX；mlx-dspark为预算/批执行机制对照，不替代两项主要竞争目标。oMLX支持且适合目标模型的DFlash等加速纳入开发集合理调优，不能故意关闭其强项。双方自动回退、资源不准入和失败均分别记录。

公平层次：

- 框架端到端：相同target权重/量化、模板、输入、采样/思考与输出要求、总资源预算；允许各框架采用其合理优化，披露不同draft和依赖版本。
- 机制归因：本项目内部固定draft、batch与验证器，逐项关闭DS预算、置信度、分桶、准入策略、流水，验证每项贡献。
- 跨模型：至少dense与hybrid分别给结果；不同模型/量化之间不计算引擎加速比。新家族baseline兼容另列，未测的优化不外推。

保留集包含短聊天、代码长输出、长系统提示复用、长文档插入、分支续聊、低/高接受率混合与并发到达。开发集调参后冻结，不用保留集选最优参数。到达负载固定且包含排队时间，不只测服务完成后再提交下一条；报告过载失败，避免仅保留完成请求美化p95。

按prompt/会话聚类统计，至少3个独立进程轮次，轮换框架顺序、独占GPU比较，warm/cold及缓存状态分开。测TTFT、ITL、任务总时间、成功任务/s、吞吐、峰值、swap及失败；任务质量和数值差异分轨报告，采样模式不要求所有种子逐token一致但需要分布/任务质量验证。等待流式输出不得从测量中扣除。

未获得稳定优势时继续定位目标瓶颈或选择有依据的适用场景，保持结论范围与数据匹配；不以优化内部微基准代替最终服务收益，也不反复跑到“恰好显著”。

## 8. GitHub与展示站：积极的产品呈现

首页围绕能帮助用户完成的事，不按失败日志组织。技术限制、历史实验与失败样本放在Compatibility、Benchmarks、Research Notes；影响安装/默认使用的信息就近简述，性能图旁保留机器/模型/条件。详细证据可点开，不在首屏堆叠免责段落，也不隐藏会误导选择的重要条件。

拟采用产品叙事（英文与中文同步）：

> **DeepInfer-Mac — Adaptive inference for Apple Silicon.**
>
> Run local models through a unified API. Explore DeepInfer-inspired scheduling on MLX, from interactive chat to concurrent workloads.
>
> **让本地模型，更懂你的 Mac。** 通用模型接口、面向交互的调度，以及可复现的性能实验。

上面“统一API/并发”文案随功能实际落地启用；规划阶段用“Building…”或路线图表达。不能把未来能力写成现有功能。取得优势后用具体文案，例如“在[M4 Pro/模型/负载]中，p95首token延迟降低[X%]”，X必须来自发布数据，不提前填数字。

README交付顺序：一句定位 → 精美真实动图/录屏 → 3个用户价值点 → 5分钟Quickstart → Python/兼容API例子 → 模型能力矩阵 → 精选性能卡 → DS方法示意 → 文档/贡献/路线图。首屏不把某个Qwen型号放进产品定义；演示模型在截图与benchmark配置中标注。

展示站：复用现有site/build.py、assets/style.css/showcase.js与冻结证据组件；增加统一视觉的hero、交互式预算分配图、模型切换示例、并排真实演示、质量/延迟切换图和复现下载。概念动画清楚标“机制示意”，记录回放标“实测回放”；不把示意动画伪装成实时加速。录屏展示代码生成、长文档问答及并发流式响应，各自使用已验收能力。

开源体验：清晰安装/升级说明、完整许可证和DS/MLX归属、贡献指南、适配器开发教程、问题模板、可复现实验模板、示例项目与good-first-issue建议。CI/版本/平台badge仅连接真实状态；发布说明用“新增能力—用户价值—验证结果”描述。预备中英文发布短文、架构图及演示素材，不自动群发或制造stars、用户评价。

验收：全新环境Quickstart可运行；文案与能力矩阵一致；桌面/移动端可用、录屏和按钮可用；数字有原始报告/版本链接；页面主旨是产品价值，深层文档保持技术透明。先产出本地可预览页面和发布包，再按已有发布授权范围执行；本计划更新不等于已经上线第三阶段内容。

## 9. 连续执行顺序、汇报与最终交付

执行顺序：核对已有工作与冻结基线 → 通用能力接口 → 请求tick与资源策略 → 跨请求预算/真实变长验证 → 时间线驱动流水 → 两框架正式对照 → 发布。README结构、视觉设计、示意动画、CPU算法验证可随主线推进；实际数字和演示在对应验证完成后填入。

Codex自主选择实现细节与实验方向，不逐项征求确认。每个节点报告“新增能力、真实结果、证据、下一步”，报告后继续：

1. 通用接口与动态请求执行可用，给实际模型/请求演示。
2. DS批次预算和逐请求长度进入真实执行，给正确性及执行工作量变化。
3. 主展示与另一架构的性能结果，报告是否达到两框架目标及适用边界。
4. 产品README、展示站、可安装候选与复现材料交付。

最终产物：源代码与测试、更新DS_DESIGN_MAPPING、模型能力矩阵、同步/流水/预算消融、竞品保留集数据与统计、可复现源码/依赖锁、安装包、README中英文、展示站与演示素材。普通CPU测试、实际Metal测试、服务验证和网页链接/移动端检查按变更完成；不把单元测试数量当性能证明。

执行入口先读本计划、PROGRESS、DS_DESIGN_MAPPING；只在必要权限或外部条件确实阻挡且无法自主解决时提出具体问题，其余工作持续推进。没有运行中的自动化时不声称全天候后台任务已启动。二阶段历史说明不覆盖本计划的通用框架定位。
