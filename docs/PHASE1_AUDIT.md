# 第一阶段核查与第二阶段探索结果

核查日期：2026-10-21。结论依据实际源码、原始报告及本轮本机复跑。下一阶段执行输入见 PHASE2_EXECUTION_PLAN.md。

## 当前版本实际完成度

v0.1.0a1 是可用的 Apple Silicon 本地推理 alpha：真实 MLX 引擎、锁定模型、普通/固定推测/实验自适应路径、精确前缀快照与字节预算、串行有界worker、流式API、本地网页、CLI、静态网站、打包与原始基准导出均已实现。

本轮重新验证：原有 portable tests 21通过；原有14项Metal控制流测试通过；静态站7页/109资源链接通过。新增延迟同步路径后，Metal测试扩为28项并全部通过。这些Metal pytest使用小型确定性转换模型，证明缓存回退控制流，不等于所有真实LLM数值正确。真实LLM证据分别在下面原始报告中。

历史公开版本地址（发布记录）：https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1 。网站：https://zevin17.github.io/deepinfer-mac/ 。本轮新实验代码/报告尚未替换该公开alpha。

当前没有官方DSpark Confidence Head、完整DeepSeek模型、连续批处理或原创Metal内核。参考头的存在不是生产源码移植。当前定位更准确的是“可测量的Mac推理部署与优化实验框架”，尚不足称已超越常用框架的新高性能底座。

## 旧版本增益究竟是什么

### 推测解码

完整报告 `benchmarks/reports/m4-pro-qwen3-4b/`：216次正式测量、288次含热身；零运行失败；全部六模式完整token相同仅27/36组。baseline与native为36/36组相同。

| 变体 | 请求耗时中位数 ms |
|---|---:|
| baseline | 793.96 |
| fixed k=1 | 701.43 |
| fixed k=2 | 631.08 |
| fixed k=4 | 808.07 |
| adaptive | 797.44 |
| native MLX-LM | 810.84 |

不能把上述不同输出工作量的中位数直接相除，宣传等价速度提升。独立进程消融共54次正式测量，adaptive与native的9/9输出一致，但9/9均为calibrated_fallback，实际已退回普通解码；因此没有证据支持自适应调度加速。独立进程和完整报告覆盖的prompt数量均有限。

### 精确缓存

`benchmarks/reports/cache-qwen3-4b.json`：1740输入tokens、2输出tokens（含EOS）。完全重复时命中1739 tokens，TTFT中位数约2118.71→25.17ms，总请求2132.57→37.24ms；修改尾部问题的分支仅命中512 tokens，TTFT约2164.12→1564.55ms。完整输出一致。

真实原因是跳过重复prefill，不是decode算子变快。重复90遍的system文字与单数字输出属于受控窄场景，不能外推一般聊天84倍提速。

### 已知限制

- 真实量化推测输出存在分歧，尚未逐个定位原因；toy tests不能消除此问题。
- 反复reload后Metal active能释放，但native/Rust RSS驻留增长仍需分析；不能把固定模型稳定性外推至模型频繁切换。
- policy观测draft+verify而非完整周期，自适应草稿有逐token同步；永久disable bucket不随任务变化重新探索。
- 缓存只保存首chunk和完整prompt；MLX-LM本身已有原生LRU、最长公共前缀和消息分段缓存，当前实现不是独有发明。

## 本轮已实际尝试的增长点

### A. 中间检查点

新增opt-in `InferenceEngine(checkpoint_interval_tokens=512)`，默认仍为0。沿已存在的prefill边界保存精确状态，不改变算子输入形状，不对未知recurrent缓存任意trim。

原始报告：`benchmarks/reports/phase2-checkpoints-qwen3-4b.json`。

相同公开1469-token分支、3次重复：命中512→1024 tokens；TTFT中位数1163.43→568.16ms；总请求1176.17→580.16ms；输出与无缓存参考相同。首次seed中位数1751.37→1752.88ms，几乎持平。所有分支观察swap增量为0。收益仅是相对本项目旧版本。

### B. 公平的原生缓存控制实验——没有竞争优势

原始报告：`benchmarks/reports/phase2-native-cache-control.json`。

共用同一模型、baseline执行器、prefill配置、512MiB预算与prompt，替换缓存实现。这个实验隔离缓存查找能力，不是完整server评测。

| 缓存策略 | 分支命中 tokens | TTFT中位数 ms | 输出一致 |
|---|---:|---:|---|
| 本项目首版 | 512 | 1162.06 | 是 |
| 本项目中间检查点 | 1024 | 564.30 | 是 |
| MLX-LM原生LRUPromptCache | 1451 | 74.82 | 是 |

原生实现对可trim模型可以取更长公共前缀，本项目只能取已保存的精确边界。因此原生在此场景明显更快。下一阶段应复用成熟原生能力，围绕27B不可trim混合状态、DS调度核心和真实会话优化寻找可证明差异；不能用自我提升掩盖此负对照。

### C. 去除草稿逐token主机同步

新增opt-in `adaptive_generate_step(..., defer_draft_sync=True)`；参考MLX-LM的device-array/async_eval模式，统一验证后读取。计时标记combined_draft_verify，不能将enqueue时间冒充GPU草稿耗时。

原始报告：`benchmarks/reports/phase2-draft-sync-qwen3-4b.json`。

固定k=2、3个prompt×3次正式重复，另含热身，所有组旧循环/新循环/原生固定推测输出一致。旧/新配对耗时比中位数1.0967，即延迟约降低8.8%；原生/新比约1.0034，基本持平。测量排除prefill，用于隔离循环开销，不是完整请求吞吐结论，也没有证明与普通greedy所有输出一致。

结论：修复了我们相对原生的执行开销，尚未超越原生。此改进来源于MLX-LM，不是DS源码移植。

## DS源码核查与项目定位调整

本轮重新获取官方main的 `csrc/engine/mtp_varlen.hpp`，与已锁定参考字节完全一致。SamplerRuntimeEstimator与ForwardRuntimeEstimator本身只依赖标准C++；MTPVarlen通过Sequence和TORCH_CHECK与上游连接。它们是最适合真正移植并用独立C++参考核对的组件。

生产方案应分三层：DS原始成本/预算核心；Mac经验接受率与测量bucket适配；MLX模型执行。没有Confidence Head就明确写经验接受率适配，不称完整DSpark。将主框架目标调整为源码可追溯的DS调度核心，而非把一般MLX包装冠以DS名称。

DS生产KV缓存未完整公开，SWA bounded replay是特定模型上的近似状态重建；当前精确前缀快照不具备“直接DS源码移植”的证据。CUDA/Ascend与大规模EP机制也不能直接作为单Mac优势。

## 开源项目实际价值

已有价值：可安装可运行的本地服务、可追踪模型/源码、对照实验与失败数据、可扩展的优化试验入口、面向Mac的演示和部署流程。

尚缺的核心价值：相对成熟方案的明确性能优势、真正生产使用的DS组件移植、27B上的有效调度加速与充分质量验证。下一阶段把这些列为结果验收要求；没有结果前不宣传“更快的MLX”或“DeepInfer完整Mac移植”。

具体执行顺序、DS组件范围、对照方案与验收见 PHASE2_EXECUTION_PLAN.md。27B实测结果将记录于 `benchmarks/reports/qwen38-27b-2bit-demo.json`；模型状态以该报告为准，而非仅以下载成功为准。


## 27B 2-bit真实运行补充

已校验并加载 keXjos 两个权重分片。M4 Pro 上7次请求全部执行完成，stream拼接、热缓存/冷缓存、分支缓存/无缓存tokens一致。解码约24.4–25.8 token/s，Metal峰值最高约10.00GB（不等于全进程/系统内存）。首次请求系统swap使用量增约3.40GB，后续请求未观测到新增；这是系统级观测，不能直接归因于模型全部占用。

1733-token输入热缓存命中1732tokens，TTFT由13.468s降至78.8ms；分支命中1536tokens，TTFT由13.431s降至1.797s。只有单轮样本，属于自身缓存有效性，不是竞品优势。

质量未通过：中文解释陷入重复，dedupe代码不完整且未实现正确去重，算术长输入重复问题；多轮代号回忆正确。原生MLX相同模板/greedy设置的中文与代码控制输出逐token一致，不能将问题归为本引擎独有，也不能未经进一步控制就断定纯量化损失。工程检查通过与展示质量通过已分开标记。

结论：选 keXjos 作为已锁定、已实测的适配基线；尚不能宣布它是质量最好的2bit版本或合格主展示。agnosticeng有额外MTP资源，但未作同机质量比较，不因附带MTP就判其更优。下一步先核对生成配置、原始模型/适配器以及另一2bit版本，再定最终展示权重。原始证据：`qwen38-27b-2bit-demo.json` 与 `qwen38-native-quality.json`。本轮仅engine/native路径实测，27B API/SSE/UI验收仍待下一阶段。
