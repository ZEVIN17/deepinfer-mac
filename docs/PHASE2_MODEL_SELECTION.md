# 第二阶段模型选型：24GiB，能力与实用性优先

核查与本机实验日期：2026-10-21。保留Qwen3.5-9B-8bit作为主展示和公平框架比较的目标模型；9B4bit保留质量/效率对照，Qwen3-14B-4bit保留后备与调度控制。执行范围见 [PHASE2_EXECUTION_PLAN.md](PHASE2_EXECUTION_PLAN.md)。这是一项有明确适用范围的工程选择，不是模型总榜排名，也不是框架加速结论。

## 选择依据与边界

本机24GiB上，9B8bit已经实际加载，baseline/native、API/SSE、缓存、取消恢复与本地UI均有实跑证据。4204输入+256输出的单请求检查完成，Metal请求峰值10,651,956,858B，加载/请求/全过程系统swap增量均0；没有硬件证据要求为了可运行性降到4bit。8300+256检查出现系统swap增加2,794,127,360B，正式8K组仍deferred，不能把模型标称上下文当作本机保证。

相同冻结运行时、模板、任务、预算及采样下，4bit在公开代码子集和若干知识/格式任务出现退化；开发预算控制也未建立质量等价。因此不能仅因其内存更小或裸tok/s更高改成主模型。它仍有实际胜出的任务：本地greedy smoke为11/12，而8bit为10/12；延长预算后的开发题9423采样正确耗时23.79s，8bit为138.14s。全部有利和不利结果保留。

14B指令子集表现较好，知识题还受到Markdown答案格式的影响，不能据严格parser分数断言其知识能力较低。9B8bit的共同采样代码子集为3/4，14B为2/4；样本很小，Wilson区间宽，不能据此宣称整体能力显著领先。保持9B8bit符合原先能力优先的选择，且本机开发任务显示可用性；14B的标准KV适配更简单，不构成把它包装成最强模型的理由。

## 实际权重与来源

| 模型 | 固定revision | 全部权重B | 纯文本tensor B | 实际加载 |
|---|---|---:|---:|---|
|Qwen3.5-9B-8bit|`16daa4818c54ce5f5436f929d52542eb65bbed9d`|10,426,592,423|9,514,412,544|是|
|Qwen3.5-9B-4bit|`8b2b98c00a6b4d291155e4890773ca8f769aee53`|5,950,221,072|5,038,041,600|是|
|Qwen3-14B-4bit|`a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`|8,307,898,514|8,307,783,680|是|

三者完整分片SHA和tensor header均已核验；9B仓库包含视觉权重，本项目仅验证文本运行。无MTP权重不能因为配置字段就宣称MTP支持。来源、许可证、每片SHA、tensor数、模板/配置hash和采样条件见 [公开模型审计](../benchmarks/reports/phase2-model-weight-audit.json)；两份models.lock同步实际加载与实验状态，权重不打包进发布物。

原27B家族已退役，精确清理六个本任务缓存/chunk目录，共19,177,137,641逻辑B（17.86GiB）。保留审计/历史失败报告、4B/0.6B回归与必要比较权重，未删除其他项目/Ollama模型。新的必要DFlash权重是单独审计的实验依赖，不能视为已经通过发布或性能验收。

## 固定公开子集：原严格分数

| 条件 | IFEval | MMLU-Pro | LCB公开tests | 辅助修复 |
|---|---:|---:|---:|---:|
|9B8bit greedy|6/8|11/14|2/4|1/1|
|9B4bit greedy|5/8|8/14|1/4|1/1|
|14B greedy|7/8|10/14|2/4|1/1|
|9B8bit共同采样|7/8|9/14|3/4|1/1|
|9B4bit共同采样|7/8|8/14|1/4|1/1|
|14B共同采样|8/8|9/14|2/4|1/1|
|14B独立官方native采样|8/8|8/14|2/4|1/1|

共同采样是Qwen3.5非思考建议条件0.7/0.8/top_k20/presence1.5、seed42；14B最后一行使用其Qwen3官方非思考条件，presence0，独立列出，不能混排为相同协议。全部189响应保留。本题集不是完整榜单，LCB仅public tests，不是完整官方pass@1；没有综合能力总分或量化等价证明。见 [子集、许可与复现](../benchmarks/quality/README.md)、[三模型逐题paired报告](../benchmarks/reports/public-quality-paired-v2.json) 和 [14B官方控制](../benchmarks/reports/public-quality-qwen3-14b-official-v3-1-summary.json)。

MMLU格式敏感性对全部98条MMLU响应统一仅删除Markdown强调/代码标记，再使用原答案regex；其余91条代码/指令响应不改。9B8/4分数均不变；14B greedy10→11，共同采样9→11，官方native8→11。六个变化都是正常停止但答案带Markdown，原严格分数保持不变。这项事后离线分析隔离格式与选项正确性，不能当作新生成或替代官方榜单：[格式敏感性报告](../benchmarks/reports/public-quality-mmlu-format-sensitivity.json)。

## 预注册开发预算控制

按照9B8开发集触cap规则，在全部追加4096运行前固定1471、9423和abc387_c，三模型、greedy/共同采样各4096输出预算，共18响应。同一冻结V3.1运行时、原prompt IDs、seed与采样，18/18新输出都包含旧同条件的完整token前缀。原27题主分不改，保留集未参与选题。

| 开发3题 | greedy原→4096 | 共同采样原→4096 |
|---|---:|---:|
|9B8bit|2/3→3/3|1/3→2/3|
|9B4bit|0/3→0/3|0/3→2/3|
|14B|0/3→0/3|0/3→0/3|

14B greedy1471另有正确字母被Markdown遮挡，统一格式敏感性后内容为1/3；原严格0/3不改。9B4 greedy两题仍到4096上限，LCB完成但解法错误；延长预算并不保证恢复。9B8 greedy LCB需要3832tokens/144.65s才正确，共同采样仅1196tokens/47.52s；9B4共同采样需3391tokens/75.19s。较长响应成本及相反胜出的题9423均公开，不把裸tok/s当作完成正确任务更快，也不把不同模型/采样的时间算为框架提速。[完整前缀/预算/耗时/资源](../benchmarks/reports/public-quality-dev4096-budget-sensitivity.json)、[开发格式敏感性](../benchmarks/reports/public-quality-dev4096-mmlu-format-sensitivity.json)。

## 本地smoke与解释限制

12题×4条件覆盖代码修复、JSON、事实提取、中文、多轮与算术。greedy/native/官方非思考/思考分别：9B8bit为10/10/11/9；9B4bit为11/11/9/7；14B为10/10/10/10，每项分母12。三个模型各12/12 greedy/native完整token相同。9B8bit的greedy球数反例仍答错，推荐非思考采样修复该题但另有硬件事实错误，因此没有宣称某一配置12题全部正确。思考最小1024预算下未结束分别3/4/0条；这不是无限预算能力结论。

中文为Codex逐条固定rubric审读，没有独立人类标注。“KV缓存”题未明确LLM范围，14B通用key/value解释不符合既定LLM rubric，但不能据此声称中文固有能力下降。14B思考dedupe函数通过三项测试，额外解释违反only-function要求，主任务仍失败；不能把格式错误误写成算法不会。所有原始输出、错误和诊断保留；更多协议见 [PHASE2_QUALITY_PROTOCOL.md](PHASE2_QUALITY_PROTOCOL.md)。

## 公开能力资料与框架贡献

初始选择参考固定版本 [Qwen3.5-9B官方卡](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) 的官方自报：MMLU-Pro82.5、GPQA Diamond81.7、IFEval91.5、LCBv6 65.6、BFCL-V4 66.1。它们不是本机MLX量化复测，不能替代上表或用于计算本机提升百分比。Ollama/HF分发证明模型可获得性，不证明agent效果或开发者满意度；没有“公认最佳”的调查结论。

9B混合状态的精确snapshot/replay和DS派生成本调度已有受限正确性/API运行证据，trained DFlash也仍是独立实验路线。模型质量、传输可用、原始C++控制重放通过，不等于框架加速。最终仍须相同主模型/质量下，与原生MLX和oMLX合理最佳条件进行独立开发选择、冻结保留验证及性能消融；未取得竞争优势时明确报告并继续。默认发布能力仍为baseline，实验路径须显式开启。
