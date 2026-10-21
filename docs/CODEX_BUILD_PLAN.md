# DeepInfer-Mac：Codex 执行计划

本计划是工程执行输入。按依赖连续完成，不按日历排期；验收点用于核对交付证据，不是等待用户批准的关卡。

## 目标与执行方式

在现有 `https://github.com/ZEVIN17/deepinfer-mac` 中构建面向 Apple Silicon 的开源本地推理引擎。复用 MLX / MLX-LM，借鉴 DeepInfer 的调度思想，首先验证自适应推测解码和多轮 Agent 场景的内存管理；同步交付本地演示、公开展示网站、可复现基准和开发者文档。

Codex 自主完成调研、实现、调试、测试、性能实验、文档、网页、打包和发布准备。常规技术选择自行决定，完成一个验收点后继续推进。仅在确实缺少外部凭据、硬件或用户决策时记录缺口，继续其余可执行工作。不得把缺少大内存 Mac 当作停止普通 Mac 路线的理由。

每次恢复只需读取本计划、`docs/PROGRESS.md`、当前 git diff 和有关测试结果。维护一个简洁进度文件：已交付证据、当前工作、下一步、外部依赖。避免构建复杂状态机或重复计划文档。耗时实验保存参数、日志、进程信息和中间结果，重启后检查进程，避免重复启动。一次只运行一组占用 GPU 的性能实验。

既有未提交代码必须保留并审查，不因仓库尚无提交就清空重建。提交只包含本任务核实过的文件，排除权重、缓存、凭据及原始私人输入。远端发布遵循用户已有授权与环境权限；没有发布权限时完成本地可发布产物并说明具体缺口。

## 已知起点（2026-10-21，只代表检查时快照）

- 本地目录：当前仓库根目录 `.`。
- 远端 origin 已指向用户仓库；本地 master 尚无提交。
- 机器：Apple M4 Pro、24 GB 统一内存。
- 本地虚拟环境已有 MLX 0.32.4、MLX-LM 0.32.2；当前 pyproject 版本范围比实际环境宽，须生成经验证的锁文件。
- 已有 `server/app.py`、`server/worker.py`、请求 schema、benchmark runner/metrics/export、doctor/config 和部分上游参考文件。
- 检查时缺少 CLI、engine、state、网页和 README 等被现有代码引用的模块/资源；重新扫描确认后补齐。
- 已有模式名 `baseline/dspark/deepinfer` 不应自动视为真实上游实现，须按实际实现修订。

## 产品边界与技术定位

首版面向可在本机容纳的量化模型，优先一个主模型和一个兼容草稿模型，覆盖代码生成、中文问答、重复系统提示词、多轮长输入。已有 qwen3-4b profile 可作为候选，具体 Hugging Face 仓库与 revision 需核实后锁定。增加第二个模型验证泛化，再通过社区扩展硬件矩阵。

DeepSeek 蒸馏模型可作后续兼容演示，但不可把其运行结果当作 V4.1 架构移植证明。完整 V4.1 的权重、CED/CSA2、专用 MTP 和置信度头支持另列实验路线；当前 24 GB Mac 不承担完整模型验证。首版不重造数组框架，不以磁盘换页运行超大模型作为高效推理卖点。

对外使用独立社区项目身份：DeepInfer-inspired inference for Apple Silicon, powered by MLX。每项能力注明是上游复用、组件移植、独立实现还是实验功能；任何速度优势注明机器、模型和工作负载。

## Demo 模型补充：Qwen3.8-27B（用户指定，2026-10-21）

将 Qwen3.8-27B MLX 4-bit 列为优先主展示候选，原有 Qwen3-4B 保留为快速回归和推测解码对照。主展示候选不等同已完成硬件验证。

- 候选仓库：https://huggingface.co/mlx-community/Qwen3.8-27B-4bit 。文件目录目前约 16.1 GB（十进制，约 15.0 GiB），不是实际运行峰值；量化为 affine 4-bit / group size 64。执行时固定 revision 和文件清单。
- 首先验证纯文本 baseline、流式 API 和网页，以输入 2K/4K tokens、输出 256 tokens、batch=1 做初始资源探测；根据实测扩大上下文，不把模型标称上下文能力作为本机可承载承诺。中文问答和代码演示分别检查质量、TTFT、decode 速度、峰值与 swap 增量。
- 本机 24 GB 需容纳系统、模型、循环状态、KV 和临时缓冲；不要预先承诺长期稳定或具体 tokens/s。优先仅加载文本权重，确认本地 qwen3_5 sanitize/load 路径与这份多模态量化权重兼容。视觉 demo 后续单独使用经过验证的 mlx-vlm 路径。
- 官方架构为 Gated DeltaNet 与全注意力混合；该量化配置 model_type=qwen3_5。当前 MLX-LM 0.32.2 的 qwen3_5.make_cache 使用 ArraysCache + KVCache，其中 ArraysCache 不能普通 trim。现有通用 speculative_generate_step 的可裁剪缓存检查因此不能直接满足。
- 先完成可运行 demo，再研究混合循环状态的正确 checkpoint/restore/replay 或已验证的上游实现。所有拒绝位置、EOS、取消和跨请求复用必须对齐普通生成；不能给 ArraysCache 简单添加 trim 假装可回退。
- 官方模型说明提到 MTP，但不代表此量化仓库保留了可执行 MTP 权重。读取权重索引确认预测头、运行时支持与额外开销后再选择 MTP 路线。
- 不直接沿用现有 Qwen3-0.6B 草稿：Qwen3.8 的 padded vocabulary 为 248320，现有 Qwen3 为 151936；须重新验证 tokenizer 和草稿兼容性。
- 27B 的自适应加速尚未通过时，网页明确显示 baseline/缓存优化模式；小模型上的加速数据单独展示，不能归因于 27B。4-bit 若资源余量不足，先降低 prefill 分块/上下文；如评估更低 bit，须作为独立质量对比，不静默替换。

验收：锁定模型真实加载并稳定完成多轮 demo、无持续增加的 swap、指标可复现后成为默认展示模型；推测解码支持独立验收。发现适配缺口继续修复，不影响已有小模型开发验证。

依据：
- https://huggingface.co/Qwen/Qwen3.8-27B （架构、MTP、词表）
- https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/main （文件大小）
- https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/raw/main/config.json （量化与模型类型）
- 本地 MLX-LM models/qwen3_5.py:347、models/cache.py:164,596 与 generate.py 的 can_trim_prompt_cache 检查。

## 0. 资料、源码与基线发现

实现：

- 检查适用 AGENTS、仓库 diff、当前环境和所有未完成引用，先运行能运行的静态检查与测试，记录真实起点。
- 固定 DeepInfer、MLX、MLX-LM 的版本或 commit，保存 `upstream.lock.json` 和 `docs/architecture.md`，记录参考路径、许可证及移植范围。
- 阅读 DeepInfer DSpark 调度、草稿接受率处理、异步调度与测试；对现有 `tests/reference/vendor/mtp_varlen.hpp` 查明来源和版本。
- 记录已核实 API 与缓存回退语义；优先复用现有实现，只在适配层隔离版本敏感部分。
- 实际核实主/草稿模型的 tokenizer、词表、特殊 token、量化格式、权重大小、上下文支持及许可证，生成 `models.lock.json`。

已核实可用 API（行号基于本次本地环境，执行时重新核对锁定版本）：

- `.venv/lib/python3.12/site-packages/mlx_lm/utils.py:648`：`load(path_or_hf_repo, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False, return_config=False, revision=None, trust_remote_code=False)`。
- `.venv/lib/python3.12/site-packages/mlx_lm/generate.py:667`：`stream_generate(model, tokenizer, prompt, max_tokens=256, draft_model=None, stream=..., **kwargs)`，草稿模型要求使用相同 tokenizer。
- 同文件 `:485`：`speculative_generate_step(prompt, model, draft_model, stream=..., *, num_draft_tokens=2, max_tokens=256, sampler=None, logits_processors=None, prompt_cache=None, prefill_step_size=512, kv_bits=None, kv_group_size=64, quantized_kv_start=...)`。没有已经核实的动态长度 callback；不可直接杜撰参数。
- 同文件 `:276` 的 `GenerationResponse` 与 `:313` 的 `generate_step` 用于核实流式响应及基线行为。

本地源码补充约束：

- `models/cache.py:15,115,122` 提供 `make_prompt_cache(model, max_kv_size=None)`、`can_trim_prompt_cache(cache)`、`trim_prompt_cache(cache, num_tokens)`。部分模型自定义缓存，不能把 max_kv_size 当作通用硬限制。
- `generate.py:725` 在推测模式移除 max_kv_size；内存预算必须在适配器层真实实现。`:597-599`、`:654-664` 的回卷、全接受补 token 及 finally 清理必须一起理解。
- `GenerationResponse.peak_memory` 是累计全局峰值，定义请求测量边界后才能作请求级指标；prompt_tps 的计时可能包含首次草稿/验证，不直接称纯 prefill 吞吐。
- `tests/reference/vendor/mtp_varlen.hpp` 依赖缺失的 sequence.hpp，仅是参考材料，当前没有可执行测试。其预算估计逻辑必须在来源核实后才能作为组件移植依据。

文档入口：

- https://github.com/deepseek-ai/DeepInfer
- https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py
- https://ml-explore.github.io/mlx/build/html/index.html

验收 A：能说明现有代码的完成程度、实际可调用 API、锁定模型和资源预算；baseline 有可复现运行入口。资料发现之后持续实现，不等待人工签字。

## 1. 补齐最小可运行引擎

实现：

- 沿现有 Python 包布局补齐 `cli.py`、`engine.py`、`state.py`、模型注册与加载逻辑，保留 server 和 benchmark 已有可用部分。
- 通过 MLX-LM 文档中的 load + stream_generate 模式运行真实主模型，统一请求、输出、计时和取消接口。
- CLI 提供 doctor、models、run、serve、bench；提供明确错误与下载进度；默认服务绑定 localhost。
- 模型路径、revision、chat template 和生成配置可追踪；显式区分加载时间、prefill、decode、总请求时间。
- CPU / 非 Apple CI 能导入不依赖 Metal 的模块；真实 Metal 测试单独运行，缺硬件属于未测试，不标为通过。
- 按资源预算选择权重；预留系统与 KV 空间，超预算时提前解释，避免通过 swap 掩盖内存不足。

参考：本地 pyproject、doctor/config、server/worker；MLX-LM utils.py 与 generate.py。

验收 B：干净安装后 doctor、CLI 生成、流式服务均能运行真实模型；中英文输出、EOS、长度上限、取消、错误路径经过检查；wheel 包含必要资源。mock 仅用于控制流测试，性能数据必须来自真模型。

## 2. 自适应推测解码：首个技术贡献

实现：

- 建立三组明确命名的实现：主模型普通解码、固定草稿长度推测解码、自适应推测解码。建议内部使用 `baseline/fixed_speculative/adaptive`，更新 API、网页、metrics 和文档中的旧名称；如保留旧别名，明确其并非官方 DSpark。
- 先复用锁定 MLX-LM 的固定长度实现，验证缓存推进/回退、拒绝位置、bonus token、EOS、tokenizer 与输出边界。
- 在独立策略模块中实现每个验证周期的草稿预算选择：利用接受率、接受位置分布、草稿成本、验证成本和内存压力估计每个有效 token 的耗时。设置平滑、探索预算和回退，避免单次噪声导致震荡。
- 可选预算必须由实测吞吐曲线决定；策略不得因为只看到接受率就选择更长草稿。包含无草稿回退，以及低收益时释放草稿模型的路径。
- 若上游只支持固定长度，基于其实际循环实现隔离适配器，保留许可证及出处；禁止每个 token 重建整个 generator 或反复全量 prefill 来伪装动态调度。
- 初版先支持 greedy；温度采样另行补上正确的接受/拒绝和残差采样算法后再开放，不能把 greedy 验证宣称为采样分布保证。
- 专用 Confidence Head / MTP 权重缺失时使用可观测统计，称作 DeepInfer-inspired；只有核实移植源码、接口和参考结果后才称组件移植。

参考：MLX-LM speculative_generate_step、缓存实现，DeepInfer 部署 Q&A 的 Speculative decoding 部分及锁定源码。

验收 C：在固定 corpus 上检查完整 token 序列、EOS 和缓存回退；覆盖全接受、首位拒绝、中间拒绝、短输出和取消。真实数值差异单独诊断并报告，不能只筛选相同输出的样本计算成绩。公开消融实验：普通、固定多个长度、自适应、自动回退。

性能成功条件：在预先选定的一类真实工作负载上，相对调优后的固定推测解码也有重复可见的端到端收益，且输出正确性检查通过。若没有收益，保留实验结果，继续缓存/内存方向，不编造提升、不无限调参追求预设倍数。

## 3. 多轮 Agent 场景的缓存与内存管理

实现：

- 复用 MLX-LM 的 prompt cache 和分块 prefill 能力，增加按实际字节预算的缓存管理、淘汰和统计，区分模型权重、活跃 KV、复用缓存与临时峰值。
- 缓存 key 包括模型 revision、量化、tokenizer/template、精确 token 前缀和影响状态的配置，确保不同模型或模板不误命中。
- 多轮分支、请求取消、共享前缀、缓存克隆和淘汰不污染其他请求；草稿与目标缓存一起正确维护。
- 提供缓存预算和 prefill 分块的自动配置及显式覆盖；根据内存压力减少驻留缓存、限制请求，而非静默截断用户上下文。
- KV 量化、滑动窗口截断和 bounded replay 分别标明精度/语义影响；近似模式不能进入默认等价性能比较。
- 暂不硬搬 DeepInfer 集群缓存服务。磁盘持久化在复用收益超过恢复成本且文件版本语义明确时加入。

参考：MLX-LM models/cache.py、server.py 的缓存使用；DeepInfer Q&A 的 KV cache hits and recomputation。

验收 D：重复长前缀确实降低处理成本；冷/热/部分命中分开报告；分支会话和淘汰后输出正确；连续多轮负载的内存趋势与峰值有报告。RSS 和 Metal allocator 指标分别命名，统一内存下不直接相加。

## 4. 服务化与本地演示

实现：

- 扩展已有 FastAPI app、schema、worker，提供已实际支持的 OpenAI Chat Completions 子集与 models 接口。
- 单一模型执行所有者避免 MLX 线程竞争；有界请求队列、取消、异常恢复、流式断开及关闭行为可测试。先确保单用户可靠，再根据 profiling 引入并发优化。
- 以实际模型注册表生成 models 列表；对不支持的参数明确拒绝，不静默忽略。
- 本地 Web UI 提供模型状态、聊天、三模式比较、TTFT/生成速度/内存/接受率，以及 JSON 结果导出；用户内容作为文本渲染。
- 修复 SSE 事件终止、错误与取消语义；限制长运行事件历史，避免页面或服务无限积累内存。
- 静态前端随 wheel 分发，不要求用户安装 Node 才能使用本地推理。

参考：现有 server/app.py、schemas.py、worker.py；锁定 FastAPI/Starlette 文档与本地 API 测试。

验收 E：CLI 一条命令启动，浏览器可执行真实生成和比较；客户端断开不会让废弃任务长期占用 GPU；队列满、加载失败、取消、退出都有确定行为。公开兼容字段清单，不宣称完整 OpenAI 协议兼容。

## 5. 基准与必要的 Metal 优化

实现：

- 改进现有 benchmark runner/metrics/export。当前整个三模式比较被标 cold 的逻辑不能证明三个模式均为冷启动；用独立进程或明确重置状态分别测量。
- baseline 必须包含同版本原生 MLX-LM；固定推测解码要合理调优。不能只用本项目包装后的 baseline 证明优于 MLX。
- 数据集包含代码、中文、短问答、重复长前缀与多轮任务，预先固定开发集/验证集。正式测量使用多次重复和轮换顺序，保存全部失败与回退结果。
- 固定权重 revision、量化、模板、token 上限、采样、上下文和缓存状态；记录 commit、依赖、机器、macOS、电源模式及可获得的温度/内存压力信息。
- 指标包括模型加载、TTFT、prefill tokens/s、decode tokens/s、请求总耗时、完成 token 数、内存峰值、草稿接受率、回退率；提供配对统计及不确定性。样本不足不宣传 p95 或统计显著。
- 在较长负载中检查降频与稳定性，避免单次热身或短输出的偶然优势。
- 只有 profiling 找到热点后才加入自定义 Metal 算子。先为热点建立参考数值测试和独立 microbenchmark，再测端到端收益；没有收益就保留 MLX 路径。
- 公开报告使用匿名标准 corpus，报告包含可追踪数据文件；原始生成输入默认不采集私人内容。

参考：现有 benchmark 模块、MLX profiling/custom Metal kernel 官方文档（动手前读取锁定版本）。

验收 F：一条基准命令产出机器可读 JSON 和可展示报告；他人可按锁文件复现。至少一组本机真实完整数据，明确支持范围；未测机型标记待验证。若自适应无优势，发布真实实验结论与已验证的其他收益。

## 6. 展示网站、文档与开源入口

实现：

- 在 `site/` 构建轻量静态网站，适配 GitHub Pages 仓库子路径 `/deepinfer-mac/`。可采用 Astro 等已有成熟方案，避免为了展示站增加后端运维。
- 首页：一句定位、真实演示、实测结论、安装命令、GitHub 链接；中英文覆盖安装、架构、基准、贡献指南。
- 架构页区分 DeepInfer 思想、MLX 复用和本项目贡献。基准页从已审核的数据文件生成图表，注明样本、版本、机器和局限；不填虚构占位性能数字。
- 公网站点展示录制演示和静态数据；本地 UI 承担真实推理。网站不会默认连接访问者 localhost 或把用户提示词发送到公共服务。
- 编写 README/README.zh-CN、LICENSE、NOTICE、第三方归属、CONTRIBUTING、问题模板、兼容矩阵、路线图、复现教程和发布说明。
- 新用户路径：了解项目 → 安装 → doctor → 获取模型 → 首次生成 → 运行比较 → 提交社区结果。安装命令必须在干净环境实测。
- 准备演示视频/截图、技术文章和简洁发布文案；传播突出复现与贡献，不承诺 star 数。外部社区发帖按用户明确授权执行。

参考：项目实测 JSON、architecture 文档、GitHub Pages 官方部署文档（执行时核实）。

验收 G：网站 production build 成功；桌面和移动端视觉检查、链接检查、子路径与资源检查通过；公开数字均可回到测试报告；无真实数据的内容明确标记。Pages 权限可用时部署并检查线上地址，否则提供完整可部署产物和唯一外部缺口。

## 7. 最终验证与首版交付

- 单元测试覆盖调度/缓存/统计；集成测试覆盖 CLI/API/安装；本机真实 Metal 测试覆盖模型输出和测量。对性能代码的断言检查行为，不写镜像实现的无意义测试。
- CI 区分无硬件逻辑检查和真实 Metal 结果；构建 sdist/wheel、安装产物、检查打包网页、检查网站。
- 审查文档所列命令和 API 均实际存在；检查旧模式名称、虚构加速数字、占位代码、未使用依赖与遗漏许可证。
- 检查模型缓存、密钥、个人目录和私人 prompt 未进入发布文件；记录上游修改和来源。
- 产出版本说明、完整证据索引、已知问题、下一轮优化入口。仅在有真实数据支持时宣布性能优势。

验收 H（首版完成）：

1. 新环境可安装并在 Apple Silicon 运行真实模型。
2. baseline / fixed speculative / adaptive 可比较，未获益时正常回退。
3. 多轮缓存、资源预算、流式服务和本地 UI 可用。
4. 有可复现基准、正确性证据、真实限制与至少本机硬件报告。
5. 网站、英文/中文 README、许可证、贡献指南和发布包齐备。
6. 能明确回答：相比原生 MLX-LM新增了什么、在哪种条件下更好、哪些仍是实验。

## 后续演进（首版交付后按证据选择）

社区硬件矩阵；更多模型；非 greedy 的正确推测采样；Agent 工具调用协议；真正受益的 Metal 融合算子；具备硬件与权重条件时实现 V4.1 CED/CSA2/MTP。避免一次铺开全部研究路线。

## 长运行与续跑约定

本文件不等同于已经启动后台任务。Codex 在活跃执行会话中连续推进；需要跨会话定期恢复时，使用 Codex 本聊天 heartbeat 自动化，读取 PROGRESS 后恢复下一项工作。已有运行中的构建/实验优先检查或接续，不能重复启动。只在里程碑、完成、失败或需要用户操作时通知；无变化保持安静。首版完成后停止构建类续跑，避免无限重复 benchmark。桌面自动化依赖本机可用和账户额度，不应描述成关机后仍在执行。
