# Phase 2：缓存、三服务兼容与正式对照入口

当前状态：三服务9B8bit真实兼容已通过；尚无正式竞争加速结论。消息边界检查点已接入生产prefill路径，但为**默认关闭的实验开关**，仅通过portable控制，需真实Metal采样/缓存一致性验证后才能参与benchmark。CPU阶段没有启动新的GPU任务，也没有改动冻结质量归档。

## 当前运行入口

- 独立oMLX环境：`/tmp/deepinfer-omlx-venv`，完整110包依赖检查通过。官方服务已实际加载过锁定9B8bit，非仅安装成功。
- 三服务启动器：`benchmarks/experiments/{deepinfer_service_server,native_lru_server,omlx_service_server}.py`。每次只允许一个模型服务，停止自有进程组后再启动下一个。
- 兼容报告：`benchmarks/reports/phase2-three-service-compat-qwen35-9b8.json`；完整实际本机路径仅存打包排除的`benchmarks/results/`，公开报告使用`<WORKSPACE>/<MODEL_SNAPSHOT>/<HOME>`。
- 正式对照组件：`prepare_service_compare.py`生成本机配置，`rotate_service_compare.py`独占启动、轮换、热身与原始采集，`summarize_service_compare.py`按session聚类bootstrap。
- 数据/配置未冻结或GPU未明确交接时，不启动正式对照。旧`/tmp`配置可能对应较早源码，须重新准备；轮换器会拒绝runtime source hash变化及超过声明总上下文上限的请求。

先验证消息检查点，以下命令仅在取得GPU后执行：

```sh
PYTHONPATH=src .venv/bin/python benchmarks/experiments/message_checkpoint_validation.py --run-metal --sampler both
```

只用v2的首个2K dev session，分别验证greedy(seed0)与官方non-thinking(seed42)；每种10请求，共20请求，预计约2–3分钟，实际耗时以运行记录为准。`--sampler official`减为10请求，约1–2分钟。比较关闭/开启检查点时cache=false、冷缓存、热缓存、改尾的完整真实生成IDs，保留失败。输出默认`benchmarks/reports/phase2-message-checkpoints-qwen35-9b8.json`。这是正确性验证，不计算竞争速度比。

正式配置示例（CPU准备，不加载模型）：

```sh
.venv/bin/python benchmarks/experiments/prepare_service_compare.py --root /tmp/deepinfer-service-compare-v2 --manifest benchmarks/experiments/cache_sessions.v2.json --total-context-cap 4096 --snapshot "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-8bit/snapshots/16daa4818c54ce5f5436f929d52542eb65bbed9d"
```

v2包含超过初始4096总cap的组，因此不能直接对全manifest启动。后续需根据实际资源probe声明阶段准入和暂未运行的组；不得静默截断或把未运行记为零分。最终模型/量化由质量选型决定，9B4bit若胜出须统一重建三方配置，不能沿用8bit结果。

## 已验证的真实兼容证据

按oMLX→native→本项目依次运行，三次SSE请求全部完成，各服务进程组均已退出。相同9B8bit权重revision、纯文本路径；首请求1514个实际prompt IDs完全相同，little-endian uint32 SHA256为：

`20e037ba3f13f748f8f1a7ca0c4e6dceff4beb36d0680de09c08c90ea7b82726`

三种请求的输出文本也逐字相同。此轮开启token审查、顺序未轮换且源码仍在变，只作兼容证据。

| 服务 | 重复命中 | 改尾命中 | 真实输出计数来源 |
|---|---:|---:|---|
| oMLX | 1507 | 0 | `stream_options.include_usage=true`，116 completion tokens |
| MLX-LM0.32.2 | 1513 | 1480 | 同上，117 completion tokens |
| 本项目native cache/512 checkpoint | 1513 | 1024 | 完成后run endpoint的117个实际生成IDs及metrics |

116/117的口径差异尚未用两方完整生成IDs定位，不擅自归因EOS。没有用文本重新tokenize或chunk数代替生成IDs/ITL。oMLX实际scheduler确认`hot_cache_only=true`、`hot_cache_max_size=536870912`、`block_aware_cache_active=true`；SSD容量字段存在但该磁盘层未启用。原生适配器实际强制max_bytes512MiB，保留system/user分段、batch与最长前缀能力。

oMLX默认prefill2048，本项目/native为512；本次oMLX还因实际内存guard将一段1507缩到1024，日志保留。不能称三方分块完全相同，也不能忽略这些条件作性能比。

项目服务驻留时额外完成官方non-thinking采样API/SSE、同seed原生控制IDs一致、422模式拒绝、断开取消与恢复：`benchmarks/reports/phase2-service-sampling-qwen35-9b8.json`。malformed JSON在oMLX/本项目返回422、native返回400。未给不支持的9B hybrid fixed模式虚构零分。

## 保留的负结果与v2预注册

v1首个dev session的三种请求×三引擎，共9次请求全部因Markdown围栏而**严格JSON失败**；它们不是9道独立题。完整输出与失败标记保留，没有strip围栏后重评分。

| 不可混用的输入版本 | SHA256 | 状态 |
|---|---|---|
| `cache_sessions.v1.json` | `1922cc460a93c2cb5d9f615bc40ca77449e04dd520fad8c6129f42f5d7b2bcc1` | 负结果保留 |
| `cache_sessions.v2.json` | `f891ff522edc120731781f5a0e8f3b211774b4dfea696af1690ec8f4c5d81ad5` | CPU校准冻结，未运行v2 GPU/heldout |

仅依据v1 dev，v2对所有引擎及dev/heldout统一追加“首字符{、末字符}、禁止Markdown/围栏/前言/结语”；严格JSON评分不变，未观察heldout输出来修改提示。v2以实际锁定tokenizer/template校准，`enable_thinking=false`，显式`return_dict=false`以避免把Transformers5 BatchEncoding的字典长度当token数。每个session保留同一个system前缀以验证真实续聊。

| 输入组 | seed/重复/改尾输入 | 续聊输入 | 输出预算 | 续聊总预算 |
|---|---:|---:|---:|---:|
| 2K | 2048 | 2156 | 256 | 2412 |
| 4K | 4096 | 4204 | 256 | 4460 |
| 8K | 8192 | 8300 | 256 | 8556 |

初始4096总上下文阶段仅允许2K组。登记profile总cap8192时不足以容纳8K输入加输出；8K组需独立资源probe及显式上限调整后准入。后续上限/硬件结果以registry与资源报告为准，本manifest不自动提升上限。校准明细：`artifacts/phase2-competitor-audit/cache-manifest-v2-calibration.json`；重现脚本：`freeze_cache_manifest_v2.py`。

输入为公开项目记录，3 dev/6 heldout，中文/英文，严格四条记录JSON提取；固定参考assistant回答使后续请求跨引擎相同，参考回答不算模型成功。这套缓存任务不替代独立代码/IFEval/知识质量集。

## 消息边界检查点：实现与剩余验证

改尾原生1480而本项目1024，说明需要追平消息分段能力。`session_boundaries.message_boundaries`只返回与实际full IDs相同的proper prefixes。Qwen3.5拒绝system-only渲染时，参考MLX-LM空user探针取真实公共前缀；dummy tokens不会送入模型或cache key。CPU实际位置：v1 system1480/user1507；v2 system2014/user2041，见`artifacts/phase2-competitor-audit/message-boundaries-tokenizer-check.json`。

`InferenceEngine(message_checkpoints=True, cache_backend="native")`已接入实际prefill；默认False。开启时仅cache=true按真实边界拆分，模型消费后`mx.eval`再clone快照，透传system/user/assistant eviction优先级。常规chunk网格保留；cache=false或关闭开关时仍用原chunk计算，不额外渲染partial messages。native_mlx委托上游prefill，不套用该分块器。服务适配器可显式传`--message-checkpoints`，正式基准尚不得默认开启。

metrics包含实际边界、写入offset/role、host tokenizer/快照成本、写入逻辑bytes、`prefill_shapes_changed`及需真实采样验证标记。逻辑bytes可能共享不可变MLX数组，不是物理RSS；host时间不冒充GPU时间。

Portable生产prefill控制覆盖：关闭及cache=false原shape、nontrimmable target+draft精确状态、分段顺序、改尾仅重放真实后缀、分支隔离、full-hit留最后输入token。相关16通过，ruff通过；这些确定性数组控制不证明真实量化模型数值等价。冷prefill分块变化可能影响greedy/采样，必须先跑上面的真实控制并保留分歧，才能用于对照或考虑默认化。

缓存复用、消息分段和此平台适配属于**独立基础优化/MLX能力追平**，不计DS核心贡献。没有关闭对手缓存来制造优势。

## 竞品来源与正式实验约束

- [oMLX v0.7.1](https://github.com/jundot/omlx/releases/tag/v0.7.1)，commit `bd1e9b112474862ebed550a9c83b4316038a1d74`。官方wheel39,925,267 bytes，SHA256 `876cff780943d679a0127be5509ab2878006e62281e7b0ec6fcba8d754484c14`。
- 官方依赖：MLX0.32.4；MLX-LM commit `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`，实际scm版本`0.31.4.dev132+g94cdcae13`；mlx-vlm0.7.8；mlx-audio0.5.7。它与本项目MLX-LM0.32.2不同，完整版本/freeze/help在`artifacts/phase2-competitor-audit/`。
- 安装时官方PyPI直连不可达，registry依赖通过清华PyPI镜像；audio改为**同一官方commit**归档传输，原源码/wheel未修改且未省略依赖。归档URL：`https://api.github.com/repos/Blaizzy/mlx-audio/tarball/94c7716212b2228f178d2f9c7619a591fd1b0b78`，SHA256 `b5b7aba848d40b60921b393023c84aa199c9cca09cef525b3ecdac83f6225560`。实际来源见`omlx-environment.json`。
- 原生server0.32.2虽解析`--prompt-cache-bytes`，run只传max_size；`native_lru_server.py`只补max_bytes，不换生成器、不关合理分段。升级版本会拒绝未复核适配器。
- oMLX使用官方`model_type_override=llm`文本路径、`enable_thinking=false`、RAM hot-only512MiB；保留成熟缓存能力，SSD另列产品场景。其Qwen3.5 MTP/精确恢复源码是oMLX/MLX工程参考，不当作DS来源或本项目已实现能力。

正式实验必须同模型repo/revision/量化、模板、采样、缓存预算、请求与并发；dev调优后冻结。每引擎独占GPU热身、轮换完整block，冷加载/暖模型/热KV分开，额外审查开关关闭。bootstrap以session为单位，保留全部质量失败和缺失；重复次数不是独立题。≥15%为待验目标，需DS开关贡献及两个主要对照共同支持，通用缓存改进不能单独完成目标。
