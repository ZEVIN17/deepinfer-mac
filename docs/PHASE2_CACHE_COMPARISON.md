# Phase 2：缓存、三服务兼容与正式对照入口

当前状态：三服务9B8bit真实兼容已通过；尚无正式竞争加速结论。消息检查点仍默认关闭，但root已用冻结源码完成双sampler共20请求、16项token检查全部一致及prompt长度一致性验证。v3仅CPU预注册新增两个独立heldout会话，使资源准入的2K+4K合计六个会话；8K因实际swap增长暂缓。尚未观察v2/v3 heldout输出。本轮准备没有启动GPU或改动冻结质量归档。

## 当前运行入口

- 独立oMLX环境：`/tmp/deepinfer-omlx-venv`，完整110包依赖检查通过。官方服务已实际加载过锁定9B8bit，非仅安装成功。
- 三服务启动器：`benchmarks/experiments/{deepinfer_service_server,native_lru_server,omlx_service_server}.py`。每次只允许一个模型服务，停止自有进程组后再启动下一个。
- 兼容报告：`benchmarks/reports/phase2-three-service-compat-qwen35-9b8.json`；完整实际本机路径仅存打包排除的`benchmarks/results/`，公开报告使用`<WORKSPACE>/<MODEL_SNAPSHOT>/<HOME>`。
- 正式对照组件：`prepare_service_compare.py`生成本机配置，`rotate_service_compare.py`独占启动、轮换、热身与原始采集，`summarize_service_compare.py`按session聚类bootstrap。
- 数据/配置未冻结或GPU未明确交接时，不启动正式对照。旧`/tmp`配置可能对应较早源码，须重新准备；轮换器会拒绝runtime source hash变化及超过声明总上下文上限的请求。

消息检查点已验证入口（将来复验仍须独占GPU）：

```sh
PYTHONPATH=src .venv/bin/python benchmarks/experiments/message_checkpoint_validation.py --run-metal --sampler both
```

只用v2的首个2K dev session，分别验证greedy(seed0)与官方non-thinking(seed42)；每种10请求，共20请求，预计约2–3分钟，实际耗时以运行记录为准。`--sampler official`减为10请求，约1–2分钟。比较关闭/开启检查点时cache=false、冷缓存、热缓存、改尾的完整真实生成IDs，保留失败。输出默认`benchmarks/reports/phase2-message-checkpoints-qwen35-9b8.json`。这是正确性验证，不计算竞争速度比。

正式dev配置示例（CPU准备，不加载模型；目标仍为质量更稳的9B8bit，不以较低质量4bit替换）：

```sh
.venv/bin/python benchmarks/experiments/prepare_service_compare.py --root /tmp/deepinfer-service-v3-frozen-dev --python "$PWD/.venv/bin/python" --manifest benchmarks/experiments/cache_sessions.v3.json --resource-approved-lengths 2048 4096 --total-context-cap 8192 --variants baseline ds-original ds-lagged --dev-tuning-grid --snapshot "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-8bit/snapshots/16daa4818c54ce5f5436f929d52542eb65bbed9d"
```

该命令复制完整服务runtime、全部experiment/helper/manifest及locks到`source/`并生成可复现`service-source.tar.gz`，执行解释器单独指定；安装的两个环境版本及实际文件字节另行指纹，不导入MLX。已有不同source extraction或普通model alias目录会拒绝并保留原目录，须选新root。model alias仅接受解析到精确snapshot的symlink，运行每次复查。先做纯CPU检查：

```sh
PYTHONPATH=/tmp/deepinfer-service-v3-frozen-dev/source/src .venv/bin/python /tmp/deepinfer-service-v3-frozen-dev/source/benchmarks/experiments/rotate_service_compare.py --configuration /tmp/deepinfer-service-v3-frozen-dev/configuration.json --manifest /tmp/deepinfer-service-v3-frozen-dev/source/benchmarks/experiments/cache_sessions.v3.json --verify-only
```

root/quality明确交接GPU且I/O空闲后，对同一冻结入口去掉`--verify-only`，追加`--split dev --repeats 3 --output "$PWD/benchmarks/reports/phase2-service-v3-dev-grid"`。不能调用live runner/helper/manifest，service的argv、cwd及PYTHONPATH也绑定此extraction。每block前后与最终检查实际源码、archive和解释器环境字节，证据写入raw，drift即禁止目标通过。这些检查不计请求timing。正式heldout前仍需4K 4204+256实际资源门槛；准入清单是预注册选择，不代替资源实测。

完整server日志和private raw只写`benchmarks/results/private-service/<run>/`；公开JSON仅包含`<PRIVATE_RESULTS>`别名、字节数和SHA256，不携带`.log`或私有traceback原文。汇总器从configuration、manifest、split及重复数重建完整顺序清单，核对所有block、请求kind/session顺序、warmup、实际payload、正且有限的timing与source evidence；重复、缺失、额外、失败和顺序异常会保留诊断，但target=false。旧不具备完整证据的report仍可读作部分诊断，不能补造target通过。

此grid为17种独立候选：本项目baseline/DS original/DS lagged各128/256/512/1024，native同四候选，oMLX保留官方2048与内存guard。轮换器拒绝grid运行heldout。先按严格质量/协议筛选，再分别以dev总耗时最低选择每引擎/模式chunk，TTFT作同分决胜；保留全部负结果，不假定相同chunk最优。正式运行前去掉`--dev-tuning-grid`，用`--own-prefill-step-size`、`--ds-original-prefill-step-size`、`--ds-lagged-prefill-step-size`、`--native-prefill-step-size`分别填实际选择并重新冻结源码/配置。调度贡献另需同chunk、同draft、同循环的on/off消融，不能用不同调参的胜负归因DS。

dev grid的3重复是探索，不构成竞争收益结论；完整grid为816请求（含408排除的warmup）。正式五variant对照以5次重复完成一个完整顺序轮换周期（`--repeats 5`，每variant五次），使每个variant处在每个顺序位置；记录`order_balance`，不完整位置轮换不会标记工程目标通过。首次正式heldout共800请求，其中200为排除的warmup；重复仍只有六独立heldout clusters，不变成30个。

`--message-checkpoints`仅显式选择；默认关闭。此基础优化可引用20请求正确性证据，不能把其中timing当竞争结果。DS服务使用真实`--experimental-hybrid-speculation --adaptive-scheduler ds --hybrid-cost-adapter original|lagged_mean`，请求`mode=adaptive`；局部`/v1/models`会标experimental，公开默认服务仍baseline。性能配置显式记录`round_diagnostics=false`，使用适配器默认False且不加`--round-diagnostics`，保留轻量预算/接受计数；heavy replay/profile另跑，不能混进竞争timing。两DS配置额外草稿`mlx-community/Qwen3.5-0.8B-4bit@da28692b5f139cb0ec58a356b437486b7dac7462`，文件625,229,487 bytes，SHA256 `f5a0d9dd3efa73510542a8023d610ff26be2b4b020d181cfc4bedaa1fcc5dd9e`，成本计入完整耗时/资源，不声称三方总驻留内存相等。

配置显式保存全部selected/deferred会话，8K三会话仍在manifest与报告中，不截断、不记零分；runner校验清单、总cap、源码和harness哈希。当前并发1；并发2/4尚需独立资源准入及真实测试，本项目FIFO不是连续batch。9B4bit不能与8bit混作纯引擎对照。

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
| `cache_sessions.v2.json` | `f891ff522edc120731781f5a0e8f3b211774b4dfea696af1690ec8f4c5d81ad5` | 输入冻结；首2K dev用于正确性，heldout未运行 |
| `cache_sessions.v3.json` | `16750b5d512df3d88a0d9086f1b2070dc5ebacffb047cc2f512cda35af03deb7` | CPU预注册11评分会话+2独立warmup；GPU/dev调优/heldout均未运行 |

仅依据v1 dev，v2对所有引擎及dev/heldout统一追加“首字符{、末字符}、禁止Markdown/围栏/前言/结语”；严格JSON评分不变，未观察heldout输出来修改提示。v2以实际锁定tokenizer/template校准，`enable_thinking=false`，显式`return_dict=false`以避免把Transformers5 BatchEncoding的字典长度当token数。每个session保留同一个system前缀以验证真实续聊。

| 输入组 | seed/重复/改尾输入 | 续聊输入 | 输出预算 | 续聊总预算 |
|---|---:|---:|---:|---:|
| 2K | 2048 | 2156 | 256 | 2412 |
| 4K | 4096 | 4204 | 256 | 4460 |
| 8K | 8192 | 8300 | 256 | 8556 |

初始4096总上下文阶段仅允许2K组；当前正式准备声明总cap8192、只准入2K/4K组。实际8300+256探测已完成但swap增加约2.60GiB，见`benchmarks/reports/context-8300-256-qwen35-9b8.json`，不将它算作稳定8K竞争性能证据。v2校准明细：`artifacts/phase2-competitor-audit/cache-manifest-v2-calibration.json`。

v3在未观察任何heldout模型输出前扩展：原v2全部9会话与cases原样保留（含8K），dev不变；增加`public-ledger-09-zh`（108行，2K）与`public-ledger-10-en`（221行，4K），使用不同记录种子与答案，不是重复测量。准入2K/4K各3个heldout，共6独立session clusters；单一length只有3个，不单独满足六cluster门槛。新增输入seed/重复/改尾2048或4096，续聊2156或4204，输出预算256，全部实际token IDs哈希在v3内。六cluster门槛、≥15%目标、严格JSON评分均未降低。

每个引擎/variant/block评分前完整执行独立`public-ledger-11-zh`与`public-ledger-12-en`的2K/4K、seed/重复/改尾/续聊共8请求热身，以覆盖准入长度与路径。两者不属于评分/dev/heldout会话，结果单列`warmups`，输入及token哈希同样冻结。`seed_cold_kv`仅为历史kind标签，不能证明零命中；`cache_observation`只读取实际run metrics或SSE usage中的cached_tokens，无此数据则null，不声称warmup后缓存为空。重现只需CPU：

```sh
.venv/bin/python benchmarks/experiments/freeze_cache_manifest_v3.py --snapshot "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-8bit/snapshots/16daa4818c54ce5f5436f929d52542eb65bbed9d"
```

输入为本项目公开、Apache-2.0的合成ledger任务，合成的是输入及确定参考答案，没有模拟性能或模型输出。固定参考assistant回答使续聊输入跨引擎相同，参考回答不算模型成功；这套事实提取缓存任务不替代独立代码/IFEval/知识质量集。

## 消息边界检查点：实现与剩余验证

改尾原生1480而本项目1024，说明需要追平消息分段能力。`session_boundaries.message_boundaries`只返回与实际full IDs相同的proper prefixes。Qwen3.5拒绝system-only渲染时，参考MLX-LM空user探针取真实公共前缀；dummy tokens不会送入模型或cache key。CPU实际位置：v1 system1480/user1507；v2 system2014/user2041，见`artifacts/phase2-competitor-audit/message-boundaries-tokenizer-check.json`。

`InferenceEngine(message_checkpoints=True, cache_backend="native")`已接入实际prefill；默认False。开启时仅cache=true按真实边界拆分，模型消费后`mx.eval`再clone快照，透传system/user/assistant eviction优先级，warm full-prefix命中也保留已验证角色。常规chunk网格保留；cache=false或关闭开关时仍用原chunk计算，不额外渲染partial messages。native_mlx委托上游prefill，不套用该分块器。

metrics包含实际边界、写入offset/role、host tokenizer/快照成本、写入逻辑bytes、`prefill_shapes_changed`及需真实采样验证标记。逻辑bytes可能共享不可变MLX数组，不是物理RSS；host时间不冒充GPU时间。

Portable控制覆盖关闭/cache=false原shape、nontrimmable target+draft精确状态、分段顺序、真实后缀重放、分支隔离、full-hit留尾token与warm边界角色。真实冻结源码20请求报告`benchmarks/reports/phase2-message-checkpoints-qwen35-9b8.json`的16项检查全相同；只覆盖首2K dev的两sampler，不外推所有长度/任务数值等价，也不证明性能收益。

缓存复用、消息分段和此平台适配属于**独立基础优化/MLX能力追平**，不计DS核心贡献。没有关闭对手缓存来制造优势。

## 竞品来源与正式实验约束

- [oMLX v0.7.1](https://github.com/jundot/omlx/releases/tag/v0.7.1)，commit `bd1e9b112474862ebed550a9c83b4316038a1d74`。官方wheel39,925,267 bytes，SHA256 `876cff780943d679a0127be5509ab2878006e62281e7b0ec6fcba8d754484c14`。
- 官方依赖：MLX0.32.4；MLX-LM commit `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`，实际scm版本`0.31.4.dev132+g94cdcae13`；mlx-vlm0.7.8；mlx-audio0.5.7。它与本项目MLX-LM0.32.2不同，完整版本/freeze/help在`artifacts/phase2-competitor-audit/`。
- 安装时官方PyPI直连不可达，registry依赖通过清华PyPI镜像；audio改为**同一官方commit**归档传输，原源码/wheel未修改且未省略依赖。归档URL：`https://api.github.com/repos/Blaizzy/mlx-audio/tarball/94c7716212b2228f178d2f9c7619a591fd1b0b78`，SHA256 `b5b7aba848d40b60921b393023c84aa199c9cca09cef525b3ecdac83f6225560`。实际来源见`omlx-environment.json`。
- 原生server0.32.2虽解析`--prompt-cache-bytes`，run只传max_size；`native_lru_server.py`只补max_bytes，不换生成器、不关合理分段。升级版本会拒绝未复核适配器。
- oMLX使用官方`model_type_override=llm`文本路径、`enable_thinking=false`、RAM hot-only512MiB；保留成熟缓存能力，SSD另列产品场景。其Qwen3.5 MTP/精确恢复源码是oMLX/MLX工程参考，不当作DS来源或本项目已实现能力。

正式实验必须同模型repo/revision/量化、模板、采样、缓存预算、请求与并发；dev调优后冻结。每引擎独占GPU热身、轮换完整block，冷加载/暖模型/热KV分开，额外审查开关关闭。bootstrap以session为单位，保留全部质量失败和缺失；重复次数不是独立题。≥15%为待验目标，需DS开关贡献及两个主要对照共同支持，通用缓存改进不能单独完成目标。

oMLX优化能力另需核验：安装的官方0.7.1源码明确支持Qwen3.5-family DFlash，默认关闭；需要匹配目标hidden-state/layer配置的训练draft checkpoint，经已安装mlx-vlm loader绑定，不等同于普通0.8B自回归草稿。它只读取显式repo/path或target内`dflash/`，锁定9B8bit仓库没有捆绑draft；本次本地源码审查尚未核实公开9B训练checkpoint的pin/大小。证据与下一步见`docs/evidence/phase2-omlx-dflash-source-audit.json`。应先核公开候选metadata/license，再协调下载和真实dev质量/资源/速度；可用优化应纳入oMLX主要对照或另列，不得以default baseline对比概括其完整能力，也不给未验证优化记零分。
