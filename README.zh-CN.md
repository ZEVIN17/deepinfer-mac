# DeepInfer-Mac

**让本地模型，更懂你的 Mac。** 以 MLX 为执行后端、以 DeepInfer 方法驱动自适应调度的 Apple Silicon 本地推理框架。

[English](README.md) · [展示站](https://zevin17.github.io/deepinfer-mac/zh.html) · [真实回放](https://zevin17.github.io/deepinfer-mac/phase2.html) · [兼容说明](docs/compatibility.md) · [参与贡献](CONTRIBUTING.md)

在终端运行模型，接入流式聊天 API，或打开内置浏览器工作台。探索根据草稿、验证与恢复的完整实测成本进行决策的执行方式。Qwen3.5-9B 是首要演示载体；模型与缓存能力才是框架的接口边界。

[![真实本地 9B 代码修复交互](docs/evidence/phase2-local-ui-qwen35-9b8.png)](https://zevin17.github.io/deepinfer-mac/phase2.html)

*真实 Qwen3.5-9B-8bit 请求，M4 Pro / 24 GiB。[查看录制的 SSE 响应回放](https://zevin17.github.io/deepinfer-mac/phase2.html)；展示站不执行模型推理。*

- **连贯的本地工作流。** CLI、Python、兼容 OpenAI 的聊天 API 子集与本地 UI 共用引擎。
- **可以接续的上下文。** 限额缓存、原生 MLX-LM LRU 与精确混合状态快照共同管理已消费前缀。
- **可以核查的调度。** 自适应路径将经验接受率、完整周期成本连接到 DeepInfer 派生预算决策，并由原始 C++ 参考核验。

## 原生优先，开箱可用

0.1.0a3 默认服务采用 MLX-LM 原生批处理，按模型与缓存能力自动选择串行回退。终端、流式 API 与内置浏览器工作台连接同一套执行能力。[运行与验收入口](docs/RELEASE.md) · [执行细节与实测数据](docs/PHASE4_NATIVE_RUNTIME.md)。

## 快速上手

需要支持 Metal 的 Apple Silicon Mac、Python 3.12/3.13 与 [uv](https://docs.astral.sh/uv/)。权重按锁定 revision 单独下载，为权重、活跃状态、快照及 macOS 预留内存。

```sh
git clone --branch main --single-branch https://github.com/ZEVIN17/deepinfer-mac.git
cd deepinfer-mac
uv sync --frozen --extra mlx
uv run deepinfer-mac doctor
uv run deepinfer-mac models --download qwen3.5-9b-8bit --target-only
uv run deepinfer-mac run --profile qwen3.5-9b-8bit --mode baseline '用中文解释统一内存。'
uv run deepinfer-mac serve
```

打开 **http://127.0.0.1:8765**，选择 `qwen3.5-9b-8bit`。默认 profile 是较小的 `qwen3-4b` 回归模型。baseline 支持 temperature、top-p、top-k、presence penalty 与 seed。运行 `uv run deepinfer-mac models --json` 查看 profile 元数据。

命令使用当前主分支。已有 checkout 可执行 `git switch main`、`git pull --ff-only`，再运行 `uv sync --frozen --extra mlx`；更新前保留本地修改。[v0.1.0a1](https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1) 是已发布的历史版本，0.1.0a3 仍为候选包，当前不假设 PyPI 已发布。

## 接入引擎

```python
from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.state import GenerationRequest

engine = InferenceEngine()
try:
    result = engine.generate(GenerationRequest(
        profile="qwen3.5-9b-8bit", prompt="Explain unified memory.",
        mode="baseline", max_tokens=128,
    ))
    print(result.text)
finally:
    engine.close()
```

启动 `serve` 后调用流式 API：

```sh
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b-8bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"temperature":0,"stream":true}'
```

服务提供约定的聊天 API 子集、有界排队、取消与流式输出。默认服务使用 baseline；显式配置入口已接入有界多请求执行和联合目标验证，提供逐请求取消与恢复。

## 接入你的模型

能力适配接口支持本地模型目录或精确 repo revision 注册。SmolLM2-135M-Instruct-8bit 已通过真实 baseline/native token 一致性、Dense 缓存复用、CLI 生成、API 与 SSE 验收，验证了新模型家族通过统一接口接入；其它自定义模型仍需独立验收。已有完整 MLX 模型目录时可使用：

```sh
uv run deepinfer-mac run --model /path/to/complete-mlx-model \
  --model-id my-local-model --max-tokens 32 'Hello'
```

模型按进程显式注册；自定义模型初始为未验证的 baseline 兼容性，不支持的优化会被拒绝。JSON 配置可让 CLI、Python、API 使用同一 ID。[适配器文档](docs/MODEL_ADAPTERS.md)说明身份、资源保护及显式回退。

[SmolLM2 acceptance / 验收报告](https://github.com/ZEVIN17/deepinfer-mac/blob/43d967652683817daf7eff0b7553ae5d5293739e/benchmarks/reports/phase3-model-adapter-integration.json) · [Weight identity / 权重核验](https://github.com/ZEVIN17/deepinfer-mac/blob/43d967652683817daf7eff0b7553ae5d5293739e/benchmarks/reports/phase3-smollm2-weight-audit.json)

## 模型与执行能力

“已验证”仅对应表中具体路径，不外推到 MLX-LM 支持的全部模型与优化。

| 模型 / 路径 | 本地 baseline | 自适应路径 | 验证范围 |
|---|---|---|---|
| Qwen3.5-9B-8bit | 已验证 CLI / API / UI、采样 | 独立配置，显式开启 | 首要文本演示；混合状态 |
| Qwen3.5-9B-4bit | 已验证 baseline / native 控制 | 公开入口未开启 | 同模型质量 / 内存对照 |
| Qwen3-14B-4bit | 已验证 baseline / native 控制 | 待验证 | Dense 工程对照 |
| Qwen3-4B | 已验证回归路径 | fixed / adaptive | 较小的 Dense 回归模型 |
| 可配置模型 / 缓存适配器 | SmolLM2-135M-8bit baseline / 缓存 / CLI / API / SSE 已验证 | 按能力开放 baseline | 路径 / repo 注册、透明 baseline 回退 |
| 联合验证服务 | 360M 目标 + 135M 草稿通过 35/35 项服务功能检查 | 显式开启联合预算 / 变长验证 | Dense 研发模型对；真实 token SSE、取消隔离与恢复 |

公开 9B API/UI 仍只开放 baseline。混合推测限定 greedy，通过 [DS 方法映射](docs/DS_DESIGN_MAPPING.md) 中的显式配置入口使用。模型成功加载、任务质量与优化验证分别记录。参见[模型选择](docs/PHASE2_MODEL_SELECTION.md)、[兼容说明](docs/compatibility.md)及不可变 [模型锁](models.lock.json)。

## 已验证的工程成绩

| 真实工作负载 | 结果 | 证据 |
|---|---|---|
| 小模型联合 API 服务 | 35/35 项真实模型服务检查；330 项 CPU + 48 项 Metal 测试通过 | [服务报告](https://github.com/ZEVIN17/deepinfer-mac/blob/d54f09ec709b3cb65513057e0d6f1eb54e13a1ae/benchmarks/reports/phase3-joint-service.json) |
| 9B 自适应开发轨迹 | 28/28 次成熟决策与原始 C++ 一致；128 个输出 token 与 baseline 一致 | [补充核验](benchmarks/reports/phase2-hybrid-lagged-dev-qwen35-9b8-supplemental-verification.json) |
| 9B 上下文复用 | 20 次请求；两种采样设置下 16/16 项完整 token 检查通过 | [缓存报告](benchmarks/reports/phase2-message-checkpoints-qwen35-9b8.json) |
| M4 Pro / 24 GiB 上的 9B baseline | 4,204 输入 + 256 输出；该次运行未观测到 swap 增量 | [资源检查](benchmarks/reports/context-4204-256-qwen35-9b8.json) |

每项成绩都保留模型、硬件与工作负载来源。[基准协议](docs/benchmarking.md) · [完整实测与技术边界](docs/PHASE4_NATIVE_RUNTIME.md)。

## 原生执行，更完整的工作流

默认服务直接采用 MLX-LM 原生批处理，围绕执行补齐有界准入、流式响应、取消隔离与恢复。当前版本通过 **364 项通用测试 + 95 项 Metal 测试**；原生与联合服务通过 **46/46 项真实模型检查**。独立安装的 wheel 已完成真实模型 CLI、API、SSE 与内置网页验收。

原生双请求任务的三进程耗时中位数为 **124.09 ms**，对应经审计的 SmolLM2 模型对开发工作负载。[服务报告](docs/evidence/phase4-service-verification.json) · [完整测量](benchmarks/reports/phase4-allocated-draft-summary.json) · [安装验收](docs/evidence/release-installed-wheel.json)。

## 自适应执行如何协作

```text
CLI / Python / 聊天 API / 本地 UI
                  │
            模型 + 缓存能力
                  │
          MLX / MLX-LM 执行
                  │
        草稿 → 目标验证 → 精确恢复
          ↑    接受率 + 完整成本 │
          └── DeepInfer 派生预算 ┘
```

调度路径已扩展为跨请求预算与真实变长目标验证，并接入有界 API 服务。逐请求状态、已提交 token 事件、取消隔离与恢复具有独立验收；完整成本校准和时间线驱动重叠仍按研究协议推进。构建[本地展示站](site/README.md)即可查看交互式预算图：明确标为机制示意，与实测轨迹分开。

MLX / MLX-LM 提供模型执行、采样和原生 LRU；DeepInfer 提供参考调度方法；混合状态恢复与服务集成属于 Mac 适配。本项目为独立社区实现。[架构](docs/architecture.md) · [源码映射](docs/DS_DESIGN_MAPPING.md) · [第三方归属](NOTICE) · [Apache-2.0](LICENSE)。

## 开发与复现

```sh
uv sync --frozen --extra mlx --extra dev
uv run pytest -q
uv run pytest -q --run-metal -m metal
uv run python site/build.py
uv run python site/check.py
```

[贡献指南](CONTRIBUTING.md) · [版本说明](docs/release-notes.md) · [证据索引](docs/evidence.md) · [第三阶段执行方案](docs/PHASE3_EXECUTION_PLAN.md)


欢迎贡献模型适配、Mac 硬件测量与真实应用案例。带上可复现命令，一起让本地推理更好用。
