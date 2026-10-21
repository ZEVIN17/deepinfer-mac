# DeepInfer-Mac

**Adaptive inference for Apple Silicon.** A local inference framework with MLX execution and DeepInfer-inspired scheduling.

[中文](README.zh-CN.md) · [Website](https://zevin17.github.io/deepinfer-mac/) · [Recorded demo](https://zevin17.github.io/deepinfer-mac/phase2.html) · [Compatibility](docs/compatibility.md) · [Contribute](CONTRIBUTING.md)

Run a model from your terminal, connect a streamed chat API, or use the bundled browser workspace. Explore execution that measures its own draft, verification and recovery cost. Qwen3.5-9B is our primary demonstration model; the framework is built around model and cache capabilities.

[![Actual local 9B code-repair interaction](docs/evidence/phase2-local-ui-qwen35-9b8.png)](https://zevin17.github.io/deepinfer-mac/phase2.html)

*Actual Qwen3.5-9B-8bit request on M4 Pro / 24 GiB. [Replay the recorded SSE response](https://zevin17.github.io/deepinfer-mac/phase2.html); the website runs no inference.*

- **One local workflow.** CLI, Python, a supported OpenAI-compatible chat API and local UI share the same engine.
- **State that carries forward.** Bounded context reuse, native MLX-LM LRU and exact hybrid-state snapshots preserve the consumed prefix.
- **Scheduling you can inspect.** The experimental adaptive path connects empirical acceptance and full-cycle cost to a DeepInfer-derived budget decision, checked against the original C++ reference.

## Quickstart

Use an Apple Silicon Mac with Metal, Python 3.12/3.13 and [uv](https://docs.astral.sh/uv/). Model weights download separately at locked revisions. Allow memory for weights, active state, snapshots and macOS.

```sh
git clone --branch codex/phase2-core --single-branch https://github.com/ZEVIN17/deepinfer-mac.git
cd deepinfer-mac
uv sync --frozen --extra mlx
uv run deepinfer-mac doctor
uv run deepinfer-mac models --download qwen3.5-9b-8bit --target-only
uv run deepinfer-mac run --profile qwen3.5-9b-8bit --mode baseline 'Explain unified memory.'
uv run deepinfer-mac serve
```

Open **http://127.0.0.1:8765** and select `qwen3.5-9b-8bit`. The default profile is the smaller `qwen3-4b` regression model. Baseline supports temperature, top-p, top-k, presence penalty and a seed. Run `uv run deepinfer-mac models --json` to inspect profile metadata.

These commands use the experimental source branch. Existing checkout: `git switch codex/phase2-core`, `git pull --ff-only`, then repeat `uv sync --frozen --extra mlx`. Preserve local changes before updating. [v0.1.0a1](https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1) is the historical published release; 0.1.0a2 remains a candidate, with no assumed PyPI publication.

## Build with the engine

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

With `serve` running:

```sh
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b-8bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"temperature":0,"stream":true}'
```

The service implements a documented chat API subset, bounded queuing, cancellation and streaming. Default service uses baseline; explicit experimental entries connect bounded multi-request execution and joint target verification with per-request cancellation and recovery.

## Experimental: register another model

The capability adapter accepts local model directories or exact repository revisions. SmolLM2-135M-Instruct-8bit has passed real baseline/native token matching, dense cache reuse, CLI generation, API and SSE acceptance. This verifies a new model family through the shared interface; other custom models require their own acceptance. For an existing complete MLX model directory:

```sh
uv run deepinfer-mac run --model /path/to/complete-mlx-model \
  --model-id my-local-model --max-tokens 32 'Hello'
```

Registration is explicit and per process; custom models start with unverified baseline compatibility, with unsupported optimizations rejected. A JSON configuration can register the same ID for CLI, Python and API. [Adapter reference](docs/MODEL_ADAPTERS.md) explains identity, resource guards and explicit fallback.

[SmolLM2 acceptance / 验收报告](https://github.com/ZEVIN17/deepinfer-mac/blob/43d967652683817daf7eff0b7553ae5d5293739e/benchmarks/reports/phase3-model-adapter-integration.json) · [Weight identity / 权重核验](https://github.com/ZEVIN17/deepinfer-mac/blob/43d967652683817daf7eff0b7553ae5d5293739e/benchmarks/reports/phase3-smollm2-weight-audit.json)

## Model and execution capabilities

“Verified” describes the listed path, not every model or optimization supported by MLX-LM.

| Model / path | Local baseline | Adaptive path | Scope |
|---|---|---|---|
| Qwen3.5-9B-8bit | Verified: CLI / API / UI, sampling | Experimental, explicit opt-in | Primary text demonstration; hybrid state |
| Qwen3.5-9B-4bit | Verified: baseline / native controls | Not enabled publicly | Same-model quality / memory control |
| Qwen3-14B-4bit | Verified: baseline / native controls | Validation pending | Dense engineering control |
| Qwen3-4B | Verified regression path | Experimental fixed / adaptive | Smaller dense regression model |
| Configurable model / cache adapters | SmolLM2-135M-8bit baseline / cache / CLI / API / SSE verified | Capability-gated baseline | Model path/repo registration and transparent baseline fallback |
| Experimental joint service | 360M target + 135M draft pass 35/35 service checks | Explicit joint-budget / variable-length verification opt-in | Dense research pair; actual token SSE, cancellation isolation and recovery |

Public 9B API/UI modes remain baseline. Hybrid speculation is greedy and uses the explicit experimental entry in [DS method mapping](docs/DS_DESIGN_MAPPING.md). Model loading is distinct from task quality and optimization validation. See [model selection](docs/PHASE2_MODEL_SELECTION.md), [compatibility](docs/compatibility.md) and immutable [model locks](models.lock.json).

## Verified engineering milestones

| Real workload | Result | Evidence |
|---|---|---|
| Small-model joint API service | 35/35 real-model service checks; 330 portable + 48 Metal tests pass | [Service report](https://github.com/ZEVIN17/deepinfer-mac/blob/d54f09ec709b3cb65513057e0d6f1eb54e13a1ae/benchmarks/reports/phase3-joint-service.json) |
| 9B adaptive development trace | 28/28 mature decisions match original C++; 128 output tokens equal baseline | [Supplemental verification](benchmarks/reports/phase2-hybrid-lagged-dev-qwen35-9b8-supplemental-verification.json) |
| 9B context reuse | 20 requests; 16/16 complete-token checks pass across two sampler settings | [Cache report](benchmarks/reports/phase2-message-checkpoints-qwen35-9b8.json) |
| 9B baseline on M4 Pro / 24 GiB | 4,204 input + 256 output tokens; no measured swap increase for this run | [Resource check](benchmarks/reports/context-4204-256-qwen35-9b8.json) |

These are correctness and resource milestones. Stable superiority over native MLX-LM and oMLX is not established; the competitive performance goal carries into Phase 3. [Benchmark protocol](docs/benchmarking.md) and [full Phase 2 record](docs/phase2.md) retain conditions, failed cases and historical results.

## Phase 3 execution core

The engine now combines capability adapters, bounded request ticks, joint budgets and actual variable-length verification. Experimental Python opt-ins add batched draft calls and direct dense KV trimming; default API behavior is unchanged. The dense checkpoint passed **338 portable + 76 real Metal tests**.

Three independent heldout diagnostic processes produced **18/18 complete outputs matching the native target**. Median paired-task times were 367.56 ms for our previous replay path, 279.13 ms for dense trim and 180.52 ms for native MLX-LM batching. This is a narrower internal improvement, not a competitor speed advantage. Strict numerical acceptance remains false: cycle-matched replay passed 12/24 checks (old replay 12/12, trim 0/12), despite matching next-token argmax and offsets. Dense trim stays experimental. [Protocol and results](docs/DENSE_FASTPATH_EXPERIMENT.md) · [Three-process evidence](benchmarks/reports/phase3-dense-heldout-summary.json).

## How adaptive execution fits together

```text
CLI / Python / chat API / local UI
                  │
          model + cache capabilities
                  │
          MLX / MLX-LM execution
                  │
     draft → target verify → exact recover
       ↑    observed acceptance + full cost │
       └──── DeepInfer-derived budget ──────┘
```

The experimental path now connects joint request budgets and real variable-length target verification to a bounded API service. Per-request state, committed token events, cancellation isolation and recovery have independent acceptance; complete-cost calibration and trace-driven overlap continue under the research protocol. Build the [local showcase](site/README.md) to explore its interactive budget diagram: explicitly a mechanism illustration, separate from measured traces.

MLX / MLX-LM provide model execution, sampling and native LRU. DeepInfer supplies the reference scheduling methods; hybrid recovery and service integration are Mac adaptations. This independent community project is not a full DSpark / DeepSeek V4.1 port. [Architecture](docs/architecture.md) · [Source mapping](docs/DS_DESIGN_MAPPING.md) · [NOTICE](NOTICE) · [Apache-2.0](LICENSE).

## Develop and reproduce

```sh
uv sync --frozen --extra mlx --extra dev
uv run pytest -q
uv run pytest -q --run-metal -m metal
python3 site/build.py
python3 site/check.py
```

[Contribution guide](CONTRIBUTING.md) · [Release notes](docs/release-notes.md) · [Evidence index](docs/evidence.md) · [Phase 3 execution brief](docs/PHASE3_EXECUTION_PLAN.md)

Capability adapters, request ticks / resource admission and the experimental joint verification service are implemented; cost calibration, trace-driven overlap and fair performance evaluation continue. Phase 2 reports and frozen artifacts remain unchanged.
