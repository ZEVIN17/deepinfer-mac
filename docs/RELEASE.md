# DeepInfer-Mac 0.1.0a3

Run local models through one Apple Silicon workflow: CLI, a streamed chat API and an included browser workspace. Native MLX-LM batching supplies the default execution path; DeepInfer-derived scheduling, cache budgets and request lifecycle controls make execution inspectable.

## Run this checkout

```sh
uv sync --frozen --extra mlx
uv run deepinfer-mac doctor
uv run deepinfer-mac models --download qwen3-4b --target-only
uv run deepinfer-mac run --profile qwen3-4b --max-tokens 64 'Explain unified memory.'
uv run deepinfer-mac serve --port 8876
```

Open <http://127.0.0.1:8876>. The bundled UI is included in the wheel; Node.js is not required. Model downloads are separate. For the already audited small local model in this workspace:

```sh
uv run deepinfer-mac serve --port 8876 --model-config examples/models/smollm2-135m-8bit.local.json
```

Select `smollm2-135m-8bit` in the UI. Its source directory must exist; these example local paths are not downloadable weights. A clean installation can instead use the locked profile above, or register its own MLX snapshot with `--model` / `--model-config`.

## What the framework adds

| Layer | Delivered behavior |
| --- | --- |
| Native execution | Pinned MLX/MLX-LM, native batched greedy generation and capability-based serial fallback |
| Request lifecycle | Bounded admission, context reservation, cancellation, SSE completion and error recovery |
| Multi-turn state | Serial native LRU / exact snapshots with byte budgets, branch-safe reuse and resource checks |
| Scheduling | Fixed/adaptive greedy modes, measured cycle accounting, original C++ scheduler oracle, allocated joint draft work |
| Product interfaces | CLI, Python, supported Chat Completions subset, bundled UI, metrics and JSON exports |
| Reproducibility | Locked dependencies/models, complete measurements, tests, bilingual showcase and license attribution |

Native batching currently does not reuse the serial prefix cache. Use `--execution-backend serial` when testing prefix-cache behavior. The Python `create_app()` default remains serial for compatibility; the CLI `serve` default is auto/native. Joint service is an explicit separate configuration. Automatic selection of speculation and strict dense-trim numerical equivalence are not enabled in the default service. Sampling is supported on the serial baseline path; speculative modes are greedy. Repeated model reloads may retain process RSS, so prefer a fixed model service. See [compatibility](compatibility.md) and [execution report](PHASE4_NATIVE_RUNTIME.md).

## Real measurements

The current native choice follows three independent development processes on the audited SmolLM2 360M/135M pair: paired-task medians **124.09 ms native**, **760.56 ms full draft**, **529.84 ms allocated draft**. All tested outputs matched. Allocated work reduced internal task time by 30.3%; the default uses the faster native executor. These are this workload's measurements, not a general speed promise or an oMLX comparison. [Complete report](../benchmarks/reports/phase4-allocated-draft-summary.json).

The historical Qwen3-4B corpus contains 216 real measured runs, zero failures, native/baseline equality in 36/36 groups and speculative all-variant equality in 27/36. It remains linked with its original source and parameters; it is not a fresh measurement of this release. [Benchmark guide](benchmarking.md).

## Verify and package

```sh
./scripts/release-check.sh dist/current
uv run pytest -q -m metal --run-metal  # Apple Silicon with Metal access
uv run python benchmarks/experiments/phase4_allocated_draft.py --help
uv run python site/build.py
uv run python site/check.py
python3 -m http.server 8088 --directory site/dist
```

`dist/current/` contains the current wheel and source archive. `site/dist/` is the deployable bilingual website. CI rebuilds and uploads both on push; Pages deploys on main. Artifact names and hashes are recorded in [release verification](evidence/release-0.1.0a3.json). Historical releases and measured archives are preserved. The repository push does not imply a PyPI upload.

## Acceptance evidence

- [Native/allocated actual-model service verification](evidence/phase4-service-verification.json): 46/46 checks, including SSE, cancellation, concurrency, recovery and native output comparisons.
- [Release tests, installed-wheel smoke and package verification](evidence/release-0.1.0a3.json).
- [Cache, API, UI and hardware evidence](PHASE3_DELIVERY.md), [source audit](PHASE4_SOURCE_MAP.md), [upstream attribution](../NOTICE).
- [Detailed native runtime measurements](PHASE4_NATIVE_RUNTIME.md), [complete benchmarks](benchmarking.md).

中文：默认服务采用原生批处理，围绕模型执行补齐请求管理、流式接口、本地工作台、缓存预算与可复现实测。安装、运行、验证命令见上；性能数字对应具体模型与工作负载，原始报告保留完整结果。
