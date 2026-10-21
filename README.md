# DeepInfer-Mac

**Local inference for Apple Silicon, powered by MLX, with executable methods derived from DeepInfer.** Independent community project · Apache-2.0 · experimental alpha.

[中文](README.zh-CN.md) · [Phase 2 evidence](docs/phase2.md) · [Architecture and attribution](docs/architecture.md) · [Reproduce measurements](docs/benchmarking.md) · [Compatibility](docs/compatibility.md)

The current primary candidate is **Qwen3.5-9B-8bit**. Real loading, baseline generation, sampled generation, API streaming/cancellation/cache consistency and a local UI request have been checked on a 24 GiB M4 Pro. Same-model 4-bit and 14B fallback comparisons remain in progress; this is not final model-quality acceptance. Retired 27B weights have been removed, while their failed experiments remain documented.

The DeepInfer MTPVarlen-derived scheduler runs in the engine with empirical acceptance estimates. The original C++ reference checks its decisions. A hybrid-state snapshot/replay adapter has produced the same 96 tokens as baseline and fixed budgets on one real 9B prompt. **No superiority over tuned fixed decoding or two common frameworks has been established.** The joint Phase 2 acceptance goal remains open.

## Run locally

Apple Silicon, Metal and Python 3.12/3.13 are required for generation. Weights download separately at immutable revisions; leave room for active state, snapshots and the operating system.

```sh
git clone https://github.com/ZEVIN17/deepinfer-mac.git
cd deepinfer-mac
uv sync --frozen --extra mlx --extra dev
uv run deepinfer-mac doctor
uv run deepinfer-mac models --download qwen3.5-9b-8bit --target-only
uv run deepinfer-mac run --profile qwen3.5-9b-8bit --mode baseline 'Explain unified memory.'
uv run deepinfer-mac serve
```

Open `http://127.0.0.1:8765` and select the 9B candidate. The default profile remains the smaller Qwen3-4B regression model. The public showcase is static and does not connect to localhost. There is no assumed PyPI publication. Published [v0.1.0a1](https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1) is historical; the working tree prepares experimental 0.1.0a2.

Baseline supports temperature, top-p, top-k, presence penalty and a seed. For example:

```sh
uv run deepinfer-mac run --profile qwen3.5-9b-8bit --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --seed 42 --json 'What can a small local model help a developer do?'
uv run deepinfer-mac run --profile qwen3-4b --mode adaptive --adaptive-scheduler ds --no-cache --json 'Write a Python binary search.'
uv run pytest -q
uv run pytest -q --run-metal -m metal
python3 site/build.py
python3 site/check.py
uv build --out-dir dist/v0.1.0a2
python3 scripts/verify_artifacts.py --dist dist/v0.1.0a2
```

Speculative modes remain greedy. The public 9B API/UI offers baseline only; hybrid speculation requires the explicit experimental Python entry in [DS_DESIGN_MAPPING.md](docs/DS_DESIGN_MAPPING.md). It is not a trained Confidence Head, full DSpark, continuous batching or a full DeepSeek V4.1 port.

## Evidence and limits

The real 9B UI clamp repair completed in 1834.24 ms with TTFT 939.62 ms and 27.94 decode tokens/s. This is one interactive request, not a competitive benchmark. The 12-task local quality smoke retained factual/arithmetic failures; native and project greedy outputs matched on all 12 tasks. Public subsets and quantization comparisons use a separate frozen-source protocol and publish sanitized summaries, not restricted benchmark text.

Historical alpha Qwen3-4B measurements contain 216 measured runs: baseline/native agreement 36/36, all-variant equality 27/36. Adaptive did not beat tuned fixed decoding; differing outputs prevent an equivalent-speedup claim. These alpha numbers do not describe the new 9B model.

Cache lookup reuses MLX-LM's native LRU for trimmable models; hybrid models use exact snapshots and suffix replay. This cache work is independent/platform adaptation, not DS core credit. RSS and Metal allocation overlap on unified memory and are reported separately. Frequent model reloads have unresolved process-RSS retention; use separate processes for reload-heavy runs.

See [model locks](models.lock.json), [source attribution](NOTICE), [release notes](docs/release-notes.md), [evidence](docs/evidence.md) and [contributing](CONTRIBUTING.md). Complete failures, outputs, settings and numerical mismatches stay in each permitted report.
