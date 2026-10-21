# DeepInfer-Mac

**DeepInfer-inspired inference for Apple Silicon, powered by MLX.** Independent community software, Apache-2.0, experimental alpha.

[简体中文](README.zh-CN.md) · [Architecture](docs/architecture.md) · [Benchmarks](docs/benchmarking.md) · [Compatibility](docs/compatibility.md) · [Contribute](CONTRIBUTING.md)

Ordinary baseline decoding is the safe default; fixed and adaptive speculation require explicit selection.

Run quantized language models locally, compare ordinary, fixed speculative and adaptive greedy decoding, and reuse exact prompt prefixes within a byte budget. The local browser demo ships in the Python wheel. This release does not claim a universal speedup; measurements and mismatches are published together. It does not port official DSpark or full DeepSeek V4.1.


**Priority showcase candidate:** Qwen3.8-27B MLX 4-bit is under model/Metal/resource validation; it is not the default. Its first scope is text baseline and exact-cache reuse. Hybrid recurrent state and a different vocabulary prevent reuse of the Qwen3-0.6B draft. Qwen3-4B remains the quick regression and speculative control; its results cannot be attributed to 27B.

## Start locally

Requires an Apple Silicon Mac, Metal and Python 3.12 or 3.13. Allow model download storage and available unified memory; 24 GB M4 Pro is the initial verification machine.

```sh
git clone https://github.com/ZEVIN17/deepinfer-mac.git
cd deepinfer-mac
uv sync --frozen --extra mlx --extra dev
uv run deepinfer-mac doctor
uv run deepinfer-mac models
uv run deepinfer-mac models --download qwen3-4b
uv run deepinfer-mac run --profile qwen3-4b --mode baseline '用中文解释推测解码。'
uv run deepinfer-mac serve
```

Open `http://127.0.0.1:8765` for chat, comparison and JSON export. First use downloads immutable model revisions from Hugging Face; weights are not part of the release. `qwen3-0.6b` is a smaller standalone baseline profile. Model and tokenizer details live in [models.lock.json](models.lock.json).

Install a built wheel with `python3 -m pip install 'dist/deepinfer_mac-0.1.0a1-py3-none-any.whl[mlx]'`. A plain install without `[mlx]` supports diagnostics and portable logic tests, but cannot generate. This repository is the installation source for the alpha; do not assume a PyPI release exists.

## Compare and reproduce

```sh
uv run deepinfer-mac run --mode fixed_speculative --draft-tokens 2 'Write a Python binary search.'
uv run deepinfer-mac run --mode adaptive --json 'Write a Python binary search.'
uv run deepinfer-mac bench --profile qwen3-4b --suite quick --repeats 3 --fixed-lengths 1 2 4 --output benchmark-results
uv run pytest -q -m 'not metal'
# Optional: deterministic Metal control tests on Apple Silicon
uv run pytest -q --run-metal -m metal
python3 site/build.py
python3 site/check.py
uv build
```

The benchmark includes same-version native MLX-LM, exact token comparisons, rotating mode order, multiple fixed lengths and separate reuse measurements. Keep every failure and numerical mismatch; do not filter the corpus to make a speed claim. [Reproduction and evidence](docs/benchmarking.md) describe measurement boundaries and available reports.

The initial M4 Pro full-corpus result does **not** establish equivalent speculative acceleration: zero failures, but only 27/36 comparison groups match across all variants. Adaptive did not beat the tuned fixed control. Baseline/native token agreement is 36/36. Complete results and the measured source archive are in `benchmarks/reports/m4-pro-qwen3-4b/`.

## Added to MLX-LM

- An experimental cost-aware draft budget with zero-draft fallback.
- Exact prefix snapshots with identity isolation, a byte budget and LRU eviction.
- A bounded single-owner local API, cancellation and a bundled chat/compare UI.
- Locked model profiles, transparent diagnostics and reproducible evidence export.

Greedy output equivalence must be checked against baseline on each workload: different matrix shapes can change floating point decisions. Only declare equivalent speedup when the entire predetermined corpus succeeds with matching full token sequences. Retain all elapsed times and label differing-output comparisons as different work; matching-subset ratios are diagnostic only. Sampling distributions, specialized confidence heads, V4.1 CED/CSA2/MTP, cluster serving and custom Metal kernels remain future work.

See [release notes](docs/release-notes.md), [roadmap](docs/roadmap.md), [NOTICE](NOTICE) and [upstream.lock.json](upstream.lock.json). Please contribute reproducible reports with model revisions, hardware and complete raw results.

Frequent profile reloads have a known RSS-retention issue despite released Metal arrays/cache; restart the service or use separate OS processes for reload-heavy sessions. Fixed-model cache stability is measured separately. See architecture resource limits.
