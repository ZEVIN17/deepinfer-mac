# Phase 2 / 第二阶段：实现与证据

Working candidate: **0.1.0a2, experimental, not yet published**. The combined acceptance goal remains open: a DS-derived executable method **and** a fair advantage over two common frameworks at equal quality. Correctness, native parity and model-selection quality are separate results.

## Current model / 当前选型

Primary candidate `mlx-community/Qwen3.5-9B-8bit@16daa4818c54ce5f5436f929d52542eb65bbed9d`; both shards and 1260 tensor headers verified. Total weights 10,426,592,423 bytes; text tensors 9,514,412,544 bytes. The 9B4bit and Qwen3-14B4bit comparisons remain in progress. See [selection rationale](PHASE2_MODEL_SELECTION.md) and [quality protocol](PHASE2_QUALITY_PROTOCOL.md).

Retired 27B profiles are absent from the active API/UI. Six exact obsolete cache/chunk directories were removed, 19,177,137,641 logical bytes (17.86 GiB); completed 9B transfer chunks were removed after verified cache assembly. Unrelated models and the 4B/0.6B regression pair were retained. Historical [27B failures](qwen27.md) remain available.

## What actually ran / 实际验证

| Evidence | Result | Limit |
| --- | --- | --- |
| 9B baseline and native, 12 local quality tasks | 12/12 full greedy token equality; greedy quality 10/12, official non-thinking 11/12 | Small project-authored smoke, not public leaderboard reproduction; failures retained |
| Real 9B API/SSE/cache/cancel | Token consistency, stream concatenation, cancellation/recovery and unsupported-mode rejection passed | Engineering correctness, not capability or speed superiority |
| Actual local UI clamp repair | 939.62 ms TTFT; 1834.24 ms request; 27.94 decode tok/s; request swap delta 0 | One interactive request; does not imply no swap during model loading |
| 4B DS execution | 31 real MTPVarlen decisions; original C++ replay agrees | Mechanism evidence only |
| 9B hybrid DS execution | baseline/fixed1/fixed2/DS: same 96 tokens; 24 DS decisions agree with original C++ | One prompt, different compile warmups; not a speed benchmark |
| Three-engine service compatibility | Actual native MLX-LM, oMLX and project transport paths audited separately | Not yet a matched performance distribution or competitive acceptance |

The hybrid probe's raw DS time is 2473 ms and fixed2 is 2363 ms. The baseline's first request includes compilation. Do not divide these times for a speed claim. No DS superiority is established.

## DS method and platform changes / 方法对应

[DS_DESIGN_MAPPING.md](DS_DESIGN_MAPPING.md) maps pinned DeepInfer functions to production code, invariants, original C++ oracle checks and on/off controls. The core keeps the runtime estimators, expected-yield/cost objective and distinct N−2/N−1 acceptance histories. A trained Confidence Head is unavailable; the name is **DeepInfer MTPVarlen-derived scheduler with empirical acceptance**.

Hybrid ArraysCache/KVCache snapshot/restore/suffix replay is a Mac platform adaptation. Native MLX-LM LRU and deferred array synchronization are independent/upstream optimizations. They do not count as DS core performance gains. Public 9B UI/API remains baseline-only; the hybrid DS path is explicitly experimental Python functionality.

## Reproduce locally / 本地复现

```sh
uv run deepinfer-mac models --download qwen3.5-9b-8bit --target-only
uv run deepinfer-mac run --profile qwen3.5-9b-8bit --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --seed 42 --json 'Explain unified memory.'
uv run deepinfer-mac serve
uv run python benchmarks/experiments/hybrid_ds_probe.py --profile qwen3.5-9b-8bit --max-tokens 96
uv run python benchmarks/experiments/verify_hybrid_ds_trace.py
```

The local text API accepts the same baseline sampling settings, for example after starting the service:

```sh
curl http://127.0.0.1:8765/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.5-9b-8bit","messages":[{"role":"user","content":"Explain unified memory."}],"max_tokens":128,"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5,"seed":42,"enable_thinking":false,"stream":true}'
```

The hybrid probe requires the locked 0.8B4bit draft, sufficient memory and exclusive GPU access. See [benchmarking](benchmarking.md) for frozen-source quality and matched service comparisons. Public dataset raw prompts, scoring fixtures and generations stay outside distributable artifacts in `results/quality-cache`; only permitted summaries are published.

## Evidence links / 原始证据

- [9B UI JSON](../benchmarks/reports/phase2-ui-qwen35-9b8.json), [API engineering JSON](../benchmarks/reports/phase2-service-qwen3.5-9b-8bit.json), [sampled API JSON](../benchmarks/reports/phase2-service-sampling-qwen35-9b8.json).
- [9B hybrid DS JSON](../benchmarks/reports/phase2-hybrid-ds-qwen35-9b8-probe.json), [independent hybrid verification](evidence/phase2-hybrid-verification.json), [core verification](evidence/phase2-core-verification.json).
- [Local quality smoke JSON](../benchmarks/reports/quality-qwen3.5-9b-8bit.json). Source changed during this initial adaptation run; subsequent formal comparisons require a frozen archive.
- Historical alpha [benchmark method/results](benchmarking.md), [published release](https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1) and [public site](https://zevin17.github.io/deepinfer-mac/). Alpha 4B graphs do not represent current 9B results.

中文：已完成 DS 机制落地、原始 C++ 对照和主模型真实正确性控制，但尚无两个常用框架上的公平性能优势。正式质量/量化选择、独占 GPU 同条件消融和竞争服务分布测量仍需完成；不以阶段性实现替代联合验收。
