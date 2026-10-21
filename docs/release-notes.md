# 0.1.0a2 — experimental candidate, unpublished

- Phase 3 adds capability adapters, bounded continuous request ticks, joint confidence/cost budgets and actual variable-length target verification. An explicit joint service passed 35/35 real-model checks, including SSE and cancellation recovery.
- Dense Python research opt-ins add batched draft calls and direct KV trimming. The source checkpoint passed 338 portable and 76 actual Metal tests. Three later diagnostic processes matched 18/18 complete outputs; strict numerical acceptance remains false and native MLX-LM batching remains faster. See [dense protocol](DENSE_FASTPATH_EXPERIMENT.md) for scope and exact results.
- Installable candidate `v0.1.0a2-d3c6ae1` binds runtime checkpoint `d3c6ae1320f09f0d21813931c6df488cf4b4909b`; later v2/heldout reports are separate evidence, not packaged runtime claims. [Package verification](evidence/phase3-dense-checkpoint-package.json).
- Primary candidate Qwen3.5-9B-8bit with real loading, baseline/sampled API/SSE/cancellation/cache and local UI evidence; final 8bit/4bit/14B quality/efficiency choice remains pending.
- Retired27B active profiles/unused weights removed; failures/audits retained. No publisheda1 package bytes changed.
- Executable DeepInfer MTPVarlen-derived scheduler, original C++ oracle, empirical acceptance adapter and complete-cycle cost accounting.
- Experimental exact hybrid-state recovery and real96-token four-mode equality/24-decision C++ replay control on9B. Public9B API/UI remains baseline-only.
- Native MLX-LM LRU reuse and deferred draft synchronization are separately attributed baseline improvements.
- Baseline temperature/top-p/top-k/presence/seed support; speculative modes remain greedy.

**Acceptance remains open:** no demonstrated superiority over tuned fixed decoding or two common frameworks. Small probes, altered compilation warmups and quality smoke are not competitive acceptance. Public subset comparisons and final candidate artifacts require frozen source and independent verification. Reload-related process RSS retention remains known.

[Current evidence](phase2.md) · [DS source mapping](DS_DESIGN_MAPPING.md) · [Reproduction](benchmarking.md)

---

# 0.1.0a1 — experimental alpha

Independent DeepInfer-inspired Apple Silicon inference powered by pinned MLX / MLX-LM. Adds greedy baseline/fixed/adaptive comparisons, exact prefix snapshots with an LRU byte budget, a bounded local API worker, bundled chat/compare UI, CLI diagnostics and real result exports.

This is an early, single-user release. Performance depends on draft/target compatibility and workload. Adaptive decoding is experimental and can be slower than fixed or ordinary decoding. Numerical shape differences can change greedy argmax; consult complete equivalence data before interpreting throughput. No sampling distribution guarantee, official DSpark, full V4.1, distributed KV store or custom Metal kernel is claimed.

Release preparation includes source/wheel builds, a static bilingual GitHub Pages showcase, dependency/model/upstream locks, portable tests and separately recorded Metal evidence. Successful checks and real reports are recorded in `docs/PROGRESS.md`; pending checks must not be represented as completed.

中文：首版提供真实 greedy 引擎、三模式比较、精确前缀缓存、有界 API、本地 UI、锁文件和基准导出。自适应不保证更快，数值差异需完整披露。验收证据以 PROGRESS 索引为准。

Measured release result: full Qwen3-4B corpus, 216 measured runs and zero failures; all-variant token equality27/36 groups, native/ordinary baseline equality36/36. Adaptive did not improve on the tuned fixed control. No equivalent speculative speedup is claimed. Raw outputs, mismatch diagnostics and a hash-verified snapshot of the measured source are preserved in `benchmarks/reports/m4-pro-qwen3-4b/`.

The release defaults to ordinary baseline decoding across CLI/API/local UI; fixed and adaptive are explicit experiments. The measured historical runtime is preserved separately in its source archive.

Known resource limitation: repeated model reloads release Metal arrays/cache, while process RSS can retain native/Rust allocator memory and rise across cycles. Fixed-model prefix reuse is separately stable in the measured short branch test. Use a restarted service or separate OS processes for frequent profile switching; do not infer a fully resolved reload-memory issue.
