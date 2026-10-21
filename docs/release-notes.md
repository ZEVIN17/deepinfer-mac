# 0.1.0a1 — experimental alpha

Independent DeepInfer-inspired Apple Silicon inference powered by pinned MLX / MLX-LM. Adds greedy baseline/fixed/adaptive comparisons, exact prefix snapshots with an LRU byte budget, a bounded local API worker, bundled chat/compare UI, CLI diagnostics and real result exports.

This is an early, single-user release. Performance depends on draft/target compatibility and workload. Adaptive decoding is experimental and can be slower than fixed or ordinary decoding. Numerical shape differences can change greedy argmax; consult complete equivalence data before interpreting throughput. No sampling distribution guarantee, official DSpark, full V4.1, distributed KV store or custom Metal kernel is claimed.

Release preparation includes source/wheel builds, a static bilingual GitHub Pages showcase, dependency/model/upstream locks, portable tests and separately recorded Metal evidence. Successful checks and real reports are recorded in `docs/PROGRESS.md`; pending checks must not be represented as completed.

中文：首版提供真实 greedy 引擎、三模式比较、精确前缀缓存、有界 API、本地 UI、锁文件和基准导出。自适应不保证更快，数值差异需完整披露。验收证据以 PROGRESS 索引为准。

Measured release result: full Qwen3-4B corpus, 216 measured runs and zero failures; all-variant token equality27/36 groups, native/ordinary baseline equality36/36. Adaptive did not improve on the tuned fixed control. No equivalent speculative speedup is claimed. Raw outputs, mismatch diagnostics and a hash-verified snapshot of the measured source are preserved in `benchmarks/reports/m4-pro-qwen3-4b/`.

The release defaults to ordinary baseline decoding across CLI/API/local UI; fixed and adaptive are explicit experiments. The measured historical runtime is preserved separately in its source archive.

Known resource limitation: repeated model reloads release Metal arrays/cache, while process RSS can retain native/Rust allocator memory and rise across cycles. Fixed-model prefix reuse is separately stable in the measured short branch test. Use a restarted service or separate OS processes for frequent profile switching; do not infer a fully resolved reload-memory issue.
