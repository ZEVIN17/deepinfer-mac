# Architecture and provenance / 架构与归属

DeepInfer-Mac is an independent Apple Silicon inference project powered by MLX. The working 0.1.0a2 candidate implements a DeepInfer-derived scheduler; it does not implement full DeepSeek V4.1 or official DSpark.

## Request path

CLI / bundled local browser → strict FastAPI schema → bounded FIFO worker → one inference owner → MLX-LM / Metal → text events and metrics. Requests execute serially; FIFO is not continuous batching. The public showcase is static and never automatically accesses a visitor's localhost.

| Layer | Origin | Scope |
| --- | --- | --- |
| Model/array execution and loading | MLX 0.32.4 / MLX-LM 0.32.2 | Real Metal execution; pinned model revisions |
| Baseline sampling | MLX-LM sampler and logits-processor reuse | Temperature, top-p, top-k, presence penalty, seed; native/cache consistency controls |
| Trimmable speculative loop | MLX-LM adaptation | Greedy target/draft rollback, rejection, bonus, EOS/close |
| MTPVarlen scheduler and estimators | Reimplementation of pinned DeepInfer methods | Empirical acceptance replaces unavailable trained confidence; original C++ oracle |
| Hybrid recovery | Mac platform adaptation | Exact ArraysCache/KVCache snapshot/restore and consumed-prefix replay |
| Prompt-cache lookup | Native MLX-LM LRU reuse; exact backend retained for controls | Trimmable common prefix; hybrid exact snapshots only |
| API, CLI, local UI and measurement | Independent implementation | Local text n=1, bounded queue and cancellation |

[DS_DESIGN_MAPPING.md](DS_DESIGN_MAPPING.md) records exact source functions, revisions, invariants and executable evidence. The immutable vendor header is compiled against a small host shim for reference tests; the production scheduler is Python. It preserves sliding median estimators, expected-yield/cost maximization and distinct N−2/N−1 rate histories. Actual 4B and 9B decisions have been replayed against original C++; this establishes method correspondence, not a speed advantage.

## State and timing boundaries

Cache identity includes model/revision/quantization/template/thinking and draft identity. Snapshots are cloned before mutable execution and subject to a byte budget. No input is silently truncated. The native backend reuses MLX-LM LRU rather than presenting that existing optimization as a DS invention. Hybrid recurrent state cannot be generically trimmed: the experimental adapter restores exact state, then replays the inputs actually consumed. Last unconsumed output, partial rejection, EOS/close and cancellation have separate controls.

DS choices happen before verification using completed history. Complete-cycle wall time includes scheduling, snapshot, draft, verification, host conversion/output and required replay. Device-array combined drafting/verification is not split into invented GPU component timings. Diagnostic traces and formal performance runs are separate.

Baseline sampled generation supports the installed MLX-LM sampler/penalty contract. Fixed/adaptive speculative modes reject non-greedy settings. Public 9B API/UI remains baseline-only; experimental hybrid entry is `InferenceEngine(adaptive_scheduler="ds", experimental_hybrid_speculation=True)` with the locked 0.8B draft. The CLI DS example in the README uses the validated trimmable 4B profile.

## Resources and limits

Primary candidate 9B8bit has real load/API/UI evidence; final 8bit/4bit/14B choice remains under evaluation. Retired 27B weights have been removed and reports retained. Generation is text-only; tools, grammars, multimodal input, embeddings, official Confidence Head, cross-request GPU batching and distributed KV are unsupported.

Model load, TTFT, request total, decode speed, request-scoped Metal peak, RSS and system swap deltas are separate measurements. RSS and Metal allocation overlap on unified memory and must not be added. The 9B initial load observed swap growth; a later request with zero delta does not establish universally swap-free operation.

Historical 4B unload controls returned Metal active allocation from about 2.6 GB to 8 bytes, but process RSS rose about 317 → 493 → 641 → 778 MB across reloads. Native/Rust allocator retention remains unresolved. Use separate OS processes for reload-heavy experiments. Fixed-model cache stability is a distinct limited result.

中文：核心 DS 方法已对应到真实执行与原始 C++ 参考；经验接受率、独立草稿成本和混合状态恢复均明确标注适配差异。单请求服务、原生缓存复用与质量改善不能替代两个框架上的同质量性能验收。详见 [Phase 2](phase2.md)、[兼容矩阵](compatibility.md) 与 [基准方法](benchmarking.md)。
