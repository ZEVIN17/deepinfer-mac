# Compatibility matrix / 兼容矩阵

| Surface | Status and boundary |
| --- | --- |
| Apple M4 Pro / 24 GiB | Actual development host; per-report resource measurements |
| Other Apple Silicon | Requires a complete hardware report; not verified by these measurements |
| Intel macOS / Linux / Windows | Portable logic/diagnostics only, no Metal generation |
| Python | 3.12/3.13 dependency range; local hardware checks on 3.12 |
| MLX / MLX-LM | Locked 0.32.4 / 0.32.2; upgrades require sampler/state re-audit |
| Qwen3.5-9B-8bit | Primary candidate, real baseline/sampling/API/UI checked; final quality/quantization choice pending |
| Qwen3.5-9B hybrid speculation | Explicit experimental Python/probe entry with locked 0.8B4bit draft; one 96-token real four-mode equality control; public API/UI baseline only |
| Qwen3.5-9B-4bit / Qwen3-14B-4bit | Quality/efficiency controls in progress; do not infer results from 9B8bit |
| Qwen3-4B + Qwen3-0.6B draft | Regression pair; fixed/legacy/DS greedy paths; tokenizer identity checks |
| Qwen3-0.6B | Small standalone baseline; self-draft is a correctness control |
| Retired Qwen3.8-27B | Removed from active profiles; obsolete weights/chunks cleaned; historical failures preserved |
| Baseline sampling | temperature 0–2, top_p (0,1], top_k, presence_penalty −2–2, optional seed; explicit enable_thinking |
| Speculative sampling | Greedy only: temperature=0, top_p=1, top_k=0, presence_penalty=0; unsupported settings fail validation |
| API | GET /v1/models; POST /v1/chat/completions text n=1, optional SSE; local run/events/cancel |
| API unsupported | Tools, images, response_format, logprobs, custom stop, n>1 and unknown fields fail validation |
| Prefix cache | Native LRU for trimmable state; hybrid exact snapshots/suffix replay; cloning, byte budgets and eviction |
| Public showcase | Static evidence and downloads; no automatic localhost or public inference |
| Full V4.1 / official DSpark | Unsupported; no trained confidence head or continuous batching |

The full 9B sampler/native/cache controls passed 15 engineering checks. Same seed is tested at the locked runtime, not promised portable across hardware/library versions. Public model-quality subsets and fair engine performance remain separate acceptance work.

中文：当前 9B 可在 API/UI 普通或采样生成，实验混合推测不能从公开接口绕过 profile 限制。无报告的模型/硬件不标成已验证；greedy token 一致不等于所有题目正确。详见 [第二阶段证据](phase2.md) 和 [27B 历史](qwen27.md)。
