# Compatibility matrix / 兼容矩阵

| Surface | Status and boundary |
| --- | --- |
| Apple M4 Pro / 24 GB / macOS | Initial real verification host; evidence reports identify actual OS and power state |
| Other Apple Silicon Macs | Expected MLX platform; unverified until a complete community report is supplied |
| Intel macOS / Linux / Windows | Portable diagnostics, policy, statistics, schemas and mocked service tests only; no Metal generation |
| Python | 3.12 and 3.13 dependency range; primary local verification on 3.12 |
| MLX / MLX-LM | 0.32.4 / 0.32.2 locked; upgrades require loop/cache re-audit |
| Qwen3.8-27B MLX 4-bit | Priority showcase candidate, unverified on this host; text baseline/exact-cache only until measured validation. Hybrid recurrent cache and vocabulary prevent old Qwen3 draft reuse; not default |
| qwen3-4b target + Qwen3-0.6B draft | Locked 4-bit pair; tokenizer compatibility checked before draft load |
| qwen3-0.6b | Small standalone baseline; self-draft is a correctness control, not a claimed acceleration |
| Sampling | Baseline is the safe default; speculation is explicitly selected. Greedy only (temperature=0); full sampled speculative correctness is future work |
| API | GET /v1/models; POST /v1/chat/completions text-only subset; local compare/run events |
| API supported input | model, messages(system/user/assistant text), max_tokens, temperature=0, stream, mode, n=1; stop=null |
| API unsupported | Custom stop, tools, images, response_format, logprobs, n>1 and unknown fields fail validation |
| Prefix cache | Exact token prefixes, cloning, byte budget and eviction; no disk persistence or approximate replay |
| Public showcase | Static; does not execute models or automatically connect to localhost |
| DeepSeek V4.1 / official DSpark | Not implemented or validated by this release |

中文：没有报告的硬件标为未验证；无 Metal CI 只覆盖控制逻辑。服务只实现明确列出的文本 greedy 子集，不能宣称完整 OpenAI 协议兼容。模型许可与代码许可独立。

See [27B resource and lower-bit evidence](qwen27.md) for the pinned 4-bit audit and untested 3/2-bit community alternatives. Estimates are budget gates, not measured runtime peaks.
