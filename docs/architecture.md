# Architecture and provenance

DeepInfer-Mac is an independent community project: **DeepInfer-inspired inference for Apple Silicon, powered by MLX**. The Python engine runs small quantized Qwen models. It does not implement the DeepSeek V4.1 architecture or official DSpark.

## Request path

CLI / bundled local browser → FastAPI schema validation → bounded FIFO worker → single engine owner → MLX-LM / MLX Metal → token events + result metrics. Only one inference request owns the model at a time; there is no cross-request batching. The public showcase is static and never sends prompts to a public inference service or automatically accesses a visitor's localhost.

| Layer | Origin | Scope |
| --- | --- | --- |
| Array execution, weights, model definitions, tokenizer loading | MLX 0.32.4 / MLX-LM 0.32.2 reuse | Actual Apple Silicon Metal execution |
| Ordinary / fixed speculative generation | MLX-LM reuse | Same tokenizer required; greedy only in this release |
| Adaptive verification loop | MLX-LM loop adaptation + independent policy | Runtime cost and acceptance observations; experimental benefit |
| Exact prompt snapshots and byte-budget LRU | Independent implementation | Target/draft state identity, cloning and eviction |
| Worker, API subset, CLI, local demo, benchmark | Independent implementation | Local single-user workflow |
| DeepInfer variable verification and runtime profiling | Design inspiration / reference only | No confidence head, batch scheduler, DSpark or CUDA port |

## Version-sensitive boundary

`upstream.lock.json` records verified upstream commits. MLX-LM `utils.load(..., revision=..., trust_remote_code=False)` and `generate.stream_generate(..., draft_model=...)` are real APIs at the lock. `speculative_generate_step` has a fixed `num_draft_tokens`; it has no dynamic-length callback. The adaptive loop is isolated in `speculative.py` and preserves target/draft cache rollback, rejection position, bonus token and generator-finalization semantics. It does not rebuild a generator for every output token.

Model identity, immutable revision, quantization, chat template, thinking setting and target/draft relationship participate in cache isolation. Cached arrays are cloned before mutable inference. Prefix reuse only applies to exact token prefixes; a request retains a final token to compute logits. Both target and draft caches advance together. Budgeted snapshots can be evicted; active KV is estimated separately and oversized inputs are rejected, never silently truncated. Rotating/sliding/quantized/approximate KV and disk persistence are outside this release.

Metrics distinguish model-load time, request total, TTFT, measured prefill, decode and request-scoped Metal allocator peak. RSS measures the process separately and must not be added to Metal figures on unified memory. Native MLX-LM prompt throughput can include initial speculative work and is not relabeled as isolated prefill. Benchmark cache states and allocator/model residency are recorded explicitly.

## DeepInfer source audit

The reference `tests/reference/vendor/mtp_varlen.hpp` exactly matches upstream `csrc/engine/mtp_varlen.hpp` (git blob `09af680aa42b724fe86315007a160d5012a1fde9`) at [commit 774ce24](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01). It uses measured throughput curves, per-position acceptance confidence, sliding median runtime estimation and a verification budget. The required upstream `sequence.hpp` exists but is absent here; no compiled reference equivalence claim is made.

The [locked deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md) describes Confidence Head guided variable verification and asynchronous CPU/device scheduling at cluster scale. Our single-request policy uses observable costs and accepted tokens instead of a trained head. DeepInfer's distributed KV store and bounded SWA replay rely on different architecture and semantics; they are not transplanted into this engine.

## Qwen3.8-27B candidate

The priority showcase candidate is a hybrid Gated DeltaNet/full-attention model through MLX-LM qwen3_5. Its recurrent ArraysCache cannot use generic KV trim rollback; the Qwen3 draft vocabulary is incompatible. Treat text-only baseline and exact cache snapshots as a separate validation track, not an adaptive-speculative demonstration. Multimodal loading/sanitization, 24 GB peak memory, swap growth and multi-turn stability require actual evidence before making it the default. No 27B throughput is inferred from smaller Qwen3 experiments.

## Runtime limits

Apple Silicon + Metal is required for real generation. Pure Python policy, statistics, schema and worker checks can run on non-Metal CI; those checks cannot certify model correctness or performance. Only locked compatible model profiles are offered. Sampling is greedy, text-only, n=1; tools, JSON grammars, embeddings and complete OpenAI protocol compatibility are unsupported. See [compatibility](compatibility.md) and [benchmark reproduction](benchmarking.md).

## 中文架构说明

请求从 CLI 或本地浏览器进入 FastAPI，在严格 schema 校验后排入有界 FIFO 队列，由唯一的推理所有者调用 MLX-LM/MLX；输出包含增量文本事件和可导出的测量结果。静态展示站只展示安装、架构和审核后的基准，不会自动访问访客 localhost，也没有公共提示词接收服务。

模型加载、量化格式、普通解码和固定草稿生成复用 MLX-LM。自适应循环是隔离的 MIT 来源适配，策略依据实际草稿/验证成本和按位置接受率选择预算，并可回退到零草稿。缓存管理、CLI、API 和基准导出由本项目实现。与上游 DSpark 的共同点是“根据收益选择验证工作量”的思想；本项目没有上游专用 Confidence Head、批调度和异步集群实现。

精确缓存存储 prompt 最后一个 token 之前的状态，缓存身份包含权重 revision、量化、模板及 thinking；复用前克隆，目标/草稿状态协同维护。512 MiB 默认驻留预算通过 LRU 淘汰控制快照，活跃 KV 另作保守估算和请求拒绝。不静默裁剪输入，不提供近似 SWA replay。RSS 与 Metal allocator 分别报告，统一内存下不可相加。适配层升级需重新审查回卷、bonus token、EOS 和取消清理。

来源锁固定 DeepInfer/MLX/MLX-LM commit。C++ 参考头与上游 blob 完全一致，但缺少依赖的 sequence.hpp，不是可执行参考测试。完整 V4.1 CED/CSA2/MTP、分布式 KV 与 CUDA/Ascend 内核均未移植。

## Resource-release limitation

Repeated profile unloading was checked separately from steady-model cache reuse. `close()` synchronizes, clears snapshots, drops model references, runs garbage collection and clears Metal allocator cache. In four reload cycles, Metal active allocation returned from about2.6 GB to8 bytes and cached bytes to zero. However process RSS after close rose approximately317 →493 →641 →778 MB across cycles. Native/Rust allocator retention is still under investigation; this release does not claim flat process memory across frequent reloads.

The fixed-model eight-branch test is a separate result: near-stable active/RSS and zero observed per-request swap delta. For frequent profile changes or long reload-heavy sessions, restart the local service or use separate OS processes. The isolated-residency ablation uses an OS process per block so process exit guarantees release; failures from earlier in-process attempts remain in the evidence.

中文：反复卸载确实把 Metal active 从约2.6 GB降至8字节、cache降至0，但close后RSS在多轮中仍从约317增长到778 MB，native/Rust allocator驻留尚待定位。不能把固定模型缓存的稳定性外推为频繁reload无增长；频繁切换模型建议独立进程或重启服务。
