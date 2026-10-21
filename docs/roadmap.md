# Evidence-driven roadmap / 路线图

0.1 alpha focuses on real greedy inference, transparent speculative comparisons, exact reusable prefixes and a reliable single-user local API.

Next improvements are selected from observed profiles: amortize adaptive controller overhead, improve conservative zero-draft fallback, extend session prefix reuse, then gather multi-hardware reports. Add true non-greedy speculative acceptance/residual sampling only with distribution tests. Introduce Metal kernels only after a measured hotspot and reference numerical tests show a practical end-to-end gain.

DeepSeek V4.1 CED/CSA2/MTP and confidence-head work requires suitable weights, architecture support and larger hardware; it remains a separate research track. Tool protocols, batching, persistent KV and wider model support require distinct correctness contracts.

中文：先依据实测优化策略开销和精确缓存，再扩展硬件矩阵。采样、Metal 算子、工具调用、并发和持久化需各自的正确性证据；完整 V4.1 不在首版支持范围。
