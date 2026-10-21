# From DeepInfer methods to one Mac

DeepInfer-Mac now executes a reimplementation of the pinned DeepInfer MTPVarlen method: runtime medians, expected accepted yield divided by cost, and separate two-generation acceptance histories for total and per-sequence budget decisions. The original unmodified C++ header is an executable test oracle. Production observations are empirical because the trained Confidence Head is unavailable. This is a MTPVarlen-derived scheduler, not official DSpark.

The MLX execution boundary adds independent draft cost, host scheduling and cache recovery. Its complete-cycle measurements include those costs. On the primary hybrid Qwen3.5-9B candidate, recurrent state is restored from exact snapshots and consumed inputs replayed after rejection or early exit; generic KV trimming would be incorrect. This recovery mechanism is platform adaptation and must not be credited as original DS code.

Real 4B and 9B traces have made decisions that agree with the original C++ scheduler. One 9B prompt also matched all 96 output tokens across baseline/fixed1/fixed2/DS. Compilation states differed, and DS did not beat fixed2 in the raw probe. Correctness correspondence therefore does not establish a speed benefit. Formal heldout, equal-quality measurements against native MLX-LM and a second mature service are still required.

Native MLX-LM LRU reuse and deferred device-array synchronization fix gaps in the first alpha. Their origin and ablations remain separate from DS method claims. Likewise, a better model or quantization can improve answers without making this inference framework faster at equal quality.

Sampling now uses the installed MLX-LM baseline sampler/penalty path, with real seed/native/cache controls. Speculative acceptance remains greedy. Public dataset subsets are fixed before running, measured from frozen sources, and reported with versions/settings/denominators; restricted raw tasks are excluded from release artifacts.

中文：方法落地、答案质量和竞争性能是三类证据。当前已有主模型真实正确性控制，尚无两个常用框架上的公平性能优势。失败与数值差异保留；未完成的联合目标不会因原生持平或单题快照而改写成完成。详见 [DS 对应](DS_DESIGN_MAPPING.md)、[证据](phase2.md) 和 [复现](benchmarking.md)。
