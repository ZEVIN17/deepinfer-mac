# Measuring speculative decoding on one Mac

A draft model can reduce target-model decoding work when accepted tokens amortize drafting and verification. A longer draft is not automatically better: rejected suffixes still cost memory traffic and verification time. A useful local engine therefore needs a trustworthy ordinary baseline, several fixed budgets and complete output comparison before it needs a marketing multiplier.

DeepInfer-Mac borrows the idea of scheduling useful verification work from DeepInfer while executing through MLX. Its first alpha adapts the greedy MLX-LM loop behind a narrow boundary. An independent policy observes draft time, verification time and positional acceptance, explores candidate budgets and can select no draft. It has no trained Confidence Head, no official DSpark implementation, and no full DeepSeek V4.1 port.

The other practical direction is exact context reuse. A repeated system prompt or shared conversation prefix need not be prefilled again when the model identity and token prefix match. The engine stores isolated prompt snapshots, clones them before mutation and evicts within a byte budget. It rejects oversized active contexts instead of claiming efficiency through hidden truncation or swap. Persistent distributed KV and approximate SWA replay solve different problems and are outside this release.

The benchmark keeps native MLX-LM in the comparison, rotates run order, warms models, clears KV between throughput runs and retains every failure. Complete token IDs including EOS are compared. Matrix-shape changes can alter floating point argmax, so elapsed-time ratios with divergent output are labeled as comparisons of different work. Small-corpus bootstrap intervals are descriptive, not evidence of universal significance; RSS and Metal peaks have separate meanings on unified memory.

The release publishes the actual machine-specific data in the evidence index. Adaptive decoding may lose to tuned fixed decoding or ordinary generation. That result is useful: it shifts optimization toward measured controller overhead, verified prefix reuse, and the workloads where drafting has a chance to amortize. A custom Metal kernel should follow a measured hotspot and a numerical reference, rather than precede them.

To reproduce, follow [the benchmark tutorial](benchmarking.md). To contribute, provide a full hardware report with locked revisions, settings and every output. This note is a technical explanation; it asserts no unmeasured speed advantage.

## 中文摘要

推测解码是否获益取决于草稿成本、验证成本和接受位置，不能只看接受率或草稿长度。首版用原生 MLX-LM、普通解码、多个固定预算和自适应策略做完整比较，保留所有失败和 token 差异。缓存路线采用精确前缀、身份隔离、克隆及按字节淘汰，不偷换成近似截断。没有自适应收益时，继续依据瓶颈优化策略开销和前缀复用；没有 profiling 与数值证据时不添加 Metal 算子。完整结果以实际证据文件为准。
