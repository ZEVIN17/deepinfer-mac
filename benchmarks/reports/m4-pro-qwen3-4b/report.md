# DeepInfer-Mac measured benchmark

Profile: `qwen3-4b`; suite: `full`; repeats: 3.

Warm loaded model measurements; all warmups retained. No cold model speed comparison. Native MLX-LM stream_generate uses the same model/template. Upstream prompt_tps includes first-token work, not pure prefill.

| Variant | Runs | Median request ms | Failures |
| --- | ---: | ---: | ---: |
| baseline | 36 | 793.96 | 0 |
| fixed_speculative:1 | 36 | 701.43 | 0 |
| fixed_speculative:2 | 36 | 631.08 | 0 |
| fixed_speculative:4 | 36 | 808.07 | 0 |
| adaptive | 36 | 797.44 | 0 |
| native_mlx | 36 | 810.84 | 0 |

Full token equality: 27 / 36 measured comparison groups.

| Reference / adaptive request time (work may differ) | Pairs | Median ratio | Mean ratio 95% bootstrap interval |
| --- | ---: | ---: | --- |
| baseline | 36 | 1.000 | 0.972–1.001 |
| fixed_speculative:1 | 36 | 0.884 | 0.893–1.016 |
| fixed_speculative:2 | 36 | 0.836 | 0.849–0.950 |
| fixed_speculative:4 | 36 | 1.063 | 1.046–1.225 |
| native_mlx | 36 | 1.021 | 0.999–1.026 |

Ratios above 1 indicate lower adaptive elapsed time, but different token sequences mean different work. Every failure and mismatch remains in summary.json and results.csv. No equivalent speedup claim is supported unless all runs succeed and all full token sequences match. Small corpus results do not establish general speed advantages or p95 latency. RSS and Metal allocation are separate metrics.

See summary.json for full configuration, hardware, versions, model metadata, every token sequence, failures and fallbacks; rounds.jsonl contains policy observations.
