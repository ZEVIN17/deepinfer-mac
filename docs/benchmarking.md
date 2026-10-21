# Reproduce the evidence / 复现基准

No universal speedup is promised. Public claims must point to an actual report and the complete machine-readable data. Model download or server startup is not a performance result.

```sh
uv sync --frozen --extra mlx --extra dev
uv run deepinfer-mac doctor --json
uv run deepinfer-mac models --download qwen3-4b
uv run deepinfer-mac bench --profile qwen3-4b --suite quick --repeats 3 --fixed-lengths 1 2 4 --max-tokens 64 --output benchmark-results
```

Use `--suite full` for the larger fixed corpus. Run one GPU experiment at a time; close competing inference processes. Save the command, output files and failures. Warm the model explicitly; a cleared prompt cache does not imply unloaded weights or cold OS pages. Reports separate cache reuse from throughput comparisons.

The comparator is same-version native MLX-LM and project ordinary decoding, several fixed speculative lengths, then adaptive. Rotate mode order, repeat the same inputs and compare complete token sequences including EOS. Greedy target results may differ when larger verification matrix shapes alter floating point tie decisions. Publish mismatches and diagnose them; do not silently drop them. A speed claim against fixed speculative must use the best reasonable measured fixed length, not a deliberately slow control.

Report model load, TTFT, isolated project prefill where available, decode throughput, total request time, completed tokens, scoped allocator peak, RSS, accepted draft tokens and fallback. Native MLX-LM prompt TPS is not pure prefill. State when short runs or limited repeats prevent credible tail-latency or significance claims. Unified memory RSS and allocator numbers overlap; do not add them.

`models.lock.json`, `upstream.lock.json` and `uv.lock` define the reproduction environment. Model identity includes chat template, tokenizer, quantization and thinking settings. The release's machine-specific evidence is indexed in [evidence.md](evidence.md). The website build copies only reviewed public data from `site/data/benchmark.json`; missing evidence is explicitly displayed, never filled with mock performance.

中文：先完成真实下载，再执行上述单组 GPU 命令。保留失败、回退及完整 token 差异；缓存冷态、权重驻留和系统页缓存分别说明。TTFT、prefill、decode、总时长与内存口径不能混用。样本不足不宣传 p95 或统计显著。只发布匿名标准语料，私人提示词默认不进入公开证据。

## M4 Pro release evidence (2026-10-21)

`benchmarks/reports/m4-pro-qwen3-4b/summary.json` and `report.md` contain the complete fixed full corpus: 12 prompts, 3 measured repeats plus retained warmups, 64-token cap, greedy, thinking disabled, locked Qwen3-4B/0.6B, fixed lengths 1/2/4, rotating order and empty KV before every run. There are 216 measured runs (288 including warmups), zero failed runs, and 27/36 comparison groups where all six variants have identical complete token sequences. Ordinary baseline matches native MLX-LM on all 36 measured groups; speculative shape differences remain visible.

| Variant | All measured runs | Median request ms |
| --- | ---: | ---: |
| Native MLX-LM | 36 | 810.84 |
| Baseline | 36 | 793.96 |
| Fixed draft 1 | 36 | 701.43 |
| Fixed draft 2 | 36 | 631.08 |
| Fixed draft 4 | 36 | 808.07 |
| Adaptive | 36 | 797.44 |

These are descriptive elapsed times for every successful run, including divergent outputs. The complete predetermined corpus does not pass full speculative equivalence, and adaptive does not demonstrate a gain over tuned fixed decoding. **No equivalent speedup or adaptive superiority is claimed.** Matching-subset ratios remain diagnostics, never a filtered headline. The next useful optimization direction is verified exact-prefix reuse and controller overhead, not a fabricated multiplier.

The runtime manifest digest is `d6de01beb2dbe1fe4434036326d6c17d7872668b5090b25be3d5d00409a43c1e`. The same report directory preserves `source.tar.gz` and `source-manifest.json`: every hashed source file was checked byte-for-byte and no private home paths occur. This snapshot makes the original measured runtime available even if later release defaults change. `git_commit` was null when measurements started; a later commit must not be presented as existing at run time.

中文：完整全语料216次正式运行、零失败，只有27/36组所有模式完整token一致，普通baseline对原生MLX-LM为36/36一致。自适应没有优于调优后的固定预算；不宣布等价加速。所有耗时和分叉保留，等价子集仅用于诊断。测量时尚无Git提交，归档精确源码与锁文件供复现。

## Exact-prefix cache evidence

`benchmarks/reports/cache-qwen3-4b.json` measures warm-model baseline generation with a1740-token repeated prompt and only2 output tokens including EOS. Its three cold/hot prompt-cache request pairs are2096.99/37.24 ms,2132.57/38.33 ms and2174.20/36.72 ms. Each hot request reuses1739 tokens; complete output tokens and text match the cold result. Model load checks are about1.6–2.4 ms after a separate warmup. This is a repeated-prefill benefit, not a general long-output or speculative speedup claim.

Eight related branch rounds reuse512 tokens and preserve complete output equality. Across those rounds the active Metal allocation changes by6 KiB and RSS by about0.26 MiB; every request's observed swap delta is zero. A1 KiB cache budget rejects the oversized snapshot and recomputation remains equivalent. RSS and Metal statistics are not added.

Reproduce this specific workload serially, with no other GPU experiment:

```sh
uv run python tests/metal_cache_benchmark.py --profile qwen3-4b --output results/cache-qwen3-4b.json
```

The cache evidence records the same frozen runtime source digest as the formal comparison archive. Release defaults may subsequently switch to baseline; archived source preserves the measured version.

Real Metal control tests are opt-in: `uv run pytest -q --run-metal -m metal`. Normal pytest skips them. Those tiny deterministic transitions verify cache rollback/control flow; they are not measured model throughput. Real engine/service/cache scripts run actual weights and publish their outputs separately.

## Isolated-process residency control

`benchmarks/reports/resident-process-ablation-qwen3-4b.json` records18 separate-process blocks,54 measured runs plus54 warmups, no failures and zero measured request swap deltas. Median request times are native567.19, baseline558.57, fixed1 439.48, fixed2 462.56, fixed4 550.78 and adaptive558.73 ms. All six variants match in6/9 paired prompt/repeat groups; baseline and adaptive each match native9/9, while each fixed variant diverges in3/9. These elapsed times do not establish equivalent speculative speedup or adaptive superiority. Two source manifests differ only in CLI/schema/UI default selection; inference core and measurement-script hashes match, and each block retains its actual manifest. The earlier same-process attempt and24MemoryErrors remain published separately.
