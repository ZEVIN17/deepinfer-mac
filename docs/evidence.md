# Public evidence index

Release 0.1.0a1 is being verified on Apple M4 Pro / 24 GB. The complete Qwen3-4B controlled benchmark is available. No equivalent speedup is asserted: speculative sequences diverge on part of the corpus and adaptive does not beat the tuned fixed control. Real Qwen3-0.6B service/inference evidence is now available.

- `upstream.lock.json`: verified commits and exact DeepInfer reference header blob.
- `uv.lock`: reproducible dependency resolution; MLX 0.32.4 / MLX-LM 0.32.2.
- `docs/evidence/showcase-desktop.png`, `showcase-mobile.png`: 1280px desktop and 390px responsive production showcase checks.
- `python3 site/check.py`: complete local links/resources and repository-subpath verification.
- `python3 scripts/verify_artifacts.py`: wheel resources, license notices and source artifact checks.

Portable tests exercise policy/cache/statistics/API control flow. They do not prove real Metal throughput. Hardware and numerical results are tracked separately. The public static site contains actual reviewed data only; it never synthesizes performance values.

- `benchmarks/reports/local-ui-demo.json`: exact public Chinese prompt and actual exported baseline result; model loading included, single interactive request, not a throughput benchmark. `docs/evidence/local-inference.png` records its full UI.
- Mobile showcase check uses a true 390×844 iframe browsing viewport within a 1280px IAB host: media queries execute at 390px. It does not emulate a physical phone, touch or DPR.
- `docs/evidence/qwen27-preflight.json`: conservative resource budget gate blocked before model load; estimates are not runtime peak, OOM or speed measurements. Candidate remains unverified and is not the default.

- [Qwen3.8-27B audit and lower-bit candidates](qwen27.md): pinned header/source evidence, actual 2K/4K budget preflight snapshots, and exact community candidate revisions. No low-bit candidate was downloaded or loaded.

- `benchmarks/reports/m4-pro-qwen3-4b/`: complete 12-prompt × 3-repeat report, every warmup/failure/mismatch, raw CSV/JSONL and exact source snapshot. 216 measured runs, zero failures, all-six-variant token equality27/36. Native/baseline equality36/36; adaptive has no demonstrated advantage over tuned fixed. See the benchmark tutorial for boundaries.

- `docs/evidence/benchmark-desktop.png` and `benchmark-mobile.png`: actual final benchmark page with complete-corpus caveat and raw-data links. Final mobile check uses a browser viewport override of390×844, then resets it; no physical phone/touch/DPR claim.

- `benchmarks/reports/cache-qwen3-4b.json`: warm-model1740-token repeated prompt,2-token output including EOS; three exact cold/hot pairs,1739-token hits, output equality. Eight512-prefix branch checks plus tiny-budget recomputation also match. Only repeated-prefix cost reduction is established; no long-output/speculative performance inference.

- `benchmarks/reports/local-ui-demo-qwen3-4b.json` + `docs/evidence/local-inference-qwen3-4b.png`: actual4B baseline UI response;1138.84ms includes401.03ms load, a single demo not a benchmark. Main showcase uses4B;0.6B evidence remains. `local-ui-27b-candidate.png` verifies disabled speculation/comparison without running27B.
- `benchmarks/reports/resident-process-ablation-qwen3-4b.json`:18 independent OS-process blocks,54 measured+54warmup runs, zero failures/observed swap growth.6/9 all-variant token equality; baseline/adaptive each9/9 match native; fixed variants each3/9 diverge. Core/script hashes stable; source manifests differ only in CLI/API/UI baseline defaults. Historical same-process report retains24MemoryErrors; do not erase or infer GPU leakage from them.
