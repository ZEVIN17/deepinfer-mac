# DeepInfer-Mac continuation

Updated 2026-10-21. Execute `PHASE2_EXECUTION_PLAN.md` and `PHASE2_MODEL_SELECTION.md`; preserve existing work. Checkpoints show evidence, not approval gates. The user authorized continuous implementation and precise removal of unused weights.

## Current outcome

- Branch `codex/phase2-core`, checkpoint `6918844` pushed with user authorization. Alpha v0.1.0a1 bytes remain historical and immutable; this checkpoint is not completion of Phase2.
- Required joint goal: executable DS-derived methods **and** fair superiority over two common frameworks. No competitive superiority is established. Model/quantization gains are not framework gains.
- Pushed fixed index:164portable tests,41Metal tests, actual20message-checkpoint requests/16checks all token-identical under greedy and recommended sampling. Root ran `.venv/bin/python -m pytest -q --run-metal -m metal`:41passed,164deselected,1.45s from `/tmp/deepinfer-phase2-final-index-check`; runtime/locks SHA `a2469e78b2e105559f2c682374483de56779d823c5715e6626d20e4c0db29a5f`.
- New cost-adapter/formal phase currently194portable tests; independent provenance/coverage review found4+4P2 being repaired by owners. New DS formal GPU/competitive measurements are pending. Earlier41Metal/20requests do not validate these later changes.

## Models and cleanup

- Main capability candidate: `mlx-community/Qwen3.5-9B-8bit@16daa4818c54ce5f5436f929d52542eb65bbed9d`;10,426,592,423B total weights,9,514,412,544B text. Same-model4bit `8b2b98c00a6b4d291155e4890773ca8f769aee53`;5,950,221,072B total,5,038,041,600B text.14B4bit fallback `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`;8,307,898,514B total.
- All three candidates are downloaded, complete shard SHA/header verified, actually loaded and public27×2settings measured. All remain `quality_review_required`, public modes baseline only.9B4bit is not an established capability-equivalent replacement.14B has task tradeoffs; final selection awaits official-protocol/smoke/budget controls.
- Compatible0.8B4bit hybrid draft `da28692b5f139cb0ec58a356b437486b7dac7462`,625,229,487B, actually loaded. Complete vocab/special IDs match9B; explicit thinkingTrue/False template tokens match despite different default-branch template files. No actual MTP heads in9B/draft shards.
- Removed exactly six abandoned27B cache/chunk paths:19,177,137,641logical B (17.86GiB), allocated19,361,185,792B. Completed new-download range duplicates also removed after SHA verification. Retained metadata/reports,4B/0.6B regressions, required comparison/draft weights; unrelated models untouched. All initial candidate transfers finished. DFlash trained-draft transfer is separately owned by `dflash_audit` (last reportedPID24258), not registry-ready; confirm I/O stopped before formal performance.
- Private cleanup evidence: `artifacts/phase2-model-audit/model-cleanup-20261021.json`; release-safe weight/template/sampling audit: `benchmarks/reports/phase2-model-weight-audit.json`. Both model locks reflect actual loading separately from quality/performance acceptance.

## Quality evidence

| Fixed27 subset |9B8greedy|9B4greedy|14Bgreedy|9B8sampled|9B4sampled|14Bshared sampled|
|---|---:|---:|---:|---:|---:|---:|
|IFEval|6/8|5/8|7/8|7/8|7/8|8/8|
|MMLU-Pro|11/14|8/14|10/14|9/14|8/14|9/14|
|LCB public tests|2/4|1/4|2/4|3/4|1/4|2/4|
|Auxiliary repair|1/1|1/1|1/1|1/1|1/1|1/1|

- Public sources/IDs/licenses: `benchmarks/quality/README.md`, manifest. This is not full leaderboard reproduction. Small per-benchmark Wilson intervals are wide; nonsignificance does not establish equivalence. Raw licensed tasks/generations remain excluded in `results/quality-cache/`.
- Complete162responses, paired per-task differences/resources/completion/Wilson: `benchmarks/reports/public-quality-paired-v2.json` and per-model summaries. Sampling here uses common Qwen3.5 preset, **not**14B's official default; separate Qwen3 controls are pending. No combined capability score or framework speed claim.
-9B8 MMLU failures all hit1024cap; LCB failures hit2048cap.14B additionally has completed wrong answers/code. These finite-budget outcomes do not demonstrate intrinsic inability.9B4 completed MMLU2586/LCBabc387_b failures reproduce exactly in native MLX and originalV2 tokens: `phase2-qwen35-9b4-native-failure-control.json`.
- Initial9B8 local12×4smoke: greedy/native10/12 with12/12full-token equality, recommended non-thinking11/12, thinking9/12 with3unfinished. Chinese factual failures retained. Chinese rubric annotations are Codex per-response reviews, not independent human labels. See `quality-qwen3.5-9b-8bit.json` and `PHASE2_QUALITY_PROTOCOL.md`.
- V2 frozen archive SHA `1b928104acee51f742b8be4893b2804cfac59779aa79904395dad13f351a09d4`.14B first attempt loaded but diagnostic cache-type logging failed before requests, not a model failure. V2.1 SHA `d97b48ea15455319a68ecde1b1d3183a565413299894a775d67c0e3bb9d5d62f` changes only cross-model cache audit and error-status recording. Explicit `harness-bridge.json` verifies exact patch/all other file hashes; original hashes are retained.

## Runtime, resources, and next work

- GPU owner=`phase2_quality`,9B4 local12×4smoke execsession33037 from frozenV3.1; log `/tmp/deepinfer-quality-qwen35-9b4-smoke-v3-1.log`.14B smoke25682 completed48/48 and closed, native12/12exact. Root DS/API/resource processes all exited. Model processes are serial. All API/competitor services from earlier controls are closed. Confirm process and explicit baton before another GPU job.
-8300input+256output already completed with explicit9216override: requestMetal10.827GB, systemswap+2,794,127,360B. Report `context-8300-256-qwen35-9b8.json`; formal8K service group remains deferred. Do **not** repeat this probe. 4204+256 under8192cap is also measured: stream identical, requestMetal10,651,956,858B; load/request/overall swap all0. Source6918844/a2469e...; `context-4204-256-qwen35-9b8.json`. This is a single-request boundary, not concurrency/stress validation. Do not repeat either probe.
- Actual9B8 API/SSE/cache/cancel/recovery, CLI recommended sampling, SafariUI clamp demonstration are linked in `benchmarks/reports/phase2-service-sampling-qwen35-9b8.json`, `phase2-cli-sampling-qwen35-9b8.json`, `phase2-ui-qwen35-9b8.json`; UI screenshot=`docs/evidence/phase2-local-ui-qwen35-9b8.png`. These are transport/demo controls, not fair speed benchmarks.
- Baseline/native accept sampling/seed/thinking; fixed/adaptive strictly reject non-default sampling.15real seed/cache/native controls passed. Experimental hybrid DS initial96-token controls were exact for baseline/fixed1/fixed2/DS;24actual core decisions. New frozenf48d probe128baseline/lagged IDs match,58cycles/28mature decisions; independent C++28/28. Original acceptedflagfalse came from tuple/list verifier input and is preserved; CPUminimalfix+supplement link `phase2-hybrid-lagged-dev-qwen35-9b8-supplemental-verification.json`, no GPUrerun. Experimental API2/2strictJSON and baseline/DSfull tokens/SSE match, sampling422: `phase2-experimental-api-qwen35-9b8.json`. Formal performance remains pending.
- Queue: finish9B4smoke;14B official native27; preregistered dev1471/dev9423/abc387_c4096-token control on all three models/two common settings; DS formal and nativeMLX/oMLX comparison. Budget controls supplement original scores, never overwrite them. Coordinate GPU gaps with root.

```sh
# Inspect current job; do not start a duplicate GPU model.
pgrep -fl phase2_model_quality
# Both resource probes already complete; inspect their reports, do not reload for them.
.venv/bin/python -m json.tool benchmarks/reports/context-4204-256-qwen35-9b8.json
.venv/bin/python benchmarks/quality/runner.py summary --responses results/quality-cache/quality-public-qwen3-14b-v2-1.json --output benchmarks/reports/public-quality-qwen3-14b-v2-1-summary.json
.venv/bin/python benchmarks/quality/compare_models.py --reports results/quality-cache/quality-public-qwen35-9b8-v2.json results/quality-cache/quality-public-qwen35-9b4-v2.json results/quality-cache/quality-public-qwen3-14b-v2-1.json --harness-bridge benchmarks/reports/phase2-quality-source-v2-1-20261021/harness-bridge.json --output benchmarks/reports/public-quality-paired-v2.json
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
PYTHONPATH=src .venv/bin/python -m deepinfer_mac.cli serve --port 8876
```

Frozen quality runtimes remain independent of active checkout/core edits; preserve their source manifests and local raw outputs. Release-safe summaries and source archives may be published; never bundle public raw task text/tests/generations or the entire private audit directory.

## History

`history/PROGRESS_ALPHA_20261021.md` preserves the original long alpha log. Read `PHASE1_AUDIT.md`, `DS_DESIGN_MAPPING.md`, and the current plans for scope. Historical alpha release/site are not Phase2 completion evidence.
