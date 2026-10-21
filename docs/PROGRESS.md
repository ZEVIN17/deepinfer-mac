# DeepInfer-Mac — current continuation

Updated2026-10-21. Read `PHASE2_EXECUTION_PLAN.md`, `PHASE2_MODEL_SELECTION.md`, `DS_DESIGN_MAPPING.md`. Preserve the uncommitted work. User authorized continuous implementation and precise removal of unused model weights; checkpoints report evidence, not approval gates.

## Current acceptance status

The joint goal remains **DS-derived executable methods + fair superiority over two common frameworks**. It is not complete: no competitive superiority has been established. Native parity or model-selection quality gains do not satisfy that goal. Alpha v0.1.0a1 is historical and its published package bytes must remain immutable.

## Model choice and cleanup

- Primary candidate9B8bit: `mlx-community/Qwen3.5-9B-8bit@16daa4818c54ce5f5436f929d52542eb65bbed9d`; both shards and all1260tensor headers SHA verified. Total weights10,426,592,423B; text9,514,412,544B; vision912,020,960B; no MTP keys.
- Same-model4bit control `8b2b98c00a6b4d291155e4890773ca8f769aee53`;14B4bit fallback `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`. 9B4bit both shards/full1260headers are SHA verified (text5,038,041,600B,vision912,020,960B), not loaded yet.14B4bit two shards/full1007headers also SHA verified (8,307,898,514B); all downloads complete, 9B4bit first actual baseline load succeeded and public comparison is running;14B not loaded yet. Actual quality/performance comparison remains pending.
- Explicitly retired27B profiles no longer appear in active API/UI. Six exact old27B HF-cache/chunk directories removed:19,177,137,641 logical B (17.86GiB), allocated19,361,185,792B. Original reports/audit metadata and4B/0.6B retained. All unrelated models untouched.
- Completed9B8bit range chunks (10,426,592,423B) removed after SHA-verified cache assembly. Cleanup/PID/path evidence: `artifacts/phase2-model-audit/model-cleanup-20261021.json`.
- DS adaptation draft0.8B4bit `da28692b5f139cb0ec58a356b437486b7dac7462` downloaded, single625,229,487B shard SHA verified. Complete vocabulary/special IDs match9B. Template files differ in default thinking branch; explicitTrue/False render equal target tokens. No public speculative mode until validation. Evidence: `artifacts/phase2-model-audit/hybrid-draft/`.

## Actual9B8bit evidence

- Fixed12task dev/heldout smoke ×4 settings completed48cases. Greedy engine/native10/12, full12/12token equality; official non-thinking11/12; thinking9/12 at1024 minimum cap (3unfinished). All failures retained: greedy ball arithmetic wrong, non-thinking Chinese HBM claim wrong, greedy capacity overclaim. Not a full benchmark or statistical equivalence.
- `benchmarks/reports/quality-qwen3.5-9b-8bit.json`: raw prompts, IDs, settings, seed42, outputs, manual rubrics, resources. Initial adaptation run overlaps source review edits; source_start/end differ. Formal subsequent runs must use frozen source.
- First load2.43s; pre-load-reset Metal peak9,514,412,552B; observed system swap+387,317,760B. This is not a no-swap acceptance claim. Request peaks/RSS/swap reported separately.
- Real API nonstream/SSE/cache token equality, unsupported-mode422, stream-disconnect cancellation and recovery passed: `benchmarks/reports/phase2-service-qwen3.5-9b-8bit.json`.
- Actual Safari UI fixed clamp repair completed, matches direct engine IDs; TTFT939.62ms,total1834.24ms,decode27.94tok/s,request swap0. One interactive request, not a benchmark. `benchmarks/reports/phase2-ui-qwen35-9b8.json`; screenshot `docs/evidence/phase2-local-ui-qwen35-9b8.png`. Temporary browser tab and API process21136 closed.

## DS and verification

- Real MTPVarlen-derived scheduler with empirical acceptance implemented; original C++ oracle + independent invariant/replay tests. Production4B probe96tokens/42cycles made31core decisions: `benchmarks/reports/phase2-ds-engine-smoke-qwen3-4b.json`. No claim of27B/9B acceleration from this.
- Review fixed native full-prompt replay on cache hit, failed speculative-cycle policy drift, and duplicated snapshot bytes in explicit memory budgets. Hybrid snapshot/restore/replay for9B passed baseline/fixed1/fixed2/DS96token exact equality on one real prompt; DS made24core decisions. These are correctness controls with unequal warmup, not speed evidence.
- Latest root portable check115passed/41Metal skipped; root independently ran `.venv/bin/python -m pytest -q --run-metal -m metal`:41passed,137deselected,1.33s. This predates subsequent opt-in message-checkpoint additions. Ruff passed. Site candidate8pages/152links and isolateda2package verification independently passed; alpha artifacts unchanged.

## Active work / next commands

- At this checkpoint GPU owner=`phase2_quality`,9B4 public comparison execsession56865;public8bit18133 completed54/54 and all downloads complete (14B83761,9B4 96080 exit0). Redundant completed range chunks removed. All three compatibility servers closed; no API server remains. Confirm process/owner before any GPU run. Ownership is handed back explicitly, not inferred from stale documentation.
- Baseline/native sampling support added (temperature/top_p/top_k/presence/seed/thinking), strict greedy-only speculative gates retained; same-seed/cache/native15/15checks passed, plus1Metal full-history penalty. CLI and actualAPI/SSE official sampling return8 on the ball task; cancel/recovery passed. Public8bit formal run is complete,8K resource probe completed (8300+256,peak10.827GB,swap+2.794GB); formal8K service group remains deferred.9B4bit public comparison is running,14B follows and fair nativeMLX/oMLX benchmarks. Public raw tasks/reports stay excluded in `results/quality-cache`; publish sanitized summary only.
- Formal quality sourcev2: `benchmarks/reports/phase2-quality-source-v2-20261021/source.tar.gz`, SHA `1b928104acee51f742b8be4893b2804cfac59779aa79904395dad13f351a09d4`;47files inclruntime/locks/runner/scorers/licenses, extracted `/tmp/deepinfer-phase2-quality-source-v2-20261021`. Public data is copied locally outside the archive. Checkouta2/version/cache experiments do not alter this frozen runtime.
- Public27greedy results complete: IFEval6/8,MMLU-Pro11/14,LCBpublic-tests2/4,auxiliaryrepair1/1. All3MMLUunparseable failures reached1024cap; bothLCBfailures reached2048cap. They are finite-budget completion failures, not demonstrated intrinsic inability. Official production sampling complete: IFEval7/8,MMLU9/14,LCB3/4,repair1/1;5MMLU+1LCB failures hit the same caps. Sampling helps some instructions/code but completes fewer MMLU tasks at this budget. No final model selection/quantization-equivalence claim yet. Sanitized `benchmarks/reports/public-quality-qwen35-9b8-v2-summary.json`. Raw `results/quality-cache/quality-public-qwen35-9b8-v2.json`, log `/tmp/deepinfer-public-quality-qwen35-9b8-v2.log`.
- Public corpus27: IFEval8, MMLU-Pro14, LCBv6 public-tests smoke4, auxiliaryrepair1. Sources/licenses/fixed IDs: `benchmarks/quality/README.md` +manifest. This is not leaderboard reproduction. `PHASE2_QUALITY_PROTOCOL.md` defines the local smoke and sampling.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q --run-metal -m metal
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
# Formal public run: use the frozenv2 working directory/PYTHONPATH and repositoryvenv.
#9B4 public comparison56865is active: inspect it before starting a duplicate.
PYTHONPATH=src .venv/bin/python benchmarks/experiments/context_resource_probe.py --input-tokens 8300 --max-tokens 256 --experimental-context-limit 9216
.venv/bin/python benchmarks/quality/runner.py summary --responses results/quality-cache/quality-public-qwen35-9b8-v2.json --output benchmarks/reports/public-quality-qwen35-9b8-v2-summary.json
PYTHONPATH=src .venv/bin/python -m deepinfer_mac.cli serve --port 8876
PYTHONPATH=src .venv/bin/python benchmarks/experiments/phase2_service_probe.py
```

The quality command above must run from a frozen copy for formal evidence; use `scripts/freeze_quality_source.py`, setPYTHONPATH to its `src`, and materialize public tasks locally outside the source archive. `scripts/audit_model_snapshot.py` checks local shard SHA/tensor identity. Existing4B regression and0.6B weights remain ready.

## Historical references

- Original alpha/publication log preserved verbatim in `history/PROGRESS_ALPHA_20261021.md`; prior phase audit=`PHASE1_AUDIT.md`.
- Historical alpha release: https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1 ; site https://zevin17.github.io/deepinfer-mac/ . Neither is currentPhase2 completion evidence.
