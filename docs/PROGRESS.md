# DeepInfer-Mac continuation

Updated 2026-10-21. Execute `PHASE2_EXECUTION_PLAN.md`; preserve existing work. User authorized continuous implementation, precise unused-weight cleanup and branch publication. Checkpoints show evidence, not approval gates.

## Current outcome and source

- Branch `codex/phase2-core`, pushed core checkpoint `66ae444` remote-confirmed.6918844 is the earlier validation source. Historical alpha v0.1.0a1 bytes remain immutable. Phase2 is **not complete**: no fair competitive superiority over two common frameworks established; model-selection gains are not framework gains.
- Root independently confirmed257portable checks/Ruff and latest actual41Metal passed,257deselected,1.05s. Fresh quality review recomputed207public/dev scores,18full old-token prefixes and144smoke counts;36targetedCPU checks passed. Provenance/coverage4+4P2 and the swap/chronology wording issues are resolved. Runtime/locks34-file SHA `868d995c295f9fbf338f0bf81ff184f1cd1263e13ebd7c92eb4087b7b23b2031`; metadata is stable for service freezes. See `docs/evidence/phase2-latest-metal-verification.json` and `phase2-model-selection-{verification,code-review,anti-pattern-check}.json`.
- Earlier pushed fixed index164CPU+41Metal and actual20message-checkpoint requests/16checks passed. Metal command `.venv/bin/python -m pytest -q --run-metal -m metal`:41passed,164deselected,1.45s from `/tmp/deepinfer-phase2-final-index-check`; runtime/locks `a2469e78b2e105559f2c682374483de56779d823c5715e6626d20e4c0db29a5f`.
- DS/API frozen archive `f48d430d2433194f4680896e699760a8e0a9602505a320833c02fbbefabe60ab`, actualruntime/locks `15c19f7f83d42fb43a3271be44f056664ee99d94f544c7f3ddbe9c45e890f72c`; quality uses separate older frozen archives. Never relabel their hashes as current checkout.

## Model choice and cleanup

- Retain9B8bit main text baseline/fair-framework target;9B4bit quality/efficiency control;14B fallback/DScontrol. Full decision with all favorable/unfavorable results: `PHASE2_MODEL_SELECTION.md`. No overall model ranking or quantization-equivalence claim. All3baseline runtimes actually loaded/native-verified; quality remains limited with known failures, public modes baseline.
- Pins:9B8=`16daa4818c54ce5f5436f929d52542eb65bbed9d`;9B4=`8b2b98c00a6b4d291155e4890773ca8f769aee53`;14B4=`a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`. Full shardSHA/header verified. Total/text B:9B8=10,426,592,423/9,514,412,544;9B4=5,950,221,072/5,038,041,600;14B=8,307,898,514/8,307,783,680. Text-only; no validated image support.
- 0.8B4bit draft `da28692b5f139cb0ec58a356b437486b7dac7462`,625,229,487B, loaded. Full vocab/special IDs and explicitthinking template IDs match9B despite default-branch file differences. No MTP heads.14B/0.6B tokenizer/template CPU-compatible; DS execution pending.
- Six exact abandoned27B cache/chunk paths removed:19,177,137,641logical B (17.86GiB), allocated19,361,185,792B. Completed new-download chunks removed afterSHA. Reports/metadata/4B/0.6B/necessary comparison weights retained; unrelated/Ollama models untouched. Initial candidates and necessaryDFlash download complete.
- Public weight/template/sampler audit=`benchmarks/reports/phase2-model-weight-audit.json`; private cleanup evidence=`artifacts/phase2-model-audit/model-cleanup-20261021.json`. Two model locks separate verified baseline loading, limited quality, experimentalDS and unvalidated competitive performance. models.py/locks edits are **STABLE** for next freeze.

## Quality evidence complete

| Public27 subset |9B8greedy|9B4greedy|14Bgreedy|9B8sampled|9B4sampled|14Bshared sampled|
|---|---:|---:|---:|---:|---:|---:|
|IFEval|6/8|5/8|7/8|7/8|7/8|8/8|
|MMLU-Pro strict|11/14|8/14|10/14|9/14|8/14|9/14|
|LCB public tests|2/4|1/4|2/4|3/4|1/4|2/4|
|Auxiliary repair|1/1|1/1|1/1|1/1|1/1|1/1|

-162shared responses plus27independent14Bofficial-native responses complete; official14B=8/8,strict8/14,2/4,1/1. Shared sampler usesQwen3.5presence1.5;14Bofficial usespresence0. Uniform offlineMarkdown-only sensitivity across189responses(98MMLU;91othersunmodified) changes14Bgreedy10→11/shared9→11/official8→11;9B8/4unchanged. Original strict scores immutable. `public-quality-paired-v2.json`, per-model summaries and `public-quality-mmlu-format-sensitivity.json` retain IDs/Wilson/conditions; not full leaderboards orLCBfullpass@1.
- Local12×4smoke complete/all rubric annotations done:greedy/native/officialnonthink/think=9B8 10/10/11/9;9B4 11/11/9/7;14B 10/10/10/10, each/12. Native12/12full-token equality each. Thinking unfinished3/4/0 at1024minimum. Chinese reviews are Codex, not independent human labels; KV prompt lacks explicitLLM scope.14Bdedupe algorithm tests pass but extra prose violatesonly-function. Full failures retained.
- Preregistered3dev1471/9423/abc387_c×3models×2settings at4096 complete18/18. Every new output contains its entire old same-setting token prefix. Strictgreedy/sample passes:9B8=3/3,2/3;9B4=0/3,2/3;14B=0/3,0/3(14Bgreedy has1Markdown-correct answer in separate sensitivity). Original27scores unchanged. `public-quality-dev4096-budget-sensitivity.json` and `public-quality-dev4096-mmlu-format-sensitivity.json` retain all costs/limits.
- Examplecosts are mixed:9B8 LCBC greedy3832tokens/144.65s,shared1196/47.52s;9B4shared3391/75.19s.9423sampled9B4=23.79s versus9B8=138.14s. These diagnose model/budget behavior, not framework speed. 18/18请求无正系统swap增量，负变化照实保留；系统级观测不能独归本进程。
- QualityV2 archiveSHA `1b928104acee51f742b8be4893b2804cfac59779aa79904395dad13f351a09d4`.14Bfirst attempt diagnostic failed before requests after successfulload; V2.1SHA `d97b48ea15455319a68ecde1b1d3183a565413299894a775d67c0e3bb9d5d62f` changes onlycache audit/errorstatus. Explicit testedharness-bridge verifies exactpatch/all other files. V3.1 dev source remains frozen. Public raw tasks/generations stay excluded in `results/quality-cache/`.

## GPU and remaining acceptance

- GPU owner=`root`/assigned`dflash_audit`; all qualitysessions3348/4743/36251/64668 and services are closed. No automatic new GPUjob. The user currently cannot free memory; large-model198-callDS and furtherDFlash GPU work remain deferred until resource conditions change and root grants an explicit baton. Continue CPU work; do not repeat the memory request. LatestDFlashbalanced preflight507 rejected beforeload; oneaggressive retry3actual4204+256 requests passed correctness/activeDFlash butcoldload→end swap+3,254,255,616B, preregisteredresourceadmission=false. Services stopped; root owns next DSablation/officialmemoryguard decisions, thenDFlashdevgrid.
- Standalone9B8 resourceprobes complete, do not repeat:8300+256 explicit9216override peak10.827GB/swap+2,794,127,360B, formal8Kdeferred.4204+256 under8192cap peak10,651,956,858B/streammatches/load-request-overall swap0 at6918844/a2469e...; single-request boundary, not concurrency/stress orDFlashresourceproof. Reports=`context-8300-256-qwen35-9b8.json`, `context-4204-256-qwen35-9b8.json`.
- DS frozenprobe128baseline/lagged IDs match,58cycles/28mature decisions, independentC++28/28. Originalacceptedfalse tuple/listverifier mismatch staysunchanged; CPUfix+supplement/noGPUrerun=`phase2-hybrid-lagged-dev-qwen35-9b8-supplemental-verification.json`. No speed claim.
- Actual experimentalAPI2/2strictJSON,baseline/DSfulltokens+SSE equal,sourceunchanged,metadataexperimental/nongreedy422=`phase2-experimental-api-qwen35-9b8.json`. Earlier production sampling/CLI/SafariUI/cache/cancel and15seed/native controls remain in service/CLI/UIreports. DFlash3strictcompatibility/51draftcalls45verifycycles passed withswap+517,996,544B; instrumentation only, not performance.
- Architecture showcase is independently approved locally:8pages/138links,desktop/mobile QA,interactive stage tabs and88 actual recorded SSE content events. Its self-contained9-file evidence bundle exactly matches core`66ae444`; historical5site/data files and backend remain unchanged. Run `.venv/bin/python site/build.py` then `.venv/bin/python site/check.py`; see `docs/evidence/phase2-showcase-independent-{verification,code-review,anti-pattern-check}.json`. New public deployment is not yet claimed.
- Next: publish the separately reviewed site-only payload after the experimental branch checkpoint. Root owns198callDSablation, reasonable-best oMLX/DFlash dev selection/newfreeze/heldout and two-framework superiority. No claimed advantage until actual fair distributions/quality/resource evidence pass.

```sh
# No quality model job should remain; coordinate explicitGPUbaton with root.
pgrep -fl phase2_model_quality
.venv/bin/python -m pytest -q tests/test_quality_compare.py tests/test_model_quality_protocol.py tests/test_mmlu_format_sensitivity.py tests/test_dev_budget_sensitivity.py tests/test_engine_core.py
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
.venv/bin/python benchmarks/quality/dev_budget_sensitivity.py --originals results/quality-cache/quality-public-qwen35-9b8-v2.json results/quality-cache/quality-public-qwen35-9b4-v2.json results/quality-cache/quality-public-qwen3-14b-v2-1.json --extended results/quality-cache/quality-dev4096-qwen35-9b8-v3-1.json results/quality-cache/quality-dev4096-qwen35-9b4-v3-1.json results/quality-cache/quality-dev4096-qwen3-14b-v3-1.json --output benchmarks/reports/public-quality-dev4096-budget-sensitivity.json
PYTHONPATH=src .venv/bin/python -m deepinfer_mac.cli serve --port 8876
```

Exactfrozenreplays/source/licenses=`benchmarks/quality/README.md`; interpretation=`PHASE2_MODEL_SELECTION.md` and `PHASE2_QUALITY_PROTOCOL.md`. Publish sanitized summaries/sourcearchives, never rawlicensed tasks/tests/generations or entireprivateauditdirectory. Oldlog=`history/PROGRESS_ALPHA_20261021.md`; prioraudit/design=`PHASE1_AUDIT.md`,`DS_DESIGN_MAPPING.md`.
