# DeepInfer-Mac continuation

Updated 2026-10-21. Continue [Phase 4 plan](PHASE4_EXECUTION_PLAN.md) on `codex/phase2-core`. Root owns GPU baton; preserve user service port 8765 and memory guards. Checkpoints are evidence, not approval gates.

## Phase 4 active checkpoint

- User direction: adopt native batching as the main executor. CLI `serve` now defaults to native-first `auto` with fixed upstream prefill; unsupported cache protocols and sampled requests use serial execution. Joint speculation remains explicit.
- Implemented allocated draft generation, shape/context-specific bounded full-cycle observations, and transactional committed-prefix handoff to native batching. A quality-gated/hysteretic decision primitive exists; speculative automatic selection is not connected to default serving.
- Fresh cache diagnosis: merge/extract and same-shape trim/slice agree exactly in the tested controls. Q=1 versus Q=4 forward partitions explain strict differences in four fresh cases. This does not promote fast-trim or rewrite Phase 3 evidence.
- Three-process development medians: native **124.09 ms**, full draft **760.56 ms**, allocated draft **529.84 ms**. Internal reduction **30.3%**, but native remains faster. Outputs match across all arms; no competitive/heldout/oMLX claim.
- Native arrival development exposed adaptive chunk collapse. Added a minimum chunk; the floor-64 candidate still lost to tuned fixed native. Keep native fixed policy as default. Do not rerun failed candidates until a new mechanism justifies it.
- Validation: **364 portable + 95 Metal tests passed**, Python lint passed, **46/46 actual service checks passed**, no service swap growth. Static site build/check: 8 pages / 138 links. [Service evidence](evidence/phase4-service-verification.json).
- [Runtime report](PHASE4_NATIVE_RUNTIME.md), [source map](PHASE4_SOURCE_MAP.md), [upstream patch candidate](UPSTREAM_CONTRIBUTIONS.md). No push, public PR, deployment or release. Prior [Phase 3 continuation](history/PROGRESS_PHASE3_FINAL_20261021.md) remains intact.
- Next: native admission/cache budgets and fresh fair competitive traces; complete-cost calibration, automatic speculative admission, cross-family optimized quality and final product performance gates remain incomplete. Upstream public prose requires human authorship under its contribution policy.

## Current result

- Implemented model/cache/draft interfaces, CLI/API identity, actual continuous ticks, joint confidence/budget allocation and real variable-length verification buckets. Experimental pair service includes bounded admission, stable slots, cancellation/recovery, API/SSE; actual service acceptance **35/35**. [Service evidence](evidence/phase3-joint-service-verification.json).
- Dense opt-ins implement batched draft forward and direct KV trim; defaults remain serial draft/exact replay. Only audited dense KVCache is eligible; hybrid/rotating/quantized cache classes reject. Fresh checkpoint tests: **338 portable + 76 actual-Metal passed**. No API capability promotion.
- Audited real pair: SmolLM2-360M-Instruct-6bit target + 135M-Instruct-8bit draft. V2 development ran depths 1,2,3 once each, then froze depth 3/batched/dense_trim/bucket before heldout. **60/60** internal and **6/6** native-batch outputs matched; **120/120** next-token/offset checks passed, **0/120** strict whole-prefix logits checks passed. [Frozen selection](evidence/phase3-dense-fastpath-development-v2.json).
- Exactly three independent heldout diagnostic processes rotated native batch/old replay/fast trim. **18/18** request outputs matched native target and exact budgets; swap growth zero, peak Metal **955,583,292 bytes**. Whole-prefix allclose passes **0/24**; cycle-partition replay oracle passes **12/24** (old replay 12/12, fast trim 0/12). Both oracle groups pass every next-token argmax and offset check. Strict numeric acceptance remains **false**, tolerance unchanged. [Verification](evidence/phase3-dense-heldout-verification.json), [three-process report](../benchmarks/reports/phase3-dense-heldout-summary.json).
- Median paired-task times across those processes: native batch **180.52 ms**, old replay **367.56 ms**, fast trim **279.13 ms**. Internal work improved in this narrow diagnostic; native MLX-LM batching remains faster. No established advantage over native MLX-LM/oMLX, general quality equivalence or release-ready claim.
- Earlier heterogeneous buckets saved **30.3% verification positions / 9.0% total decode positions**, without measured latency gain; historical failures stay intact. [Earlier cost evidence](PAIR_COST_EXPERIMENT.md).

## Reproduce and artifacts

```sh
.venv/bin/python -m pytest -q -m 'not metal'
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
# Explicit root GPU baton required for device commands:
.venv/bin/python -m pytest -q -m metal --run-metal
# Each round 0,1,2 is a separate process; exact private workload required.
.venv/bin/python benchmarks/experiments/phase3_dense_heldout.py --execute --round 0 --output artifacts/phase3-dense-fastpath/fresh-round0.json
# Existing experimental service entry; do not use port 8765:
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 --model-config examples/models/research-smollm2-pair.local.json --experimental-joint-verification --joint-profile research-smollm2-pair --joint-depth 3 --max-active 2 --batch-context-tokens 4096
```

- [Dense protocol](DENSE_FASTPATH_EXPERIMENT.md) and `phase3_dense_heldout_protocol.json` freeze source, order, input/output budgets and comparison scope. Raw prompts/outputs and all measured source snapshots remain excluded under `artifacts/phase3-dense-fastpath/`. Public hashes alone are not a downloadable prompt corpus. Export scripts reproduce redaction without GPU.
- First dense runtime checkpoint **d3c6ae1320f09f0d21813931c6df488cf4b4909b** is pushed to the experimental branch. Candidate `dist/candidates/v0.1.0a2-d3c6ae1/` contains that checkpoint; **v2 development and heldout diagnostics postdate it**. [Package evidence](evidence/phase3-dense-checkpoint-package.json). No new release/tag or runtime merge to main.
- Research evidence is pushed at `d684205`; product/delivery documentation at `b1a1af0`. Public bilingual site is now website-only main **`1a5b68b25a23fc2b4375dc0370a64cfbd7e21544`**. [Pages run 39586997096](https://github.com/ZEVIN17/deepinfer-mac/actions/runs/39586997096) passed; live EN/ZH homepages and JS/CSS equal the reviewed build byte-for-byte. [Deployment evidence](evidence/phase3-homepage-deployment.json). [Engineering delivery](PHASE3_DELIVERY.md) distinguishes completed implementation from research gates.

## Remaining gates and next execution

- Separate quantized shape/partition numerical differences from cache-state errors; cycle-matched old replay now passes, fast trim still differs. Keep strict failures and dense-only opt-in scope; do not promote capability based on matching sampled tokens alone.
- API cost calibration remains unaccepted; no automatic online calibration or trained confidence head. Cross-model/hybrid optimized equivalence, trace-justified ZOS overlap, broad quality/long-context/overload acceptance and competitive performance gates remain incomplete.
- This ignored-EOS fixed-budget protocol is internal/MLX-only. oMLX needs a newly frozen natural-EOS fair protocol; do not reuse heldout for tuning. Large 9B fairness remains resource-limited; user cannot free RAM, do not ask again or bypass guards.
- Independent code/export/claim review, explicit public branch synchronization and homepage deployment are complete. Continue the unresolved research gates above without changing historical evidence. Exclude raw artifacts, results, weights, dist and private paths from pushes.

## History

[Previous dense-first continuation](history/PROGRESS_PHASE3_DENSE_FIRST_20261021.md) preserves its exact bytes, including links to all earlier history. Historical measurements and hashes remain unchanged.
