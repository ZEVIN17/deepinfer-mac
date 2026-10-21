# DeepInfer-Mac continuation

Updated 2026-10-21. Execute [PHASE3_EXECUTION_PLAN.md](PHASE3_EXECUTION_PLAN.md) and [CODEX_EXECUTION_PROMPT.md](CODEX_EXECUTION_PROMPT.md). Preserve existing work; checkpoints record evidence, not approval gates. Branch: `codex/phase2-core`.

## Verified Phase3 section 2

- Config-driven `ModelSpec` / `ModelRegistry` / `ModelAdapter` / `CacheAdapter` / `DraftProvider` now connect Python, CLI and API without scheduler edits. Registration accepts local content identity or exact repo revision; unsupported optimized modes reject before array loading or explicitly report baseline fallback. Public metadata redacts local paths.
- Actual new-family acceptance: SmolLM2-135M-Instruct, local 8bit/group64, 143,030,401 weight bytes. All 9 integration steps passed: shared identity, English/Chinese native-MLX full-token equality, explicit fallback, 3 dense-cache branches, cancel/recovery, exact KV snapshot/restore logits, public metadata, CLI and API/SSE (34 events). This is dense baseline compatibility, not broad model/quality/performance acceptance.
- Existing hybrid generator now uses the guarded cache adapter; portable and tiny actual-Metal regression passes do not replace a new current-source pretrained hybrid optimized-path run. Batch, trained confidence and true per-request variable-length verification remain unverified/disabled for this new model.
- Actual validation: 273 portable tests passed / 41 Metal deselected; Ruff passed. Fresh tiny Metal regression: 41 passed / 273 deselected, 0.89s, system swap delta 0. No large model was loaded by that regression.
- Evidence: [integration report](../benchmarks/reports/phase3-model-adapter-integration.json), [verification](evidence/phase3-adapter-verification.json), [fresh code review](evidence/phase3-adapter-code-review.json), [anti-pattern check](evidence/phase3-adapter-anti-pattern-check.json), [checkpoint](evidence/phase3-adapter-checkpoint.json). Reviews have no current findings. Anti-pattern refresh binds final Metal-appended reports while retaining earlier review hashes.
- Interface guide: [MODEL_ADAPTERS.md](MODEL_ADAPTERS.md); exact local/repo examples: `examples/models/smollm2-135m-8bit.{local,repo}.json`. Public weight audit: `benchmarks/reports/phase3-smollm2-weight-audit.json`; private model audit stays in excluded `artifacts/phase3-model-audit/`.

## Phase3 section 3 experimental checkpoint (strict gate unfinished)

- Opt-in continuous CLI/API now executes one real MLX-LM 0.32.2 `BatchGenerator.next()` tick, stable request/slot generations, bounded queue/active/context/padded-KV reserves, cancellation/exception cleanup and group fairness. Default serial remains; non-greedy/speculative/compare jobs run exclusively. Continuous prefix cache is explicitly cold. Details: [CONTINUOUS_EXECUTION.md](CONTINUOUS_EXECUTION.md).
- Both actual Smol policy processes completed 7 cases / 13 outputs, with true decode/prefill shared ticks, both arrival directions, prefill/decode cancellation and recovery, unchanged sources and swap delta 0. Each has **12/13 full serial-native matches; original strict gate remains false**. The Chinese B4 request diverges at zero-based index 39 (serial token 27830, batch 179). Direct native B4 reproduces all four wrapper outputs; Chinese direct B1 matches all 48 serial tokens. This localizes an observed batch/shape-path boundary, not a proved rounding cause. No speed advantage is established.
- Evidence: [interactive](../benchmarks/reports/phase3-continuous-interactive.json), [upstream policy](../benchmarks/reports/phase3-continuous-upstream.json), [direct native control](../benchmarks/reports/phase3-continuous-upstream-diagnosis.json), [actual API/SSE](../benchmarks/reports/phase3-continuous-api.json), [independent verification](evidence/phase3-tick-verification.json), [code review](evidence/phase3-tick-code-review.json), [anti-pattern check](evidence/phase3-tick-anti-pattern-check.json). Direct B4 uses fixed chunk32/all-at-once insertion, not a faithful arrival/shape replay. Raw tasks/output/events remain excluded in `artifacts/phase3-continuous/`; failed gates and hashes are retained.
- Fresh verification: **296 portable passed / 41 deselected; 41 actual tiny-Metal passed / 296 deselected**; full Ruff passed. Actual ASGI single/two concurrent SSE plus recovery completed four streams, 50 events each; API cancellation/recovery and worker close passed. Runtime/harness freeze is bound in reports; later docs do not relabel source identity. This does not supersede section 2's verified dense baseline compatibility.

## Resource boundary and reproducible entry points

- User cannot currently free memory. Preserve conservative pre-load/context guards; no repeated memory request, no large-model DS/DFlash benchmark, no bypass. Only the 135M model was used for new-family integration; observed Metal peak 418,080,804 B and sampled RSS peak 448,561,152 B are scoped observations, not absolute OS peaks; overall system swap delta 0.
- Actual continuous probes used only the already prepared 143 MB Smol model, started no listening service and did not touch port 8765. Large-model DS/DFlash comparisons remain deferred. Root owns the next GPU baton; do not rerun models merely to refresh documentation.

```sh
# Portable verification; does not load model arrays.
.venv/bin/python -m pytest -q -m 'not metal'
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
# Metadata registration only (local example expects prepared results/phase3-models/ weights).
.venv/bin/python -m deepinfer_mac.cli models --model-config examples/models/smollm2-135m-8bit.local.json --json
# Actual small-model integration: explicit GPU baton / resource check required.
.venv/bin/python benchmarks/experiments/phase3_verify_model_adapters.py
# Optional local service; select a free port and retain memory guards.
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 --model-config examples/models/smollm2-135m-8bit.local.json
# CPU-only continuous probe plan; no arrays or model download.
.venv/bin/python benchmarks/experiments/phase3_verify_continuous.py
# Actual continuous probe requires root's GPU baton; use a fresh, nonexisting output.
.venv/bin/python benchmarks/experiments/phase3_verify_continuous.py --execute --policy interactive --output results/phase3-continuous/fresh-interactive.json
```

## Product and publication

- Product-led bilingual README/site candidate is locally verified: 8 pages / 138 links, desktop/mobile layout, interactive budget illustration. The separate [product acceptance](evidence/phase3-product-acceptance.json) updates only verified Smol dense baseline acceptance. Cross-request budgets / pipeline stay building; no new performance claims or unverified batch acceptance enter the product page.
- Published site remains https://zevin17.github.io/deepinfer-mac/ (`zh.html` Chinese), website-only `main@7b7beb8`, Pages run `39567560655`. This checkpoint does not deploy or change main/tag/releases. Historical a1 release remains immutable.
- Existing a2 wheel/sdist/site candidates: `dist/candidates/v0.1.0a2-cbbe35a/`; these retain their old source identity and are not repackaged as this adapter checkpoint. The adapter packaging check inspected a separate 366-member sdist containing both example configs and no weights; no new release published.

## Next execution

- Section 3 follow-up: preserve the failed exact serial-native gate and investigate the same fixed pre-divergence prefix under B1/B4 (logits/top-two margin, prefill/decode/padding/state position). Do not change tolerance, drop the Chinese case or substitute a new control. Keep continuous opt-in; default serial remains the established path. Also resolve the single-prefill-slot limit: new short prefill currently waits behind an existing long prefill, although existing decode advances each tick. No claim of full section 3 acceptance.
- Proceed with section 4's independently safe confidence contracts, cross-request rank/tie/fresh-remap CPU/reference checks and explicit requested/executed work accounting. Connect a real verifier only after correctness controls; budget changes must change actual verification work. Illustration remains hand-set input, not implemented budget execution or measured performance. ZOS-style overlap follows a verified timeline and dynamic UID/remap/state lifecycle. Preserve section 3's two open boundaries while progressing.
- Phase2 fair two-framework superiority remains unfinished; no stable advantage over native MLX-LM/oMLX established. Phase3 retains the real quality/resource/statistical acceptance target.

## Immutable history and full evidence

- [Complete Phase2 progress + user Phase3 handoff](history/PROGRESS_PHASE2_20261021.md) preserves the previous PROGRESS bytes. Earlier [alpha history](history/PROGRESS_ALPHA_20261021.md), [model choice and complete quality results](PHASE2_MODEL_SELECTION.md), [quality protocol](PHASE2_QUALITY_PROTOCOL.md), [DS source/adaptation mapping](DS_DESIGN_MAPPING.md), [Phase3 source audit](phase3-source-audit.json).
- Historical runtime/locks SHA `868d995c295f9fbf338f0bf81ff184f1cd1263e13ebd7c92eb4087b7b23b2031` belongs to Phase2 core, not this checkout. DS/API frozen archive SHA `f48d430d2433194f4680896e699760a8e0a9602505a320833c02fbbefabe60ab` and all quality archives/reports stay unchanged; use their exact frozen sources for replay.
- Keep raw tasks/tests/generations, private audits, `results/`, weights and `dist/` excluded. Publish only sanitized source/docs/reports/evidence with explicit file lists and normal experimental-branch pushes.
