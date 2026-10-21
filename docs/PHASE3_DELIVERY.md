# Phase 3 engineering delivery

Updated 2026-10-21. This records the implemented experimental framework and its
acceptance boundaries against [the execution plan](PHASE3_EXECUTION_PLAN.md).
It does **not** mark the plan's competitive-performance or broad optimized-quality
gates complete. [PROGRESS](PROGRESS.md) is the current checkout/deployment entry;
this document links stable implementation and measurement evidence.

## Delivered architecture and evidence

| Plan area | Delivered | Evidence and remaining boundary |
|---|---|---|
| Model integration | Config-driven model/cache/draft contracts, exact identity, resource guards, explicit capability fallback; consistent Python/CLI/API identity | [Adapter acceptance](evidence/phase3-adapter-checkpoint.json): new SmolLM2 dense baseline, 9 actual integration steps. Supported interface is broader than optimized validation. |
| Request execution | Real MLX-LM batch ticks; bounded queue, active/context/KV budgets, stable slot generations, cancellation/recovery, priority aging | [Tick verification](evidence/phase3-tick-verification.json). Existing decode interleaves long prefill; new prefill cannot preempt its occupied slot. Strict serial equality is 12/13 in each policy, not full acceptance. |
| Joint adaptive work | Conditional/cumulative confidence contract, censored observations, two-generation history, measured cost keys, actual MLX rank allocation and per-length verification buckets | [Reference mapping](PHASE3_BUDGET_REFERENCE.md), [joint verification](evidence/phase3-joint-verification.json). Empirical confidence, no trained head; hybrid verification is isolated single-row. |
| Experimental service | Distinct target/draft loading, bounded concurrent ownership, verified-token SSE, cancellation and recovery, aggregate-only public metadata | [35/35 actual service checks](evidence/phase3-joint-service-verification.json). Explicit opt-in and greedy dense pairs; no exact-key calibrated curve means full budget. |
| Profile-guided execution | Default-off wall-stage tracing; opt-in batched draft forward and dense KV trim; rollback, identity and cache-class guards | [Profile verification](evidence/phase3-profile-verification.json), [dense tests](evidence/phase3-dense-fastpath-verification-first.json): 338 portable and 76 actual-Metal tests at the frozen checkpoint. Dense fast paths are not promoted to default service capabilities. |
| Reproducibility and distribution | Frozen protocols, source/model hashes, redacted reports, independent reviews; isolated wheel/sdist/site build and installation checks | [Package evidence](evidence/phase3-dense-checkpoint-package.json) binds candidate `v0.1.0a2-d3c6ae1` to commit `d3c6ae1`. Later diagnostics and homepage updates are separate from this package. |

The architecture improvement over the project's previous whole-request serial
path is concrete: requests have independently managed lifecycles, admission is
bounded, and cross-request budgets reach actual model shapes. MLX-LM owns the
underlying model operators and native batch/cache machinery; these are reused,
not claimed as new algorithms. [DS design mapping](DS_DESIGN_MAPPING.md) separates
DeepInfer-derived rules from Mac integration and experimental optimizations.

## What the measurements establish

The distinct SmolLM2 360M/135M pair demonstrated **30.3% fewer verification input
positions and 9.0% fewer total decode input positions** in a heterogeneous bucket
control. Complete latency did not improve in that run. This is real work reduction,
not a general speedup. [Protocol and report](PAIR_COST_EXPERIMENT.md).

The frozen dense diagnostic used three independent processes, rotated control
order and two previously reserved prompts. All **18/18** completed request outputs
matched native tokens at equal budgets, with zero swap growth. Median paired-task
time was **279.13 ms** for fast trim, **367.56 ms** for old replay and **180.52 ms**
for native MLX-LM batch. The internal path improved in this narrow observation;
**native batching remained faster**. No superiority over native MLX-LM or oMLX
has been established. [Full summary](../benchmarks/reports/phase3-dense-heldout-summary.json).

Strict numerical acceptance remains **false**. Whole-prefix logit checks passed
0/24; a cycle-partition-matched oracle passed 12/12 old-replay checks and 0/12
fast-trim checks, despite matching next-token argmax and cache offsets. The original
tolerance is unchanged. Matching these short outputs is not proof of general
quality or equivalent cache state. [Source-bound verification](evidence/phase3-dense-heldout-verification.json).

## Run and reproduce

With installed dependencies and the audited local weights available:

```sh
# Default baseline service, using a free port.
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 \
  --model-config examples/models/smollm2-135m-8bit.local.json
# Explicit experimental joint service; stop the previous example before reusing its port.
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 \
  --model-config examples/models/research-smollm2-pair.local.json \
  --experimental-joint-verification --joint-profile research-smollm2-pair \
  --joint-depth 3 --max-active 2 --batch-context-tokens 4096
# Portable verification.
.venv/bin/python -m pytest -q -m 'not metal'
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
# Device verification requires exclusive GPU ownership and successful resource admission.
.venv/bin/python -m pytest -q -m metal --run-metal
```

The existing user service on port 8765 is outside these examples. Installation,
model registration and service behavior are documented in the bilingual READMEs
and [joint service guide](JOINT_VERIFICATION.md). No weights ship in the package.
[DENSE_FASTPATH_EXPERIMENT.md](DENSE_FASTPATH_EXPERIMENT.md) contains exact diagnostic
and CPU-only export commands. Raw prompts/outputs and measured source snapshots
remain private under `artifacts/`; public hashes alone do not supply that workload.
Do not overwrite historical reports or relabel them with newer source identities.

## Gates that remain open

- Resolve fast-trim numerical differences before promoting optimized equivalence;
  preserve dense-only guards and separate tiny controls from pretrained evidence.
- Complete broader quality, long-context, overload and cross-model/hybrid optimized
  acceptance. Public API automatic cost calibration and trained confidence are not delivered.
- ZOS cross-round overlap is not implemented. Current profile supports reducing
  draft/replay work; it does not establish sufficient host wait to justify ZOS.
- Competitive performance acceptance remains open. Current fixed-budget/ignored-EOS diagnostics
  cannot fairly compare oMLX, whose audited HTTP path cannot disable EOS. A future
  comparison needs a new natural-EOS protocol; exposed heldout data cannot be reused
  for tuning. Large 9B comparisons remain resource-limited, with guards retained.
- The repository and homepage may present the implemented architecture and verified
  service achievements. Package validation, branch push, site deployment and a tagged
  release are distinct events; consult PROGRESS for their current verified identities.
