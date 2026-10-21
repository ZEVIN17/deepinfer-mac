# Dense fast-path validation protocol

This is a prospective, bounded mechanism protocol for the audited local
SmolLM2-360M-Instruct-6bit target and SmolLM2-135M-Instruct-8bit draft. It is not an
API throughput benchmark or a comparison with oMLX. No device execution is
permitted without the root GPU baton. No downloads, memory-guard exceptions or
changes to the user service on port 8765 are part of this protocol.

The private workload `artifacts/phase3-dense-fastpath/workloads-v1.json` is frozen
by `benchmarks/experiments/phase3_dense_workload_manifest.json` before fast-path
execution. It contains two development and two new held-out prompts, unequal
input lengths and output budgets of 17 and 32 tokens. The old pair-cost held-out
prompts are already exposed and may only be labelled regression workloads.
Prompt hashes alone do not make this a publicly reproducible prompt corpus.

First validate all four combinations of serial/batched draft and exact replay/
dense trim. Check strict native-target tokens, exact output budgets, next-state
logits and offsets, cancellation/exception rollback, dynamic batch shrinkage,
row reorder and slot-generation reuse. Tiny actual-Metal controls cover rejection
and acceptance boundaries deterministically; pretrained observations remain
separate. A numerical tolerance for next logits never excuses token divergence.
Failures stay in their first source-bound report and cannot be removed by
changing the prompt set.

Development may compare fixed draft depths 1, 2 and 3 with complete prefill and
identical final output budgets. Both target and draft remain resident during the
native control as well. Loading and warm-up are reported separately. Every
control begins with empty KV state. Record whole task, first output availability,
cycle wall time, actual submissions, all model-input positions, memory and swap;
cycle timing is not streaming API end-to-end latency. A full speculative cycle
releases several tokens together, so those tokens share their observed arrival
time. Diagnostic next-state checks run outside timed task intervals.

Only after development is complete may depth, runtime source hashes, exact model
content/quantization and control order be frozen for new held-out execution. For
a speed conclusion, run at least three independent processes, rotate control
order, keep fixed simultaneous arrival schedules and report all failures. The
controls include native target, old serial draft plus replay, fast batched draft
plus trim, and bucket/shared-cap controls with the same recovery mechanism.
Summarize paired task results by prompt/session cluster; retain individual process
measurements and avoid treating tokens or cycles as independent replicates.
Native serial and paired batch have different execution schedules and must be
labelled as such. Two prompts are only a narrow mechanism scope, not evidence of
general quality, hybrid support or superiority over mature engines.

Admission retains the prior conservative 85% load / 65% future-KV limits, two-GiB
runtime reserve and three-copy KV reserve. Each invocation has a five-minute
bound, rejects overwrite and stops on swap growth. Reports bind repository-relative
source hashes, dependency versions, model identities and workload hashes. Private
reports retain raw outputs/errors; public export uses hashes/counts and must derive
coverage from actual observations rather than expected constants. No calibration
curve from a different execution identity is reusable.

## First development observation

The first depth-three development process executed ten controls (the four draft /
recovery combinations under full-depth bucket and shared-cap, plus two forced
heterogeneous controls for batched draft / dense trim). All 20 request outputs
matched native target tokens at their declared 17/32-token budgets. All 40 target
and draft next-token argmax checks and logical-offset checks passed. Swap did not
increase; Metal peak was 817,138,492 bytes. Independently executed tiny device
regressions passed 76 tests.

The run remains **not accepted by the strict numerical gate**: independently
recomputing the entire consumed prefix in one prefill produced next-logit
allclose(1e-5) failures in every control, including the unchanged serial draft /
exact replay baseline. Absolute differences ranged from 0.03125 to 0.0703125.
This observation does not isolate a fast-path defect; it does require separating
ordinary quantized partition sensitivity from changes due to the optimization.
The tolerance is unchanged and every difference remains in the report.

Old serial draft / replay bucket completion was 316.17 ms and batched draft /
dense trim bucket completion was 250.79 ms in this one development process;
native serial tasks totalled 180.39 ms. These observations are not accepted speed
claims. The full-depth bucket/shared-cap pair executes the same requested shape;
only the explicitly heterogeneous pair examines different execution shapes.
A native same-batch control, development selection freeze and independent held-out
repetitions are required before broader performance interpretation.

Evidence: [first public report](../benchmarks/reports/phase3-dense-fastpath-development-first.json)
and [independent verification](evidence/phase3-dense-fastpath-verification-first.json).
The first private report and all 36 measured source files are preserved under
`artifacts/phase3-dense-fastpath/`; the first harness is
`benchmarks/experiments/phase3_dense_fastpath.py`. Re-export without device access:

```sh
.venv/bin/python benchmarks/experiments/phase3_dense_export.py \
  artifacts/phase3-dense-fastpath/development-depth3-first.json \
  /tmp/phase3-dense-first-export.json
```

## V2 development selection

After preserving the first checkpoint, v2 ran depths **1, 2, 3 once each in that
predeclared order**, adding native `mlx_lm.generate.BatchGenerator` with batch two,
prefill batch two and chunk 256. Both model weights remained resident; both
requests arrived at time zero. Empty KV state, input IDs, greedy sampling and
17/32-token budgets were equal. EOS is ignored uniformly here; this protocol is
not suitable for an oMLX comparison that cannot disable EOS.

Across the three development processes, all 60 internal request outputs and six
native-batch outputs matched native serial output. All 120 next-token argmax and
offset checks passed, but **0/120** whole-prefix next-logit allclose checks passed.
The strict numerical gate remains false. No process grew swap. These are three
different-depth development runs, not three repetitions of one final configuration.

The prospective minimum full-task wall-time rule selected depth three, batched
draft, dense trim and bucket execution. Its development task time was 231.04 ms,
versus 315.47 ms for same-process old serial draft / replay and 136.14 ms for native
batch. Thus internal overhead decreased in this observation, while native batching
remained faster. Depth-one native batch took 256.05 ms, and depth-two native batch
136.92 ms; all observations remain visible. Warm-up is only four tokens per model,
so first-use batched compilation/cache effects are included, not silently discarded.

Selection, all candidates, report hashes and explicit gate counts are frozen in
[the v2 development evidence](evidence/phase3-dense-fastpath-development-v2.json).
The new held-out split remains unused. No performance acceptance or release
promotion follows from this development selection.

## Frozen held-out diagnostic

Exactly three new processes ran the reserved held-out split with the chosen depth
three/batched draft/dense trim/bucket configuration, the old serial/replay control
and native same-batch control. Their order rotated across processes. No held-out
parameter selection or extra favourable-result run occurred. All 18 request
outputs matched the native target at their exact budgets, with zero added swap.

Median paired-task time was 180.52 ms for native batch, 367.56 ms for old replay
and 279.13 ms for fast trim. The native engine remains faster in this bounded
scope. The [summary](../benchmarks/reports/phase3-dense-heldout-summary.json)
retains every process plus per-prompt completion, TTFT and ITL observations;
tokens/cycles are not treated as independent samples. Two prompts do not establish
broad serving performance or statistical significance.

A second, untimed oracle replays the exact per-row consumed chunks from each
speculative cycle. The old replay control passes all 12 numerical checks against
that oracle, while fast trim passes 0/12. Whole-prefix checks pass 0/24 across both
controls. Both oracle groups match all next-token argmax values and logical cache
offsets. This separates the original whole-prefix recomputation sensitivity from
the remaining fast-path numerical differences; it does not hide them or establish
float16 state equivalence. Strict acceptance remains false at unchanged tolerance.
[Source-bound verification](evidence/phase3-dense-heldout-verification.json)
records the commands and logs. No runtime or release capability was promoted.

CPU-only reproduction of the public export:

```sh
.venv/bin/python benchmarks/experiments/phase3_dense_heldout_export.py \
  --inputs artifacts/phase3-dense-fastpath/heldout-round0.json \
  artifacts/phase3-dense-fastpath/heldout-round1.json \
  artifacts/phase3-dense-fastpath/heldout-round2.json \
  --output-dir /tmp/deepinfer-heldout-export
```
