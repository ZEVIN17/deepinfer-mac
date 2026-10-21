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
