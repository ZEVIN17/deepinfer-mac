# Distinct pretrained-pair cost experiment

`benchmarks/experiments/phase3_pair_cost.py` measures the audited local
SmolLM2-360M-Instruct-6bit target and SmolLM2-135M-Instruct-8bit draft. It does not
download models or enable a public serving capability. Exact content identities,
complete vocabulary/template checks and a conservative 2 GiB runtime plus
three-copy KV reserve precede loading. Execution requires the root GPU baton.

The public `phase3_pair_workload_manifest.json` freezes the hashes and scope of
two development and two held-out prompts. Raw prompts and outputs remain under
excluded `artifacts/phase3-pair-cost/`. Reproduction requires that exact private
workload file; the public hash alone is not a downloadable benchmark corpus.
Development uses 128 input tokens; held-out uses 256. All final controls produce
exactly 32 tokens per request, with EOS ignored equally. These are bounded
mechanism workloads, not an instruction-quality benchmark.

```sh
# CPU-only scope/manifest validation; imports no MLX runtime.
.venv/bin/python benchmarks/experiments/phase3_pair_cost.py \
  --output artifacts/phase3-pair-cost/development.json
# Only after GPU ownership and independent source review:
.venv/bin/python benchmarks/experiments/phase3_pair_cost.py --execute \
  --stage development --output artifacts/phase3-pair-cost/development.json
# Frozen development source/model/workload identity must match exactly.
.venv/bin/python benchmarks/experiments/phase3_pair_cost.py --execute \
  --stage heldout --calibration artifacts/phase3-pair-cost/development.json \
  --output artifacts/phase3-pair-cost/heldout.json
```

Development observes a short full-depth prefix, then measures two actual full
cycles for every legal total budget at batch sizes one and two. Each sample uses
fresh target/draft KV state and records its actual row lengths. The median curve
includes draft, snapshots, verification, exact recovery, draft catch-up and host
scheduler work; prefill is separately timed. Its key includes both model content
identities, both quantizations, batch size, 512-token context bucket, bucket
execution strategy and calibration identity. This representative curve is not a
universal function of total token count: different bucket shapes can cost more.

Held-out compares cost-driven bucket execution with full-depth fixed bucket and
shared-cap controls. A separate forced heterogeneous bucket/padded pair isolates
executed-shape work, using the same draft depth and exact recovery. Every run has
the same final output budget. Dynamic batch shrinkage uses separately measured
batch-one costs. Native serial control calls the installed MLX-LM
`generate_step`; diagnostic log-probability margins are computed outside its
measured interval. Cold task time includes target/draft prefill with resident
warmed weights. Initial pair loading and warm-up are separately reported. No
per-token report writes are included in measured task time.

Frozen development probabilities are scored on only the actually verified
held-out prefix. First rejection is observed; all subsequent unverified positions
remain null. Position coverage, observed acceptance and Brier scores are retained
separately from online adaptation. Small counts cannot establish calibration.
All cycles retain actual model submissions/shapes, verification and total input
positions, complete wall time and output divergence. Near ties do not relax exact
native-token equality. Each stage is one process with a five-minute bound, stops
on swap growth, preserves partial failure artifacts and refuses overwriting.
Three independent benchmark processes and broader quality coverage would still
be needed for a performance claim.

## Recorded bounded result

The first frozen development and held-out processes completed with zero added
swap and a held-out Metal peak of 809,765,692 bytes. All five controls matched
native target output for both held-out tasks (10/10), each with the same 32-token
budget. This new scope does not erase previous batch-path equality failures.

The forced heterogeneous bucket control executed 76 verification positions versus
109 padded positions (30.3% fewer). Including draft/recovery/catch-up, decoding
used 333 versus 366 model-input positions (9.0% fewer). Their complete task times
were 912.66 and 912.24 ms: the saved positions did not yield a measured latency
benefit. Adaptive bucket time was 576.54 ms versus 563.57 ms for fixed full depth
(2.3% slower) and 252.38 ms for the two serial native tasks combined (2.28 times
native time). One process and two tasks do not establish statistical significance.

The adaptive cycles totalled 481.99 ms: 200.13 ms inside verification/recovery,
10.99 ms in the snapshot/scheduler/allocation interval, and 270.87 ms remaining
for draft/catch-up/statistics/other host work. These nested timing intervals
support investigating proposal and exact replay overhead; they do not isolate a
GPU kernel bottleneck or justify asynchronous overlap claims.

Development used forced lengths and therefore **did not include ordinary
`JointBudget.choose` or MLX rank allocation**. The frozen curve retains this
limitation. Held-out adaptive cycles execute and time both. No measurement or
curve was retried to select a favorable result.

[Summary and source-bound reports](evidence/phase3-pair-cost-experiment.json)
retain all numbers and sparse held-out confidence diagnostics. Public reports
replace generated token arrays with count/SHA256; raw outputs remain private.
`benchmarks/experiments/phase3_pair_export.py` reproduces this transformation from
the two raw artifacts without rerunning inference.

## Development-only wall-stage diagnostic

`--profile-stages` adds optional timing to the same development workload. It is
rejected for held-out runs. Default execution keeps the original serial draft and
exact replay algorithms; no new cache commit or batching optimization is enabled.

```sh
# Requires explicit GPU ownership; same admission, five-minute bound and no-overwrite guard.
.venv/bin/python benchmarks/experiments/phase3_pair_cost.py --execute \
  --stage development --profile-stages \
  --output artifacts/phase3-pair-cost/development-stage-profile.json
```

`cycle_stages_ms` consists of disjoint host wall intervals; its sum plus
`cycle_profile_residual_ms` equals `full_cycle_ms`. Its `verification_total`
contains the separate `verification_stages_ms` breakdown: do not add nested
verification stages to the cycle stages. Forward/argmax/eval intervals include
lazy graph construction and the existing synchronization, not isolated GPU kernel
time. Host scalar/list extraction is recorded separately. Copy/merge/input setup,
target restoration/replay, draft restoration/catch-up and statistical commit are
also labelled. No extra device synchronization is inserted, so lazy copy/merge
work can be charged to the subsequent eval. These labels are not hardware hotspot
attribution. Clock/recording overhead remains part of profiled wall time.

Profiled cost identities append `draft=serial|recovery=exact_replay|profile=wall_v1`
to the verification strategy. They cannot silently reuse original unprofiled
curves. This diagnostic does not replace previous frozen reports or establish
latency benefit; optimization selection must follow its actual observations.

The held-out loader additionally requires explicit unprofiled execution metadata,
exactly the batch-one and batch-two expected CostKeys, and finite positive points
for every legal budget. Missing, duplicate or foreign keys reject before model
loading instead of silently selecting full-budget fallback. Older artifacts without
this metadata remain historical evidence; they are not relabelled as new calibration
inputs. The independently recorded stage profile predates this input-only guard;
its original measured source hashes remain unchanged.
