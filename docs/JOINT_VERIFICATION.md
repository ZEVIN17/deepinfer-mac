# Experimental joint budget and variable-length verification

This opt-in Python mechanism also has a separate explicit experimental service entry
(described below); the default serial and baseline continuous services are unchanged. Each tick can supply a new complete
cost key for dynamic batch membership; strategy and identity mismatches reject even
when forced lengths are used. It
has no pretrained-model, hybrid-batch, quality or comparative speed acceptance.
The default API/CLI remains unchanged. Source controls precede performance claims.

## Confidence and cost boundary

`ds/joint_budget.py` gives every vector explicit `conditional` or `cumulative`
semantics, model content identity, calibration identity and provenance. Conditional
vectors become cumulative survival once; cumulative vectors bypass multiplication.
`EmpiricalConfidence` observes a contiguous true prefix, optionally the first
actual false, then censored `None`. Position coverage and pre-observation conditional Brier
scores remain diagnostic outputs; they are not independent held-out calibration. EOS, output truncation and cancellation never
turn the unobserved suffix into rejections. Its beta-smoothed conditional estimates
are empirical predictions, not a learned ConfidenceHead. A `trained_head` label
is only a data contract; there is no trained-weight loader or enabled trained head.

`JointBudget` keeps two committed histories keyed by `(uid, generation)`. Only
successful whole rounds advance them. The original oracle-tested MTPVarlen selects
total work from N−2 and allocates from N−1. A new or reused slot has zero confidence.
A zero score can still win a zero-score flat-index tie; it is not forbidden work.
The MLX allocator implements actual array cumprod/rank with current row-major
flat-index ties. Host shape materialization remains explicit. No fused Metal kernel
or upstream CUDA oracle execution is claimed.

`CostKey` separates target, draft, quantization, batch size, context bucket,
execution strategy and calibration identity. Curves must be measured **complete
cycles**, including draft, verification, exact recovery, catch-up and host work.
The existing MTP cost selector receives that curve once, not an additional sum of
already included components. No curve means full available work, not a fabricated
cost model. This experimental session does not auto-calibrate a curve; callers
must freeze independent development observations and install them explicitly.

## Actual executed shapes and transactional caches

`MLXVariableVerifier.verify(rows, lengths, strategy=...)` takes per-row `q` including
the golden/unconsumed tail token. A row provides at least `q−1` proposals. Strategies:

- `bucket`: equal-q dense rows merge through the real locked MLX-LM KVCache API;
  the target receives `[bucket_size, q]`, so different budgets change actual work.
- `padded_control`: the target receives `[batch, max(q)]`, while each row observes
  only its requested prefix. Right-padded temporary cache states are **discarded**;
  they are never extracted or committed. This is a causal-logit control, not a
  valid cache-reuse path and not a compute-saving claim.
- `shared_cap`: every row actually verifies `max(q)`, requiring sufficient real
  proposals. It uses the same exact recovery strategy as the bucket path.

Audited recurrent/hybrid caches execute **single rows**, never cross-row hybrid
microbatches. Every original target cache is snapshotted before the round. All
verification uses disposable deep-copied state. On success the original single-row
state is restored and exactly replays `[old_tail, *emitted_tokens[:-1]]`. Even full
acceptance uses this conservative replay. Rejection, EOS and output truncation
therefore have the same recovery policy across controls. Any exception/cancellation
restores every original row. No approximate bounded recurrent replay is used.

`JointSpeculativeSession.tick` runs real autoregressive draft model calls, chooses
joint budgets, invokes the verifier, then restores/replays draft caches onto the
same committed prefix. This is actual execution, not only an offline scheduler.
It accepts an explicitly labelled same-target mechanism control but never promotes
it to a production draft or speed result. The public model adapter's capability
matrix is deliberately unchanged until independent real-model acceptance.

Returned accounting includes requested/effective/padded/executed verification
work, replay work, all real model-call shapes and counts, draft/catch-up work and
complete cycle wall time. A model submission is **not** a GPU kernel count; the
latter is `null`. Verifier-only timing explicitly excludes draft and has null
full-cycle time. Session timing includes all work and statistical updates. Device
work is synchronized before reporting. Callers own normal model/context memory
admission and must never bypass it to run this experimental API.

## Verification entry points

```sh
# Portable; 2,025 independent source-reference comparisons plus contract tests.
.venv/bin/python -m pytest -q tests/test_joint_budget.py
# Requires explicit GPU baton. Tiny random models only; no downloads or service.
.venv/bin/python -m pytest -q tests/test_varlen_metal.py --run-metal
```

The tiny device controls exercise real MLX rank, real length buckets, cancellation
rollback and distinct target/draft network execution. Random models are mechanics
controls, not language quality or performance evidence. Actual batch shape can
change near-tie greedy decisions, as preserved in the section 3 diagnosis; exact
serial equality and same-batch controls must remain separately reported. Larger
pretrained performance comparisons remain subject to existing resource guards.

## Observed small-model mechanism control

The separately source-bound [Smol mechanism report](../benchmarks/reports/phase3-joint-mechanism.json)
uses the target itself as a deliberately labelled draft control, not a deployable
acceleration configuration. With forced heterogeneous lengths, bucket verification
executed 7 positions versus 12 for the padded control; including draft and exact
recovery/catch-up, total model-input work was 30 versus 35 positions in that round.
This establishes changed real executed work in that specific control, **not lower
latency, DS adaptive policy acceptance, or an advantage over mature engines**.

The English/code controls matched serial-native output, while the Chinese control
retained a near-tie divergence at index 3 across bucket/padded/shared-cap paths.
The strict serial gate remains false. Target/draft next logits on their committed
histories agreed; independent token-by-token reconstruction kept the next argmax
but had nonzero logits differences. These are separately recorded diagnostics,
not grounds to relax exact-token acceptance or discard the failing case. Reports
keep their original source hashes; later contract guards do not relabel those runs.

## Experimental service entry (independent acceptance required)

The explicit service opt-in connects the joint executor to the existing bounded
queue, admission, UID/slot-generation lifecycle and SSE transport. Default serial
and ordinary continuous services retain their existing behavior. A configured
custom ModelSpec must contain distinct target and draft sources (local content
identity or pinned repository commits); both resolve through the existing pair
snapshot/tokenizer and combined weight/runtime preload checks. Dense `KVCache`
only is accepted by this service, independently of the mechanism's isolated
hybrid tests. Example configuration does not constitute model acceptance.

```sh
# Only after checking available memory; use a free port, not an existing service.
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 \
  --model-config examples/models/research-smollm2-pair.local.json \
  --experimental-joint-verification --joint-profile research-smollm2-pair \
  --joint-depth 3 --max-active 2 --batch-context-tokens 4096
```

Python uses `create_app(experimental_joint_verification=True,
 joint_config=JointServiceConfig(profile=...), model_registry=registry,
 continuous_config=ContinuousConfig(...))`. The registered pair receives greedy
`mode="baseline"` requests; results explicitly record `requested_mode=baseline`
and `effective_mode=joint_experimental`. Unsupported sampling, profiles, modes
and comparison reject with 422; there is no silent promotion or fallback. Model
metadata lists this experimental capability separately with `verified=false`.
CLI opt-in cannot combine with `--execution-backend continuous`.

Admission includes both models, both contexts, conservative target/draft
snapshot/copy/replay scratch and longest-row padded verification memory. Both
models reload through the combined preload guard for each new drained group;
this deliberately sacrifices warm group reuse. Prefix reuse is cold. Prompt
prefill runs synchronously in bounded chunks; a new prefill cannot be preempted
by an existing decoder. `tick_token_budget`, upstream continuous prefill batches
and adaptive chunk policy do not bound joint ticks. They are omitted from joint
result metrics. The first tail is the last prompt token; both caches initially
contain exactly `prompt[:-1]`. Completed rounds leave the final emitted token
pending in both caches. EOS and remaining output limits release the row.

Only a committed verification round creates token events. Cancellation during
prefill discards its private state without failing existing requests. Cancellation
during joint verification is observed at the completed-round boundary; remaining
events for a cancelled request are suppressed and its caches discarded. A failed
joint round terminates affected requests after the executor rolls back, then the
worker rebuilds the session. No completed cache is exported or reused. Exact
serial-native agreement remains unvalidated: shape-dependent near-tie differences
remain part of the experimental boundary.

Without `--joint-cost-file`, all available draft work is allocated; empirical
history is not a trained or calibrated confidence head. An optional JSON cost
file must contain `scope: complete_cycle`, `partition: development`, exact `depth`,
`mlx_lm_version`, and `source_sha256` from `joint_service.runtime_source_hashes()`.
Each `curves` entry has a complete `CostKey` under `key` and actual `measurements`
containing `total_q` and `full_cycle_ms` for every legal total budget. The loader
computes medians; it does not synthesize missing costs. Pair, quantization,
strategy, calibration, active count and exact consumed context must match. A
missing dynamic key uses full budget and reports `cost_curve_available=false`.
These provenance checks cannot certify the truth of caller-supplied measurements;
independent measurement and held-out evaluation are still required.

Portable service contract checks: `python -m pytest -q tests/test_joint_service.py`.
Their controlled session fixtures test lifecycle/API contracts only and are never
model, language-quality, latency or throughput evidence. Real service acceptance
must be recorded separately before capability promotion.
