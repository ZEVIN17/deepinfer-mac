# Experimental joint budget and variable-length verification

This opt-in Python mechanism is separate from the public continuous worker. Each tick can supply a new complete
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
