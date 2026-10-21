# Phase 4: native execution first

Updated 2026-10-21. Local implementation checkpoint; not a published release.

## Runtime decision

Use MLX-LM `BatchGenerator` as the primary serving executor. The project's
contribution sits around native execution: bounded admission, cache identity,
request lifecycle, optional prefill work control, and measured strategy costs.
The joint speculative executor remains an explicitly enabled research candidate.

In this checkout, `deepinfer-mac serve` defaults to `--execution-backend auto`
and `--continuous-policy upstream`. Auto selects native batch for supported greedy
baseline requests; sampling and unsupported native cache protocols run serially.
Only a specific unsupported-cache exception triggers fallback. Model loading,
resource, identity and arbitrary runtime errors are not hidden by retries.
The Python `create_app()` default remains serial for compatibility; pass
`execution_backend="auto"` explicitly. Direct `InferenceEngine.generate()` is
unchanged. Auto does not yet select speculative execution from measured curves.

```sh
# Native-first local service; choose a free port.
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 \
  --model-config examples/models/smollm2-135m-8bit.local.json
# Explicit old serial mode.
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 --execution-backend serial
# Research-only allocated draft work, exact replay recovery.
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 \
  --model-config examples/models/research-smollm2-pair.local.json \
  --experimental-joint-verification --joint-profile research-smollm2-pair \
  --joint-draft-budget allocated --max-active 2 --batch-context-tokens 4096
```

Stop an example before reusing its port. Existing user service port 8765 is not
used by validation. API responses report selected native/serial strategy and do
not advertise speculative automatic admission.

## New execution mechanisms

- Allocated draft generation respects verification budgets. Serial generation
  stops at each assigned depth; the experimental batched path groups by depth.
  Shared-cap controls still generate the actual shared depth. Unobserved suffixes
  remain censored, not recorded as rejections. The default full-depth control is
  retained, and cost identities distinguish policies.
- `MLXJointBatchSession.switch_to_native()` transfers committed target cache plus
  the unconsumed tail and remaining output allowance to a real BatchGenerator.
  Transfer uses private copies. Failure preserves source state; success discards
  draft caches/history. It is one-way within an epoch and must be called by the
  owner after the prior round has committed and its responses have been emitted.
  Native cancellation, newcomers and drain are covered. Resident draft weights
  remain charged; no memory-release claim is made. This method is not invoked by
  the default auto service, which starts native without loading the draft.
- Bounded complete-cycle observations key model/draft/quantization/execution,
  concrete cache classes, context composition and actual model-call shapes.
  Service response processing is included; observation overhead is separately
  reported. No extrapolation from equal total token counts is performed.
- `ExecutionPolicy` is a portable decision primitive with native cold start,
  quality admission, gain margin and confirmation hysteresis. It is not yet wired
  to automatic speculative service switching. Cost observations are not a claim
  of accepted online calibration.

## Development evidence and limitations

Three independent rotated processes, two public self-authored tasks, natural EOS,
32-token output limits and the same 360M/135M pair measured task medians:

| Executor | Paired task median |
|---|---:|
| Native MLX-LM batch | 124.09 ms |
| Full-depth draft, exact replay | 760.56 ms |
| Allocated draft, exact replay | 529.84 ms |

Allocated draft improves the internal control by approximately 30.3%, but remains
about 4.27 times slower than native. All controls produced identical output IDs
in these three processes. This is a small development ablation with imposed
heterogeneous budgets, not proof of a DS learned-budget gain, broad quality,
heldout acceptance or advantage over oMLX. Load is excluded; prompt prefill and
decode are included; both model weights are resident for all arms. Reports:
`benchmarks/reports/phase4-allocated-draft-round{0,1,2}.json`.

Fresh shape diagnostics on both audited models and contexts 33/65 show exact
cache/logit equality for merge/extract without forward and trim versus independent
same-shape slicing. Four comparisons of Q=4 retained prefixes versus Q=1 replay
fail the original strict logit tolerance; Q=2 and Q=4 replay controls pass. This
points to forward partition sensitivity in these cases, not a proven trim
mutation error. No historical failure is overwritten and fast-trim remains
experimental. The first diagnostic's independent oracle used the wrong state
arity; its failed report is retained alongside corrected v2 results.

The first native arrival development trace exposed a bad adaptive setting:
with a 10 ms target and no useful minimum chunk, task time rose from 1.384 s
(fixed 1024) to 4.202 s, despite lower p95 output intervals. It fails throughput
acceptance. A minimum chunk now bounds the controller; fixed native policy is
the default. Adaptive results require comparison against tuned fixed chunks,
not only the weakest native configuration. No concurrency speedup is accepted.

Reproduce with `phase4_cache_diagnosis.py`, `phase4_allocated_draft.py`,
`phase4_native_arrivals.py` and `phase4_verify_service.py` under
`benchmarks/experiments/`. All require `--execute`; performance runners reject
existing output files and guard memory/swap. Development traces are exposed and
must not become heldout acceptance tasks.

The follow-up floor-64 development candidate completed in 1.804 s versus
1.390 s for fixed 1024, with p95 ITL 19.43 ms versus 11.06 ms. Outputs matched
across all four controls and swap did not grow. It also fails acceptance; no
extra favourable repetitions were run. The implementation's conservative default
floor is 32 for explicit interactive use; the measured floor-64 configuration is
not represented as that default's performance.

Final checkpoint validation: 364 portable tests, 95 real-Metal tests, repository
Python lint, and actual service 46/46 checks. See
[service evidence](evidence/phase4-service-verification.json). The unchanged
static site rebuild passes 138 local links across 8 pages. These are local
verification results; no deployment, public PR or release was performed.
