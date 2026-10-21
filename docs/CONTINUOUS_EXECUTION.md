# Experimental continuous execution

Phase3 section 3 adds an opt-in worker that advances one real MLX-LM 0.32.2
`BatchGenerator.next()` per scheduling tick. The default service remains serial.
This is greedy baseline continuous batching, not cross-request DS speculative
verification, a trained confidence head, or ZOS overlap. Runtime signatures and
cache merge behavior were inspected against the installed locked source.

## Run

From the repository, using the already prepared small-model config and a free
port (the example does not modify an existing service):

```sh
.venv/bin/python -m deepinfer_mac.cli serve --port 8877 \
  --model-config examples/models/smollm2-135m-8bit.local.json \
  --execution-backend continuous --max-active 4 \
  --continuous-policy interactive --tick-token-budget 512 \
  --batch-prefill-chunk 256
```

The API model ID is `smollm2-135m-8bit`. Use `mode: baseline`, `temperature: 0`.
The optional request `priority` is `interactive` (default) or `background`.
`/health` exposes the execution backend, policy, active UID count, queued count,
preparing request and cleanup error count. Public model capability metadata is
not promoted automatically when a cache protocol probe succeeds.

Current status is an experimental execution checkpoint, **not full section 3
acceptance**. The actual small-model probes below found a batch-shape dependent
token divergence from the serial native control. Keep the default serial backend
when that exact native greedy token sequence is required. Opting into batching
does not guarantee batch-size invariant tokens.

Continuous execution accepts compatible baseline greedy requests into one
same-profile group. Sampling, penalties, speculative modes and comparison jobs
wait for the batch to drain, then run through the guarded serial engine
exclusively. Different model profiles also wait for a group boundary. A profile
name never causes different weights to share an upstream batch; the registry
and loaded snapshot identity remain authoritative.

Prefix reuse in this experimental backend is currently cold, even when the
request has the default `use_cache: true`. Results explicitly record
`use_cache_requested`, `use_cache_effective: false`, `cache_reuse: cold` and zero
cache hits. Default serial cache behavior remains available. No batched cache is
stored under an inferred token boundary.

## Lifecycle, identity and resource ownership

Each request has a stable `request_id`, an external slot and a monotonically
increasing `slot_generation`. Backend UIDs map to request objects, never to row
indices. UID counters may restart with a new generator; generations remain
distinct across group changes. The public historical `status` vocabulary remains
queued/running/completed/cancelled/failed. The lifecycle is:

`queued -> prefill -> decode -> draining -> finished`

Cancellation/failure can terminate any unfinished phase. Exclusive serial work
may go from prefill to draining because it does not expose individual backend
steps. `draining` finalizes the host detokenizer/output, not a claim that every
UID's speculative GPU work has been independently synchronized. Upstream's same
stream dependencies retain pending arrays; group close explicitly synchronizes.
EOS is kept in full token IDs consistently with the existing engine, while never
being passed to the detokenizer. Every request owns its detokenizer and buffers.

Only the owner thread loads models, inserts/removes UIDs, advances ticks or
closes device resources. `remove([uid], return_prompt_caches=False)` handles
cancellation at prefill and decode. Completion boundaries come from actual
`GenerationBatch.Response.prompt_cache` and `all_tokens`. The worker records
their supplied boundary and discards completed caches. It does not guess a
consumed prefix from generated output length.

The external queue is bounded, including a request being prepared. All pending,
prefill and decode UIDs count toward `max_active`, independently of upstream
`completion_batch_size` behavior. Completed history/events are bounded. Queue
overflow returns 429. A request that cannot fit even as the only active UID is
rejected with 503; oversized inputs are never truncated. Capacity pressure while
other requests are active defers admission until they drain. Preparation errors
return 422 and backend failures return 500. A failed insertion discards the whole
potentially mutated batch before recovery.

Admission reserves model memory once, per-request exact KV and final extraction
space, batched padding to the longest context, prefill logits scratch, and a
temporary reserve. It checks aggregate context tokens and actual MLX allocator
usage. RSS and Metal are reported separately, never added as independent memory
owners. OS available memory only pays for the not-yet-materialized reservation.
The optional `--batch-memory-mb` also reaches the default engine's pre-load
guard; it is an admission budget, not an OS hard allocation limit.

The existing adapter's hybrid estimate includes a conservative fixed 2 GiB
state allowance. Keeping KV/extraction reserves can therefore charge at least
4 GiB per hybrid request before padding and scratch. This can legitimately reject
hybrid concurrency on a 24 GiB machine. No estimate or guard is bypassed to
manufacture an accepted batch. Method presence is only protocol eligibility;
real model acceptance must be reported separately.

Shutdown, disconnect, cancellation and backend exceptions terminate waiting jobs
and release external reservations/slots. ASGI disconnect is covered even during
the first role-only SSE chunk, before the event iterator starts. Removal/close
errors are retained; cleanup failure cannot silently leave SSE clients waiting.
An uncertain batch is discarded, and a failed session close recreates the engine
before recovery. An engine-close failure stops the owner and terminates queued
requests. `close()` reports retained cleanup failures after joining the thread.

## Tick work and fairness

The native container already decodes before processing prefill. This ordering
belongs to MLX-LM. With one prefill row and chunk size bounded by
`tick_token_budget - max_active`, the wrapper limits a tick's prefill token work
while reserving room for all active decode/transition rows. It does not provide
per-request draft/verification vector caps. Upstream `insert_segments` receives
only each request's **total output limit** through `max_tokens`.

`next()` returns after one upstream step, including pure prefill ticks. The
worker then handles cancellation and new admission before the next tick.
`next_generated()` is deliberately unused: it swallows pure prefill ticks.
Decode output delivery still waits for the current tick's synchronous prefill;
the service does not claim nonblocking prefill or hard latency bounds.

There is currently one native prefill slot. An already decoding short request can
advance while a long prompt is processed chunk by chunk. A newly arriving short
prompt waits behind a long prompt that already owns that prefill slot. Inserting
its UID does not preempt or rotate the long prefill. The real probe preserves
both arrival directions so this distinction remains visible.

`interactive` adjusts the next prefill chunk against measured whole-tick time,
gives interactive work priority, ages background work, and closes admission after
the group quota when an incompatible waiter exists. Finite active requests then
drain before another group is selected. `upstream` keeps a fixed chunk and FIFO
selection, while retaining bounded admission, cancellation and the group quota.
Aging and quotas prevent endless compatible arrivals from permanently excluding
another model or background request. These are platform work/resource policies,
informed by DS scheduling separation, not the Phase3 section 4 joint confidence
allocator.

For mechanism comparisons use the same workload with:

| Switch | Meaning |
| --- | --- |
| `--execution-backend serial` | Existing complete-request engine/worker |
| `continuous --continuous-policy upstream --max-active 1` | Native chunked tick path, no simultaneous UID batch |
| `continuous --continuous-policy upstream --max-active 4` | Native continuous batching, fixed chunk/FIFO |
| `continuous --continuous-policy interactive --max-active 4` | Same native batching with chunk/priority policy |

Match the token budget, maximum chunk and memory budget when comparing policies.
Changing the active cap is an explicit workload ablation, not evidence of DS
algorithmic gains. No speed advantage is inferred from these switches existing.

## Reproduce validation

Portable tests exercise the real wrapper boundary with device-free stubs and an
independent control fixture; they are not model or performance data:

```sh
.venv/bin/python -m pytest -q -m 'not metal'
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
# CPU-only execution plan; does not import MLX or load arrays.
.venv/bin/python benchmarks/experiments/phase3_verify_continuous.py
```

The real probe is prepared but must run only with the coordinator's GPU baton:

```sh
.venv/bin/python benchmarks/experiments/phase3_verify_continuous.py --execute \
  --output results/phase3-continuous/smollm2-interactive-fresh.json
```

It uses existing local <=200 MB dense Llama-family weights only, downloads
nothing, starts no listening server and refuses to overwrite an earlier report.
Native full-token controls, all divergences and first mismatch positions are
retained. Cases cover fixed arrivals at concurrency 1/2/4, both arrival directions,
prefill/decode cancellation and recovery. Acceptance requires actual shared
decode/prefill ticks, both arrival directions, both cancellation phases, full
native token equality, all seven cases and unchanged runtime/harness hashes.
Actual TTFT/ITL are stamped at emission using the monotonic submission origin,
including queue/load and prefill synchronization. One-process descriptive
p50/p95, completion counts, whole-case throughput, version/identity and resource
snapshots are recorded. These are functional observations, not the required
three-process competitive performance acceptance.

## Actual Smol checkpoint: strict serial control remains failed

Frozen-runtime probes on SmolLM2-135M-Instruct 8bit/group64 completed all seven
cases under both `interactive` and fixed-chunk `upstream` policies. Each policy
completed 13 outputs, observed genuine decode/prefill shared ticks, both arrival
directions, and both cancellation phases with recovery. Runtime/harness hashes
stayed unchanged and system swap delta was zero. This verifies those observed
execution/control facts, not full section 3 or competitive performance acceptance.

The original exact serial-native gate remains **false** in both reports:

| Control | Observed result |
| --- | --- |
| Interactive, all completed outputs | 12/13 exactly equal to serial native |
| Upstream policy, all completed outputs | 12/13 exactly equal to serial native |
| Policies compared to each other | All 13 corresponding full outputs identical |
| Independent native BatchGenerator B4 | All four outputs identical to both wrapper B4 outputs |
| Independent native BatchGenerator Chinese B1 | All 48 tokens identical to serial native |

The single divergence is the Chinese request in the four-request batch, at
zero-based output index **39** (40th token): serial native emits token **27830**,
while both wrapper policies and the direct native B4 container emit **179**.
Full outputs and the original failed gates remain intact. The policy reports
preserve model content identity, rendered prompt lengths, full output hashes,
versions, actual queue/emit timing, resource snapshots and raw-report hashes:

- [Interactive actual report](../benchmarks/reports/phase3-continuous-interactive.json)
- [Fixed-chunk upstream actual report](../benchmarks/reports/phase3-continuous-upstream.json)
- [Direct native B1/B4 diagnosis](../benchmarks/reports/phase3-continuous-upstream-diagnosis.json)
- [Actual API/SSE cancellation and recovery](../benchmarks/reports/phase3-continuous-api.json)
- [Independent verification and source/report hashes](evidence/phase3-tick-verification.json)

The separate real-Metal ASGI check completed one request, two concurrent SSE
requests and a recovery request: four completed streams, 50 SSE events each,
matching full tokens/text, plus API prefill cancellation. It started no listener,
closed its worker and left the user's port 8765 untouched. Fresh verification
also passed 296 portable tests and 41 actual tiny-Metal regressions; these counts
are correctness checks, not performance evidence.

An additional CPU inspection of the retained raw outputs verified equal native
prompt/output controls and model identity across the two policy processes,
26 completed request event histories with contiguous token indices and matching
result tokens/cache boundaries, and unique request/slot generations. No observed
identity mismatch, token event remapping error or cross-request output exchange
explains this divergence. Matching direct upstream B4 behavior localizes the
observed boundary to the upstream batch execution/shape path; it does **not**
prove a floating-point rounding cause or rule out every backend state issue.
The direct control inserts all four rows together at fixed chunk 32; it does not
faithfully replay every wrapper arrival and dynamic shape trajectory. No
teacher-forced logits or top-two margins have been collected yet.

Continue by comparing B1/B4 at the same fixed pre-divergence prefix, with identical
prompt IDs, weights, cache positions and dtype. Inspect actual logits, top-two
margins and prefill/decode/padding boundaries before selecting any fix. A
deterministic per-row or serial fallback may preserve an exact control at a cost
that must be measured; no such new fallback is claimed implemented. Do not add
token tolerances, remove the Chinese case, change its seed or replace the original
strict gate with batch-native agreement. B1 matching this one sample is also not
a universal batch-size invariant guarantee.

The other remaining section 3 boundary is new short-prefill preemption: the
current single prefill slot still lets the long prefill finish before starting a
new short prefill. The two-direction timelines are retained for that next design
decision. Cross-request DS confidence/work allocation can proceed as a separate
correctness-controlled mechanism without declaring these two limitations solved.
