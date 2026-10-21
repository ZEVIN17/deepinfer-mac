# Phase 4 source map

Checkpoint: 2026-10-21. Execute [the Phase 4 plan](PHASE4_EXECUTION_PLAN.md).
DS reference remains `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
Local installed MLX-LM is 0.32.2; this is not a latest-upstream audit.
No frozen dependency or historical numerical tolerance changed.

| Method / ownership | Current entry | Phase 4 work | Acceptance |
|---|---|---|---|
| DS expected-output/cost budget and two-generation feedback | `ds/joint_budget.py`, `varlen_verify.py` | Full-cycle cost curves, bounded updates, native candidate; bucket-shape correction explicitly a Mac extension | Fixed depth / DS / DS+Mac / native same-loop ablation and fresh reserved tasks |
| DS token/context/cache scheduling; Mac prefill feedback | `continuous.py`, `server/continuous_worker.py` | Work-sensitive prefill control, admission and waiting-time budgets | Fixed arrival traces, p95 TTFT/ITL, throughput, cancellation, memory and starvation |
| MLX-LM batch and KV primitives | `MLXBatchSession`, `BatchGenerator.next/insert_segments/remove` | Reuse actual ticks and preserve upstream state ownership | In-flight updates, future batches, cancellation and completion |
| Project execution selection | `adapters.py`, `engine.py`, `joint_service.py` | Cold native path, cost identity and hysteresis, explicit safe state transfer | Cross-family fallback, EOS, cancellation, same-history logits; sampling separate |
| Mac dense recovery optimization | `varlen_verify.py`, `hybrid_speculative.py` | Diagnose trim/shape/partition differences before promotion | Frozen strict logits tolerance; Phase 3 failures retained |
| DS ZOS state mapping; MLX async baseline | Existing default-off profiling | Only add overlap after timeline identifies profitable host waits | Dynamic membership, generation IDs, synchronous ablation |
| Upstream contribution candidate | `continuous.py:set_prefill_chunk` | Public dynamic-prefill setter proposal after latest upstream audit | Current/future batches and invalid inputs; reviewable standalone patch |

## First implemented checkpoint

`PrefillController` now ignores decode-only and final prompt-transfer ticks when
adjusting prefill work. Previously a long decode-only interval could repeatedly
halve the next prompt's chunk to one, even though there was no prefill work in
those observations. Two consecutive cheap prefill ticks are required to increase
the chunk; interrupted evidence resets the streak. Mixed decode/prefill wall time
is charged in full. The configured token cap reserves `max_active` decode rows;
this is a conservative chunk bound, not a total GPU work or latency guarantee.

The audited bridge updates both the generator configuration used by future
batches and the current prompt batch. Invalid chunks and unsupported prompt
configuration are checked before mutation. It remains a private-interface bridge
under the existing exact 0.32.2 version guard, not a claimed upstream API.

Validation: 350 portable tests passed; repository Python lint passed. One new
real-Metal tiny Llama regression passed for in-flight chunk change, cancellation,
new prompt batch and completion. The initial device test used a scalar output
limit, which upstream rejects; it was corrected to its documented per-request
list. This was a test setup failure, not evidence of an upstream bug.

No pretrained model equivalence, latency gain, automatic execution selection,
competitive performance acceptance or upstream submission follows from this
checkpoint. Next: fast-trim same-history diagnosis and unified strategy/cost
contracts, followed by frozen arrival-trace measurements. Latest upstream
release/contribution/duplicate-issue audit remains pending before preparing a
submission candidate.

## Native-first execution checkpoint

The [native runtime report](PHASE4_NATIVE_RUNTIME.md) supersedes the first
checkpoint's pending implementation list: native-first CLI auto mode and serial
capability fallback, allocated draft work, transactional one-way native handoff,
shape-specific bounded cost recording, and a quality-gated decision primitive are
implemented. The decision primitive is not yet wired to automatic speculative
service selection. Fixed native prefill is the default; failed adaptive controls
are retained. A [standalone upstream setter candidate](UPSTREAM_CONTRIBUTIONS.md)
has a patch and failing-before/passing-after regression.

Remaining performance work: admission/cache scheduling on native execution,
independently frozen arrival traces against tuned native and oMLX, and accepted
complete-cost calibration. Fast-trim strict numeric acceptance remains open.
