# Portions adapted from MLX-LM 0.32.2 generate.py speculative_generate_step.
# Copyright © 2023 Apple Inc. MIT license; see third_party/licenses/MLX-LM.txt.
"""Version-isolated greedy dynamic-budget adapter, retaining upstream rewind semantics."""

from __future__ import annotations

import time


def adaptive_generate_step(
    prompt,
    model,
    draft_model,
    model_cache,
    draft_cache,
    policy,
    max_tokens,
    rounds,
    cancel=None,
    memory_pressure=None,
    defer_draft_sync=False,
    diagnostics=True,
):
    import mlx.core as mx
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

    if not can_trim_prompt_cache(model_cache) or not can_trim_prompt_cache(draft_cache):
        raise ValueError("Adaptive decoding requires trimmable target and draft caches")
    if hasattr(policy, "capture_replay"):
        policy.capture_replay = diagnostics
    y = mx.array(prompt, mx.uint32)
    draft_y = y
    emitted = 0
    budget = accepted = 0
    cycle_pending = False

    def step(network, cache, inputs, n=1, evaluate=True):
        logits = network(inputs[None], cache=cache)[:, -n:, :].squeeze(0)
        tokens = mx.argmax(logits, axis=-1).astype(mx.uint32)
        if evaluate:
            mx.eval(tokens)
        return tokens

    def rewind():
        trim_prompt_cache(model_cache, budget - accepted)
        trim_prompt_cache(draft_cache, max(budget - accepted - 1, 0))

    try:
        while emitted < max_tokens:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Request cancelled")
            cycle_begin = time.perf_counter()
            budget = min(
                policy.choose(
                    max_tokens - emitted, memory_pressure=bool(memory_pressure and memory_pressure())
                ),
                max_tokens - emitted,
            )
            cycle_pending = True
            scheduler_ms = (time.perf_counter() - cycle_begin) * 1000
            replay = getattr(policy, "last_replay", None) if diagnostics else None
            catchup_tokens = max(0, len(draft_y) - 2)
            accepted = 0
            begin = time.perf_counter()
            draft = []
            current = draft_y
            for _ in range(budget):
                current = step(draft_model, draft_cache, current, evaluate=not defer_draft_sync)
                if defer_draft_sync:
                    mx.async_eval(current)
                    draft.append(current)
                else:
                    draft.append(int(current.item()))
            draft_ms = (time.perf_counter() - begin) * 1000
            if defer_draft_sync:
                draft_array = mx.concatenate(draft) if draft else mx.array([], mx.uint32)
                predicted_array = step(
                    model, model_cache, mx.concatenate([y, draft_array]), budget + 1, evaluate=False
                )
                mx.eval(predicted_array, draft_array)
                predicted, draft = predicted_array.tolist(), draft_array.tolist()
                # Work overlaps: report the combined interval, never pretend
                # enqueue time measures draft GPU execution.
                verify_ms = (time.perf_counter() - begin) * 1000
                draft_ms = 0.0
            else:
                begin = time.perf_counter()
                verify_inputs = mx.concatenate([y, mx.array(draft, mx.uint32)])
                predicted = step(model, model_cache, verify_inputs, budget + 1).tolist()
                verify_ms = (time.perf_counter() - begin) * 1000
            n = 0
            while n < budget and predicted[n] == draft[n]:
                n += 1
            produced = min(n + 1, max_tokens - emitted)
            record = {
                "budget": budget,
                "accepted": min(n, produced),
                "produced": produced,
                "draft_ms": draft_ms,
                "verify_ms": verify_ms,
                "fallback": budget == 0,
                "scheduler_ms": scheduler_ms,
                "catchup_tokens": catchup_tokens if budget else 0,
                "timing_scope": "combined_draft_verify" if defer_draft_sync else "synchronized_parts",
            }
            if replay is not None:
                record["scheduler_replay"] = replay
            if diagnostics:
                rounds.append(record)
            else:
                # Retain only aggregate-count inputs; no replay/state serialization.
                record_minimal = {k: record[k] for k in ("budget", "accepted", "produced", "fallback")}
                rounds.append(record_minimal)
            emitted_before = emitted
            try:
                while accepted < n and emitted < max_tokens:
                    accepted += 1
                    emitted += 1
                    yield predicted[accepted - 1], True
                if emitted < max_tokens:
                    emitted += 1
                    yield predicted[accepted], False
                if emitted < max_tokens:
                    y = mx.array([predicted[accepted]], mx.uint32)
                    if budget == 0:
                        # Catch up only when a later cycle requests draft work.
                        draft_y = mx.concatenate([draft_y, y])
                    elif accepted == budget:
                        draft_y = mx.array(draft[-1:] + [predicted[accepted]], mx.uint32)
                    else:
                        draft_y = y
            finally:
                # Complete even on EOS, cancellation or mid-block close. This
                # wall interval includes synchronous consumer detokenize/emit.
                rewind()
                actual_produced = emitted - emitted_before
                record["produced"] = actual_produced
                record["accepted"] = accepted
                if not diagnostics:
                    record_minimal.update(produced=actual_produced, accepted=accepted)
                record["cycle_ms"] = (time.perf_counter() - cycle_begin) * 1000
                record["cycle_scope"] = "host wall including scheduler, output consumer and rewind"
                if hasattr(policy, "observe_cycle"):
                    policy.observe_cycle(
                        budget, accepted, draft_ms, verify_ms, actual_produced,
                        cycle_ms=record["cycle_ms"], catchup_tokens=record["catchup_tokens"],
                    )
                else:
                    policy.observe(budget, accepted, draft_ms, verify_ms, actual_produced)
                cycle_pending = False
                budget = accepted = 0

    finally:
        # Yielded cycles are rewound/observed by their inner finally, including
        # EOS and generator.close. A failure before that boundary has unknown
        # device progress: discard request-local caches, never guess a rewind
        # or fabricate acceptance feedback for the persistent scheduler.
        if cycle_pending and hasattr(policy, "abort_cycle"):
            policy.abort_cycle()
