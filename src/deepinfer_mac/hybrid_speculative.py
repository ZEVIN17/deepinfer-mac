"""Greedy hybrid speculation with exact checkpoint restoration and prefix replay.

DS MTPVarlen supplies the budget through the same policy protocol as the normal
adapter. Recovery is an explicit Mac platform adaptation, not an upstream DS
recovery implementation. Locked MLX-LM 0.32.2 ArraysCache/KVCache state setters
include recurrent tensors, lengths/padding, and KV offsets. We never trim a
recurrent state or replace cache objects owned by another container.
"""

from __future__ import annotations

import copy
import time


def snapshot_cache(caches):
    """MLX array deepcopy retains immutable device buffers; copy host containers."""
    return copy.deepcopy([cache.state for cache in caches])


def restore_cache(caches, snapshot):
    if len(caches) != len(snapshot):
        raise ValueError("Cache checkpoint shape changed")
    for cache, state in zip(caches, snapshot):
        cache.state = copy.deepcopy(state)


def hybrid_generate_step(
    prompt, model, draft_model, model_cache, draft_cache, policy, max_tokens,
    rounds, cancel=None, memory_pressure=None, diagnostics=True,
    verification_hook=None, cache_adapter=None,
):
    """Yield (token, accepted) while keeping the final output token unconsumed.

    The input prompt is the unconsumed tail after identical target/draft prefill.
    For hybrid operation it must contain one token. Budget zero does no draft
    network work. When speculation resumes, pending inputs are included in its
    first draft forward; all restore/replay work belongs to complete cycle cost.
    """
    import mlx.core as mx

    if len(prompt) != 1:
        raise ValueError("Hybrid adapter expects one unconsumed prompt-tail token")
    if hasattr(policy, "capture_replay"):
        policy.capture_replay = diagnostics
    take_snapshot = cache_adapter.snapshot if cache_adapter is not None else snapshot_cache
    apply_snapshot = cache_adapter.restore if cache_adapter is not None else restore_cache
    y = int(prompt[0])
    pending = []
    emitted = 0
    cycle_pending = False
    target_snapshot = draft_snapshot = None

    def replay(network, cache, inputs):
        if inputs:
            network(mx.array(inputs, mx.uint32)[None], cache=cache)
            mx.eval([c.state for c in cache])

    try:
        while emitted < max_tokens:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Request cancelled")
            cycle_begin = time.perf_counter()
            budget = min(policy.choose(max_tokens - emitted,
                                       memory_pressure=bool(memory_pressure and memory_pressure())),
                         max_tokens - emitted)
            cycle_pending = True
            scheduler_ms = (time.perf_counter() - cycle_begin) * 1000
            old_y, old_pending = y, pending
            target_snapshot = draft_snapshot = None
            begin = time.perf_counter()
            if budget:
                target_snapshot, draft_snapshot = take_snapshot(model_cache), take_snapshot(draft_cache)
            snapshot_ms = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            draft_arrays = []
            current = mx.array([*old_pending, old_y], mx.uint32)
            for _ in range(budget):
                logits = draft_model(current[None], cache=draft_cache)[:, -1:, :].squeeze(0)
                current = mx.argmax(logits, axis=-1).astype(mx.uint32)
                mx.async_eval(current)
                draft_arrays.append(current)
            draft_array = mx.concatenate(draft_arrays) if draft_arrays else mx.array([], mx.uint32)
            verify_logits = model(mx.concatenate([mx.array([old_y], mx.uint32), draft_array])[None],
                                  cache=model_cache).squeeze(0)
            predicted_array = mx.argmax(verify_logits, axis=-1).astype(mx.uint32)
            mx.eval(predicted_array, draft_array)
            predicted, draft = predicted_array.tolist(), draft_array.tolist()
            combined_ms = (time.perf_counter() - begin) * 1000
            n = 0
            while n < budget and predicted[n] == draft[n]:
                n += 1
            limit = min(n + 1, max_tokens - emitted)
            if verification_hook is not None:
                verification_hook({
                    "generated_before": emitted, "produced": limit, "budget": budget,
                    "inputs": [old_y, *draft], "predicted": predicted,
                    "cache_snapshot": target_snapshot, "logits": verify_logits,
                })
            accepted = 0
            actual = []
            record = {"budget": budget, "accepted": 0, "produced": 0, "fallback": budget == 0}
            if diagnostics:
                record.update(snapshot_ms=snapshot_ms, scheduler_ms=scheduler_ms,
                              draft_ms=0.0, verify_ms=combined_ms,
                              timing_scope="combined_draft_verify", catchup_tokens=len(old_pending),
                              recovery="exact snapshot restore + consumed-prefix replay")
                record["draft_token_ids"] = draft
                replay_input = getattr(policy, "last_replay", None)
                if replay_input is not None:
                    record["scheduler_replay"] = replay_input
            rounds.append(record)
            try:
                for index in range(limit):
                    from_draft = index < n
                    token = predicted[index]
                    actual.append(token)
                    accepted += int(from_draft)
                    emitted += 1
                    yield token, from_draft
            finally:
                begin = time.perf_counter()
                consumed = [old_y, *actual[:-1]] if actual else []
                restored_target = restored_draft = 0
                if budget:
                    # Target verified old_y + all proposed drafts. Only keep it
                    # when that is precisely the already-consumed prefix.
                    if len(actual) != budget + 1:
                        apply_snapshot(model_cache, target_snapshot)
                        replay(model, model_cache, consumed)
                        restored_target = len(consumed)
                    if len(actual) == budget:
                        # Draft already consumed pending+old_y+draft[:-1].
                        pending = []
                    elif len(actual) == budget + 1:
                        # Full accept: final accepted draft is still unconsumed
                        # by the draft model. Batch it into the next first step.
                        pending = draft[-1:]
                    else:
                        apply_snapshot(draft_cache, draft_snapshot)
                        replay(draft_model, draft_cache, [*old_pending, *consumed])
                        restored_draft = len(old_pending) + len(consumed)
                        pending = []
                else:
                    pending = [*old_pending, *consumed]
                if actual:
                    y = actual[-1]
                recovery_ms = (time.perf_counter() - begin) * 1000
                cycle_ms = (time.perf_counter() - cycle_begin) * 1000
                record.update(accepted=accepted, produced=len(actual))
                if diagnostics:
                    record.update(recovery_ms=recovery_ms, target_replay_tokens=restored_target,
                                  draft_replay_tokens=restored_draft, cycle_ms=cycle_ms,
                                  cycle_scope="host wall including snapshots, draft/verify, output and restore/replay")
                if hasattr(policy, "observe_cycle"):
                    policy.observe_cycle(budget, accepted, 0., combined_ms, len(actual),
                                         cycle_ms=cycle_ms, catchup_tokens=max(0, len(old_pending) - 1),
                                         stage_ms={"snapshot_ms": snapshot_ms,
                                                   "combined_draft_verify_ms": combined_ms,
                                                   "recovery_ms": recovery_ms})
                else:
                    policy.observe(budget, accepted, 0., combined_ms, len(actual))
                cycle_pending = False
    finally:
        if cycle_pending:
            # A pre-yield device error must not poison persistent policy state.
            # Snapshot restoration preserves container identity. The engine
            # discards request-local caches; no invented acceptance is recorded.
            if target_snapshot is not None:
                apply_snapshot(model_cache, target_snapshot)
            if draft_snapshot is not None:
                apply_snapshot(draft_cache, draft_snapshot)
            if hasattr(policy, "abort_cycle"):
                policy.abort_cycle()
