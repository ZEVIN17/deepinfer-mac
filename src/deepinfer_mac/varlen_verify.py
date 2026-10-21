"""Experimental greedy verification with actual sequence-length microbatches.

A mechanism executor, not a production DraftProvider or trained confidence head.
No BatchGenerator vector-cap API is assumed. Dense cache merge follows locked
MLX-LM 0.32.2 models/cache.py KVCache.merge / BatchKVCache. Hybrid uses isolated
single-row exact replay until its batched state protocol is independently proved.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

from .adapters import CacheAdapter
from .ds.joint_budget import RequestKey


class StageTimer:
    """Optional non-overlapping host wall intervals at existing synchronization boundaries.

    No extra device synchronization: a forward interval includes lazy graph building,
    argmax and its existing eval. Host-only labels do not imply GPU kernel timing.
    """
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.stages = {}
        self.last = time.perf_counter() if enabled else None

    def mark(self, name):
        if self.enabled:
            now = time.perf_counter()
            self.stages[name] = self.stages.get(name, 0.0) + (now - self.last) * 1000
            self.last = now


def execution_identity(strategy, profiling=False, *, draft_execution="serial", recovery="exact_replay"):
    if draft_execution not in ("serial", "batched") or recovery not in ("exact_replay", "dense_trim"):
        raise ValueError("Unknown draft execution or recovery policy")
    if draft_execution == "serial" and recovery == "exact_replay" and not profiling:
        return strategy  # Preserve the original unprofiled control's identity.
    profile = "wall_v1" if profiling else "off"
    return f"{strategy}|draft={draft_execution}|recovery={recovery}|profile={profile}"


def dense_cache(cache):
    """Exact audited class only; recurrent/rotating/quantized caches cannot opt in."""
    return bool(cache) and all(type(c).__module__ == "mlx_lm.models.cache"
                               and type(c).__name__ == "KVCache" for c in cache)


def commit_cache(destination, source):
    if len(destination) != len(source):
        raise ValueError("Cache layer count changed")
    for dst, src in zip(destination, source):
        dst.state = src.state


def trim_cache(cache, count):
    if count < 0:
        raise ValueError("Cannot trim a negative prefix")
    for layer in cache:
        if layer.trim(count) != count:
            raise ValueError("Cache shorter than consumed speculative prefix")


@dataclass
class VerifyRow:
    key: RequestKey
    cache: list
    tail: int
    proposals: tuple[int, ...]
    remaining: int
    eos: frozenset[int] = frozenset()

    def validate(self):
        if not isinstance(self.key, RequestKey):
            raise TypeError("Stable request identity required")
        if type(self.remaining) is not int or self.remaining < 1:
            raise ValueError("Remaining output budget must be a positive integer")
        if any(type(token) is not int or token < 0 for token in (self.tail, *self.proposals, *self.eos)):
            raise ValueError("Token IDs must be nonnegative integers")

    def __post_init__(self):
        self.validate()
        self.proposals = tuple(self.proposals)
        self.eos = frozenset(self.eos)


def verification_groups(lengths, strategy):
    if strategy not in ("bucket", "padded_control", "shared_cap"):
        raise ValueError("Unknown verification strategy")
    if any(type(q) is not int or q < 1 for q in lengths):
        raise ValueError("Each request requires a golden token")
    if not lengths:
        return []
    if strategy in ("padded_control", "shared_cap"):
        return [(max(lengths), tuple(range(len(lengths))))]
    return [(q, tuple(i for i, x in enumerate(lengths) if x == q)) for q in sorted(set(lengths))]


def greedy_result(tail, proposals, predicted, remaining, eos):
    """Return emitted tokens and conditional outcomes; suffixes remain censored."""
    if type(remaining) is not int or remaining < 1 or len(predicted) != len(proposals) + 1:
        raise ValueError("Invalid verification output/remaining budget")
    if any(type(token) is not int or token < 0 for token in (tail, *proposals, *predicted, *eos)):
        raise ValueError("Token IDs must be nonnegative integers")
    emitted, outcomes = [], [None] * len(proposals)
    for i, proposal in enumerate(proposals):
        token = predicted[i]
        outcomes[i] = token == proposal
        emitted.append(token)
        if token in eos or len(emitted) == remaining or token != proposal:
            return tuple(emitted), tuple(outcomes)
    if len(emitted) < remaining:
        emitted.append(predicted[-1])
    return tuple(emitted), tuple(outcomes)


class MLXVariableVerifier:
    """Transactional per-round target execution. Caller owns model/memory admission.

    All original row caches restore on cancellation or failure. Default recovery
    replays the exact consumed prefix even on full acceptance. The separate dense
    opt-in extracts and trims disposable verified state to that consumed prefix.
    Counts are model calls/tokens, not a fabricated GPU kernel count. Cost excludes
    proposal generation; caller must add measured draft/scheduler costs for cycles.
    """
    def __init__(self, model):
        self.model = model
        self.adapter = CacheAdapter()

    def verify(self, rows, lengths, *, strategy="bucket", cancel=None, profiling=False, recovery="exact_replay"):
        import mlx.core as mx
        start = time.perf_counter()
        timer = StageTimer(profiling)
        for row in rows:
            row.validate()
        if len(rows) != len(lengths) or len({r.key for r in rows}) != len(rows):
            raise ValueError("Unique rows and matching lengths required")
        if len({id(c) for r in rows for c in r.cache}) != sum(len(r.cache) for r in rows):
            raise ValueError("Rows must own distinct cache layer objects")
        groups = verification_groups(lengths, strategy)
        if any(r.remaining < 1 or q > len(r.proposals) + 1 for r, q in zip(rows, lengths)):
            raise ValueError("Insufficient proposals or output budget")
        if any(not self.adapter.capabilities(r.cache)["snapshot_replay"] for r in rows):
            raise ValueError("Unaudited cache cannot be restored exactly")
        if strategy == "shared_cap" and any(len(r.proposals) + 1 < max(lengths) for r in rows):
            raise ValueError("Shared cap requires actual proposals for every executed position")
        if recovery not in ("exact_replay", "dense_trim"):
            raise ValueError("Unknown recovery policy")
        if recovery == "dense_trim" and any(not dense_cache(r.cache) for r in rows):
            raise ValueError("dense_trim requires audited dense KVCache rows")
        snapshots = [self.adapter.snapshot(r.cache) for r in rows]
        calls, predictions, verified_caches = [], {}, {}
        effective = [max(lengths)] * len(rows) if strategy == "shared_cap" and rows else list(lengths)

        def check_cancel():
            if cancel is not None and cancel():
                raise InterruptedError("Verification cancelled")

        def forward(indices, q):
            timer.mark("verification_control_host")
            check_cancel()
            dense = all(dense_cache(rows[i].cache) for i in indices)
            if len(indices) > 1 and not dense:
                for i in indices:
                    forward((i,), q)
                return
            caches = [copy.deepcopy(rows[i].cache) for i in indices]
            # Use only audited KVCache.merge; no recurrent padded batch is executed.
            merged = [layer.merge([c[k] for c in caches]) for k, layer in enumerate(caches[0])] \
                if len(indices) > 1 else caches[0]
            inputs = []
            for i in indices:
                row = rows[i]
                tokens = [row.tail, *row.proposals[:effective[i] - 1]]
                inputs.append(tokens + [0] * (q - len(tokens)))
            timer.mark("verification_copy_merge_inputs_host")
            logits = self.model(mx.array(inputs, dtype=mx.uint32), cache=merged)
            chosen = mx.argmax(logits, axis=-1)
            mx.eval(chosen, [c.state for c in merged])
            timer.mark("verification_forward_argmax_eval")
            actual = chosen.tolist()
            timer.mark("verification_argmax_host")
            calls.append({"phase": "verify", "batch": len(indices), "sequence_length": q,
                          "tokens": len(indices) * q})
            for position, (i, prediction) in enumerate(zip(indices, actual)):
                predictions[i] = prediction[:effective[i]]
                if recovery == "dense_trim":
                    # Locked cache.py BatchKVCache.extract removes historical left
                    # padding. Trim only after extract: accepted k differs by row.
                    cache = [c.extract(position) for c in merged] if len(indices) > 1 else merged
                    verified_caches[i] = (cache, q)
            timer.mark("verification_extract_host")
            check_cancel()

        try:
            for q, indices in groups:
                forward(indices, q)
            outputs, outcomes = [], []
            for i, row in enumerate(rows):
                check_cancel()
                output, observed = greedy_result(row.tail, row.proposals[:effective[i] - 1],
                                                  predictions[i], row.remaining, row.eos)
                outputs.append(output)
                outcomes.append(observed + (None,) * (len(row.proposals) - len(observed)))
                # Cache stores all emitted tokens except final unconsumed tail.
                consumed = [row.tail, *output[:-1]]
                timer.mark("verification_result_host")
                if recovery == "dense_trim":
                    cache, executed_q = verified_caches[i]
                    # Includes causal right-padding in padded_control; every emitted
                    # prefix is valid before that padding. Final emitted tail is pending.
                    trim_cache(cache, executed_q - len(consumed))
                    mx.eval([c.state for c in cache])
                    commit_cache(row.cache, cache)
                    timer.mark("target_trim_commit_host")
                else:
                    self.adapter.restore(row.cache, snapshots[i])
                    timer.mark("target_restore_host")
                    self.model(mx.array([consumed], dtype=mx.uint32), cache=row.cache)
                    mx.eval([c.state for c in row.cache])
                    timer.mark("target_replay_eval")
                    calls.append({"phase": "replay", "batch": 1, "sequence_length": len(consumed),
                                  "tokens": len(consumed)})
            check_cancel()
        except BaseException:
            for row, snapshot in zip(rows, snapshots):
                self.adapter.restore(row.cache, snapshot)
            raise
        timer.mark("verification_accounting_host")
        executed = sum(c["tokens"] for c in calls if c["phase"] == "verify")
        return {"outputs": outputs, "outcomes": outcomes, "requested_q": sum(lengths),
                "effective_verified_q": sum(effective), "executed_verify_q": executed,
                "padded_q": executed - sum(effective), "executed_replay_q": sum(
                    c["tokens"] for c in calls if c["phase"] == "replay"),
                "model_calls": calls, "model_submissions": len(calls), "gpu_kernel_count": None,
                "verify_recovery_host_ms": (time.perf_counter() - start) * 1000,
                "full_cycle_ms": None, "draft_cost_included": False, "strategy": strategy, "recovery": recovery,
                "verification_stages_ms": timer.stages if profiling else None}


class JointSpeculativeSession:
    """Experimental real-draft tick, separate from the public continuous backend.

    Caller supplies already prefilled single-row target/draft caches with the
    same unconsumed tail. Real draft generation, target verification and exact
    draft catch-up are synchronized and included in full_cycle_ms. Using the
    target itself as draft is allowed ONLY as labelled mechanism_control.
    No pretrained performance or quality acceptance is implied by this interface.
    """
    def __init__(self, target, draft, budget, cost_key, *, draft_identity,
                 calibration_identity, mechanism_control=False, allocator="cpu", profiling=False,
                 draft_execution="serial", recovery="exact_replay"):
        from .ds.joint_budget import EmpiricalConfidence
        if allocator not in ("cpu", "mlx"):
            raise ValueError("Unknown allocator")
        if target is draft and not mechanism_control:
            raise ValueError("Same-target draft requires explicit mechanism_control")
        if (draft_identity != cost_key.draft_identity
                or calibration_identity != cost_key.calibration_identity):
            raise ValueError("Draft identity differs from cost curve")
        self.verifier = MLXVariableVerifier(target)
        self.draft, self.budget, self.cost_key = draft, budget, cost_key
        self.calibration_identity = calibration_identity
        self.mechanism_control, self.allocator = mechanism_control, allocator
        execution_identity("bucket", profiling, draft_execution=draft_execution, recovery=recovery)
        self.draft_execution, self.recovery = draft_execution, recovery
        self.profiling = profiling
        self.predictors = {}
        self.predictor_type = EmpiricalConfidence

    def tick(self, rows, draft_caches, *, strategy="bucket", forced_lengths=None, cancel=None, cost_key=None):
        import mlx.core as mx

        from .ds.joint_budget import allocate_mlx, remap_keys
        start = time.perf_counter()
        timer = StageTimer(self.profiling)
        for row in rows:
            row.validate()
        key = cost_key or self.cost_key
        if (key.model_identity != self.budget.model_identity
                or key.draft_identity != self.cost_key.draft_identity
                or key.quantization != self.cost_key.quantization
                or key.calibration_identity != self.calibration_identity
                or key.execution_strategy != execution_identity(
                    strategy, self.profiling, draft_execution=self.draft_execution, recovery=self.recovery) or key.context_bucket < 0):
            raise ValueError("Cost key identity/strategy mismatch")
        if len(rows) != len(draft_caches) or len(rows) != key.batch_size:
            raise ValueError("Draft caches/cost key must match active rows")
        if len({r.key for r in rows}) != len(rows):
            raise ValueError("Duplicate request identity")
        layers = [c for r in rows for c in r.cache] + [c for cache in draft_caches for c in cache]
        if len({id(c) for c in layers}) != len(layers):
            raise ValueError("Target and draft rows must own distinct cache layer objects")
        if self.draft_execution == "batched" and any(not dense_cache(c) for c in [*[r.cache for r in rows], *draft_caches]):
            raise ValueError("Batched draft requires audited dense target and draft KVCache rows")
        if self.recovery == "dense_trim" and any(not dense_cache(c) for c in [*[r.cache for r in rows], *draft_caches]):
            raise ValueError("dense_trim requires audited dense target and draft KVCache rows")
        adapter = self.verifier.adapter
        target_snapshots = [adapter.snapshot(r.cache) for r in rows]
        draft_snapshots = [adapter.snapshot(c) for c in draft_caches]
        keys = [row.key for row in rows]
        lengths = self.budget.choose(keys, key) if forced_lengths is None else tuple(forced_lengths)
        verification_groups(lengths, strategy)
        if len(lengths) != len(rows) or any(q > self.budget.depth + 1 for q in lengths):
            raise ValueError("Lengths exceed configured draft depth")
        draft_calls, catchup_calls = [], []
        working = []
        try:
            if forced_lengths is None and self.allocator == "mlx":
                previous = list(self.budget.latest)
                device_lengths = allocate_mlx(list(self.budget.latest.values()),
                                               remap_keys(previous, keys), sum(lengths), self.budget.depth)
                mx.eval(device_lengths)
                lengths = tuple(device_lengths.tolist())
            scheduler_ms = (time.perf_counter() - start) * 1000
            timer.mark("snapshot_scheduler_allocation_host")
            # Keep the historical serial/replay control unchanged (no new copy
            # overhead). Opt-in paths use private state until success; all paths
            # retain independent snapshots for complete rollback on failure.
            proposal_rows = [[] for _ in rows]
            draft_work = draft_caches if (self.draft_execution == "serial" and self.recovery == "exact_replay") \
                else copy.deepcopy(draft_caches)
            if self.draft_execution == "batched" and rows:
                # cache.py KVCache.merge right-aligns heterogeneous history and
                # preserves per-row RoPE offsets. One actual [B,1] call per step.
                merged = [layer.merge([c[k] for c in draft_work])
                          for k, layer in enumerate(draft_work[0])]
                current = [row.tail for row in rows]
                for _ in range(self.budget.depth):
                    if cancel is not None and cancel():
                        raise InterruptedError("Draft cancelled")
                    timer.mark("draft_control_host")
                    logits = self.draft(mx.array(current, dtype=mx.uint32)[:, None], cache=merged)
                    chosen = mx.argmax(logits[:, -1, :], axis=-1)
                    mx.eval(chosen, [c.state for c in merged])
                    timer.mark("draft_forward_argmax_eval")
                    current = chosen.tolist()
                    timer.mark("draft_argmax_host")
                    for proposals, token in zip(proposal_rows, current):
                        proposals.append(token)
                    draft_calls.append({"phase": "draft", "batch": len(rows), "sequence_length": 1,
                                        "tokens": len(rows)})
                # At depth zero an empty cache has no tensors to extract.
                if self.budget.depth:
                    draft_work = [[c.extract(i) for c in merged] for i in range(len(rows))]
                timer.mark("draft_extract_host")
            else:
                for row, cache, proposals in zip(rows, draft_work, proposal_rows):
                    current = row.tail
                    for _ in range(self.budget.depth):
                        if cancel is not None and cancel():
                            raise InterruptedError("Draft cancelled")
                        timer.mark("draft_control_host")
                        logits = self.draft(mx.array([[current]], dtype=mx.uint32), cache=cache)
                        chosen = mx.argmax(logits[:, -1, :], axis=-1)
                        mx.eval(chosen, [c.state for c in cache])
                        timer.mark("draft_forward_argmax_eval")
                        current = chosen.item()
                        timer.mark("draft_argmax_host")
                        proposals.append(current)
                        draft_calls.append({"phase": "draft", "batch": 1, "sequence_length": 1, "tokens": 1})
            working = [VerifyRow(row.key, row.cache, row.tail, tuple(proposals), row.remaining, row.eos)
                       for row, proposals in zip(rows, proposal_rows)]
            timer.mark("draft_control_host")
            result = self.verifier.verify(working, lengths, strategy=strategy, cancel=cancel,
                                          profiling=self.profiling, recovery=self.recovery)
            timer.mark("verification_total")
            for row, cache, private, snapshot, output in zip(
                    rows, draft_caches, draft_work, draft_snapshots, result["outputs"]):
                if cancel is not None and cancel():
                    raise InterruptedError("Catch-up cancelled")
                timer.mark("catchup_control_host")
                consumed = [row.tail, *output[:-1]]
                if self.recovery == "dense_trim":
                    k, depth = len(consumed), self.budget.depth
                    if k <= depth:
                        trim_cache(private, depth - k)
                        mx.eval([c.state for c in private])
                        timer.mark("draft_trim_host")
                    else:
                        # Full acceptance consumes the last proposal too. At D=0
                        # there is no proposal, so the pending old tail is consumed.
                        self.draft(mx.array([[consumed[-1]]], dtype=mx.uint32), cache=private)
                        mx.eval([c.state for c in private])
                        timer.mark("draft_catchup_eval")
                        catchup_calls.append({"phase": "draft_catchup", "batch": 1,
                                              "sequence_length": 1, "tokens": 1})
                    commit_cache(cache, private)
                else:
                    adapter.restore(cache, snapshot)
                    timer.mark("draft_restore_host")
                    self.draft(mx.array([consumed], dtype=mx.uint32), cache=cache)
                    mx.eval([c.state for c in cache])
                    timer.mark("draft_catchup_eval")
                    catchup_calls.append({"phase": "draft_catchup", "batch": 1,
                                          "sequence_length": len(consumed), "tokens": len(consumed)})
            if cancel is not None and cancel():
                raise InterruptedError("Round cancelled")
            # Commit statistical/history state only after all target/draft work succeeds.
            predictors = copy.deepcopy(self.predictors)
            observations = []
            for row, outcomes in zip(rows, result["outcomes"]):
                predictor = predictors.setdefault(row.key, self.predictor_type(
                    self.budget.depth, self.budget.model_identity, self.calibration_identity))
                predictor.observe(outcomes)
                observations.append((row.key, predictor.predict()))
            if rows:
                self.budget.commit(observations)
                self.predictors = {key: predictors[key] for key in keys}
        except BaseException:
            for row, snapshot in zip(rows, target_snapshots):
                adapter.restore(row.cache, snapshot)
            for cache, snapshot in zip(draft_caches, draft_snapshots):
                adapter.restore(cache, snapshot)
            raise
        timer.mark("statistics_commit_host")
        all_calls = [*draft_calls, *result["model_calls"], *catchup_calls]
        result.update({"model_calls": all_calls, "model_submissions": len(all_calls),
                       "executed_total_q": sum(call["tokens"] for call in all_calls),
                       "full_cycle_ms": (time.perf_counter() - start) * 1000,
                       "draft_cost_included": True, "scheduler_ms": scheduler_ms,
                       "allocator": self.allocator, "seqlens_q": lengths,
                       "draft_source": "mechanism_control" if self.mechanism_control else "external_model",
                       "confidence_source": "empirical", "cost_curve_scope": "whole_cycle_once",
                       "profiling": self.profiling, "draft_execution": self.draft_execution,
                       "execution_identity": key.execution_strategy,
                       "cycle_stages_ms": timer.stages if self.profiling else None})
        if self.profiling:
            result["cycle_profile_residual_ms"] = result["full_cycle_ms"] - sum(timer.stages.values())
        return result
