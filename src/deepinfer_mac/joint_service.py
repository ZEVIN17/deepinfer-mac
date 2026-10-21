"""Explicit experimental joint service boundary; portable until owner creation."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from .continuous import AdmissionBudget, Prepared, batch_eligible
from .cycle_cost import CycleCostTable, CycleShape
from .ds.joint_budget import CostKey, JointBudget, RequestKey
from .varlen_verify import execution_identity


@dataclass(frozen=True)
class JointServiceConfig:
    profile: str
    depth: int = 3
    strategy: str = "bucket"
    curve_file: str | None = None
    calibration_identity: str = "uncalibrated"
    draft_budget: str = "full"

    def __post_init__(self):
        if not self.profile or type(self.depth) is not int or not 1 <= self.depth <= 16:
            raise ValueError("Joint service requires a profile and depth between 1 and 16")
        if self.strategy not in ("bucket", "padded_control", "shared_cap"):
            raise ValueError("Unknown joint verification strategy")
        if self.draft_budget not in ("full", "allocated"):
            raise ValueError("Unknown draft budget policy")
        if not self.calibration_identity:
            raise ValueError("Calibration identity must be explicit")

    def validate_request(self, request, compare=False):
        if request.profile != self.profile or not batch_eligible(request, compare):
            raise ValueError("Experimental joint service accepts only its configured profile and greedy baseline requests")


def public_round_metrics(result):
    """Aggregate allowlist: never expose another row's outputs or observations."""
    names = ("requested_q", "effective_verified_q", "executed_verify_q", "padded_q",
             "executed_replay_q", "model_submissions", "gpu_kernel_count", "full_cycle_ms",
             "draft_cost_included", "strategy", "executed_total_q", "scheduler_ms",
             "allocator", "draft_source", "confidence_source", "cost_curve_scope",
             "draft_budget", "executed_draft_q", "handoff_ms", "service_cycle_ms", "cost_observation_ms")
    return {name: result[name] for name in names if name in result}


def runtime_source_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("joint_service.py", "varlen_verify.py", "cycle_cost.py", "native_handoff.py", "ds/joint_budget.py", "ds/mtp_varlen.py")}


def read_cost_curves(path, depth):
    """Consume source-bound development samples; compute medians, never invent cost."""
    payload = json.loads(Path(path).read_text())
    if (payload.get("scope") != "complete_cycle" or payload.get("partition") != "development"
            or payload.get("depth") != depth or payload.get("source_sha256") != runtime_source_hashes()
            or payload.get("mlx_lm_version") != version("mlx-lm")):
        raise ValueError("Cost file requires current-source complete-cycle development observations")
    curves, seen = [], set()
    for item in payload.get("curves", []):
        key = CostKey(**item["key"])
        if key in seen or key.batch_size < 1:
            raise ValueError("Duplicate or empty cost key")
        seen.add(key)
        points = {}
        for sample in item["measurements"]:
            q, elapsed = sample["total_q"], sample["full_cycle_ms"]
            if (type(q) is not int or not key.batch_size <= q <= key.batch_size * (depth + 1)
                    or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0):
                raise ValueError("Invalid complete-cycle measurement")
            points.setdefault(q, []).append(elapsed)
        if set(points) != set(range(key.batch_size, key.batch_size * (depth + 1) + 1)):
            raise ValueError("Cost measurements must cover every legal total budget")
        curves.append((key, [(q, statistics.median(values)) for q, values in sorted(points.items())]))
    if not curves:
        raise ValueError("Cost file contains no measured curves")
    return curves


class MLXJointBatchSession:
    """Real two-model transactional ticks adapted to the bounded worker protocol.

    Prefill is synchronous, chunked for memory only; it is not a preemptible
    upstream prefill queue. Token responses exist only after a whole verified
    round commits. Both caches exclude the last unconsumed tail token.
    """
    def __init__(self, engine, profile, config, *, joint):
        if version("mlx-lm") != "0.32.2":
            raise ValueError("Joint service requires audited mlx-lm==0.32.2")
        if profile != joint.profile:
            raise ValueError("Joint profile mismatch")
        metadata = engine.model_registry.get(profile)
        if not metadata.get("custom_model") or not metadata.get("draft"):
            raise ValueError("Joint service requires a custom exact-identity ModelSpec with external draft")
        import mlx.core as mx
        import psutil
        from mlx_lm.models.cache import make_prompt_cache

        from .varlen_verify import JointSpeculativeSession
        self.engine, self.profile, self.config, self.joint = engine, profile, config, joint
        self.mx, self.make_cache = mx, make_prompt_cache
        self.rows, self.next_uid, self.closed = {}, 0, False
        self.native_generator, self.native_uids = None, {}
        self.handoff_ms = 0.0
        self.cycle_costs = CycleCostTable()
        self.load_started = time.perf_counter()
        self.model_warm = False
        engine._check_snapshot_unchanged()
        # Re-enter the combined preload guard with neither model resident.
        # This avoids the serial engine warm-target incremental-load boundary.
        engine.close()
        engine._load(profile, True)
        engine.clear_cache()
        self.loaded_at = time.perf_counter()
        self.adapter = engine.model_registry.adapter(profile)
        self.metadata = engine.model_registry.get(profile)
        self.configs = [json.loads((Path(engine.paths[k]) / "config.json").read_text()) for k in ("target", "draft")]
        for model in (engine.model, engine.draft_model):
            caches = make_prompt_cache(model)
            if not caches or any(type(c).__name__ != "KVCache" for c in caches):
                raise ValueError("Experimental joint service supports audited dense KVCache only")
        identity = engine.resolved_identity
        if not identity or identity["target"] == identity["draft"]:
            raise ValueError("Joint service requires distinct audited target and draft contents")
        self.target_identity, self.draft_identity = identity["target"], identity["draft"]
        self.quantization = json.dumps([c.get("quantization", {}) for c in self.configs], sort_keys=True)
        self.joint_budget = JointBudget(joint.depth, self.target_identity)
        if joint.curve_file:
            for key, costs in read_cost_curves(joint.curve_file, joint.depth):
                if (key.model_identity != self.target_identity or key.draft_identity != self.draft_identity
                        or key.quantization != self.quantization or key.execution_strategy != execution_identity(joint.strategy, draft_budget=joint.draft_budget)
                        or key.calibration_identity != joint.calibration_identity):
                    raise ValueError("Curve identity differs from loaded pair")
                self.joint_budget.set_curve(key, costs)
        self.session = JointSpeculativeSession(engine.model, engine.draft_model, self.joint_budget,
                                              self.cost_key(1, 0), draft_identity=self.draft_identity,
                                              calibration_identity=joint.calibration_identity, allocator="mlx",
                                              draft_budget=joint.draft_budget)
        resident = mx.get_active_memory()
        capacity = resident + int(psutil.virtual_memory().available * 0.65)
        if engine.memory_limit_bytes is not None:
            capacity = min(capacity, engine.memory_limit_bytes)
        self.budget = AdmissionBudget(config, resident, capacity)
        self.last_report = None

    def cost_key(self, count, context):
        return CostKey(self.target_identity, self.draft_identity, self.quantization, count, context,
                       execution_identity(self.joint.strategy, draft_budget=self.joint.draft_budget),
                       self.joint.calibration_identity)

    def prepare(self, request):
        self.joint.validate_request(request)
        self.engine._check_snapshot_unchanged()
        tokens = self.engine._tokens(request)
        context = len(tokens) + request.max_tokens + self.joint.depth + 1
        limits = [self.metadata["context_limit"], self.config.max_context_tokens]
        limits.extend(c.get("max_position_embeddings", context) for c in self.configs)
        if not tokens or context > min(limits):
            raise ValueError("Joint prompt/output/draft scratch exceeds context limit; input is never truncated")
        kv = sum(self.adapter.memory_estimate(c, context + 512) for c in self.configs)
        # Originals, outer snapshots, verifier snapshots, copied and merged
        # verification states, replay/catch-up and lazy intermediates. Padding
        # is separately charged by AdmissionBudget's longest-row allowance.
        logits = max(self.config.prefill_chunk, self.joint.depth + 1) * max(c["vocab_size"] for c in self.configs) * 4
        reserve = 8 * kv + 2 * logits
        detokenizer = copy.deepcopy(self.engine.tokenizer.detokenizer)
        detokenizer.reset()
        return Prepared(tokens, reserve, self.engine._identity(request, True), detokenizer,
                        self.engine.model_registry.public_metadata(self.metadata),
                        set(self.engine.tokenizer.eos_token_ids), 4 * kv)

    def resources(self):
        import psutil
        return {"actual_bytes": self.mx.get_active_memory(), "available_bytes": psutil.virtual_memory().available,
                "actual_cache_bytes": 0}

    def insert_request(self, prepared, request, job):
        return self.insert(prepared, request, key=RequestKey(job.id, job.slot_generation), cancel=job.cancel.is_set)

    def insert(self, prepared, request, *, key=None, cancel=None):
        self.next_uid += 1
        uid = self.next_uid
        if self.native_generator is not None:
            if cancel is not None and cancel():
                from .server.continuous_worker import RequestInsertionCancelled
                raise RequestInsertionCancelled("Native insertion cancelled before mutation")
            native_uid = self.native_generator.insert([prepared.tokens], max_tokens=[request.max_tokens])[0]
            self.native_uids[native_uid] = uid
            self.rows[uid] = SimpleNamespace(target=[], draft=[], tail=prepared.tokens[-1],
                                             remaining=request.max_tokens, eos=prepared.eos_tokens,
                                             consumed=len(prepared.tokens) - 1, key=key or RequestKey(str(uid), 1))
            return uid
        target, draft = self.make_cache(self.engine.model), self.make_cache(self.engine.draft_model)
        # Allocated only after the worker checks admission. No output is emitted
        # from this prefill; final prompt token remains pending in both models.
        prefix = prepared.tokens[:-1]
        for start in range(0, len(prefix), self.config.prefill_chunk):
            if cancel is not None and cancel():
                from .server.continuous_worker import RequestInsertionCancelled
                raise RequestInsertionCancelled("Joint prefill cancelled; local caches discarded")
            chunk = self.mx.array([prefix[start:start + self.config.prefill_chunk]], dtype=self.mx.uint32)
            self.engine.model(chunk, cache=target)
            self.engine.draft_model(chunk, cache=draft)
            self.mx.eval([c.state for c in target + draft])
        self.rows[uid] = SimpleNamespace(target=target, draft=draft, tail=prepared.tokens[-1],
                                         remaining=request.max_tokens, eos=prepared.eos_tokens,
                                         consumed=len(prefix), key=key or RequestKey(str(uid), 1))
        return uid

    def switch_to_native(self):
        """Owner-thread boundary after a committed round, before the next tick.

        One-way within this epoch. Failed insertion preserves target/draft state;
        successful transfer drops draft caches and speculative history. Resident
        draft weights remain charged until normal engine/session cleanup.
        """
        from .native_handoff import CommittedPrefix, transfer_to_native
        if self.closed:
            raise RuntimeError("Session is closed")
        if self.native_generator is not None:
            return
        started = time.perf_counter()
        generator, mapping = transfer_to_native(
            self.engine.model,
            [CommittedPrefix(uid, r.target, r.tail, r.remaining) for uid, r in self.rows.items()],
            stop_tokens=[[t] for t in self.engine.tokenizer.eos_token_ids],
            max_active=self.config.max_active, prefill_chunk=self.config.prefill_chunk)
        self.native_generator, self.native_uids = generator, mapping
        for row in self.rows.values():
            row.target, row.draft = [], []
        self.session = None
        self.joint_budget.previous.clear()
        self.joint_budget.latest.clear()
        self.handoff_ms = (time.perf_counter() - started) * 1000

    def tick(self):
        if self.native_generator is not None:
            started = time.perf_counter()
            prompts, outputs = self.native_generator.next()
            for response in prompts + outputs:
                response.uid = self.native_uids[response.uid]
            for response in outputs:
                row = self.rows[response.uid]
                row.remaining -= 1
                row.tail = response.token
                row.consumed += 1
                if response.finish_reason:
                    self.rows.pop(response.uid)
                    native_uid = next(k for k, v in self.native_uids.items() if v == response.uid)
                    del self.native_uids[native_uid]
            self.last_report = {"strategy": "native_batch", "draft_cost_included": False,
                                "full_cycle_ms": (time.perf_counter() - started) * 1000,
                                "handoff_ms": self.handoff_ms}
            return prompts, outputs
        from .varlen_verify import VerifyRow
        cycle_started = time.perf_counter()
        ids = list(self.rows)
        if not ids:
            return [], []
        rows = [self.rows[uid] for uid in ids]
        contexts = tuple(r.consumed for r in rows)
        cache_classes = tuple(f"{type(c).__module__}.{type(c).__name__}" for c in rows[0].target + rows[0].draft)
        key = self.cost_key(len(rows), max(r.consumed for r in rows))
        result = self.session.tick([VerifyRow(r.key, r.target, r.tail, (), r.remaining, r.eos) for r in rows],
                                   [r.draft for r in rows], strategy=self.joint.strategy, cost_key=key)
        self.last_report = public_round_metrics(result)
        self.last_report["cost_curve_available"] = key in self.joint_budget.schedulers
        responses = []
        for uid, row, output in zip(ids, rows, result["outputs"]):
            for i, token in enumerate(output):
                row.remaining -= 1
                reason = "stop" if token in row.eos else "length" if row.remaining == 0 else None
                responses.append(SimpleNamespace(uid=uid, token=token, finish_reason=reason,
                                                 all_tokens=None, prompt_cache=None))
                if reason:
                    if i != len(output) - 1:
                        raise RuntimeError("Verifier emitted beyond terminal token")
                    self.rows.pop(uid)
            row.consumed += len(output)
            row.tail = output[-1]
        service_cycle_ms = (time.perf_counter() - cycle_started) * 1000
        observation_started = time.perf_counter()
        shape = CycleShape(key, cache_classes, contexts, tuple(
            (call["phase"], call["batch"], call["sequence_length"]) for call in result["model_calls"]))
        self.cycle_costs.observe(shape, service_cycle_ms, sum(map(len, result["outputs"])))
        self.last_report["service_cycle_ms"] = service_cycle_ms
        self.last_report["cost_observation_ms"] = (time.perf_counter() - observation_started) * 1000
        return [], responses

    def remove(self, uid):
        if self.native_generator is not None:
            native_uid = next((k for k, v in self.native_uids.items() if v == uid), None)
            if native_uid is not None:
                self.native_generator.remove([native_uid], return_prompt_caches=False)
                del self.native_uids[native_uid]
        self.rows.pop(uid, None)

    def close(self):
        if not self.closed:
            self.closed = True
            self.mx.synchronize()
            if self.native_generator is not None:
                self.native_generator.close()
                self.native_generator = None
                self.native_uids.clear()
            self.rows.clear()
            self.session = None
            self.mx.clear_cache()
