"""Experimental greedy baseline ticks over MLX-LM 0.32.2 BatchGenerator.

Importing this module is portable. Arrays and the real upstream generator are
created only on the owning worker thread. This is ordinary continuous batching,
not DS variable-length speculative verification or a new decode-first algorithm.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path


@dataclass(frozen=True)
class ContinuousConfig:
    max_active: int = 4
    tick_token_budget: int = 512
    prefill_chunk: int = 256
    max_context_tokens: int = 32768
    memory_budget_bytes: int | None = None
    temporary_reserve_bytes: int = 64 * 1024**2
    policy: str = "interactive"
    target_tick_ms: float = 30.0
    group_quota: int = 8
    aging_seconds: float = 2.0

    def __post_init__(self):
        for name in ("max_active", "tick_token_budget", "prefill_chunk", "max_context_tokens", "group_quota"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.tick_token_budget <= self.max_active:
            raise ValueError("tick_token_budget must leave space for prefill after all active decode rows")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes < 1:
            raise ValueError("memory_budget_bytes must be positive")
        if self.temporary_reserve_bytes < 0:
            raise ValueError("temporary_reserve_bytes must be nonnegative")
        if self.policy not in ("upstream", "interactive"):
            raise ValueError("continuous policy must be upstream or interactive")
        if not math.isfinite(self.target_tick_ms) or self.target_tick_ms <= 0:
            raise ValueError("target_tick_ms must be finite and positive")
        if not math.isfinite(self.aging_seconds) or self.aging_seconds <= 0:
            raise ValueError("aging_seconds must be finite and positive")


def batch_eligible(request, compare=False):
    """Non-greedy, comparison and speculation remain exclusive serial work."""
    return not compare and request.mode == "baseline" and (
        request.temperature == 0 and request.top_p == 1 and request.top_k == 0
        and request.presence_penalty == 0
    )


@dataclass
class Prepared:
    tokens: list[int]
    reserve_bytes: int
    identity: object
    detokenizer: object
    metadata: dict
    eos_tokens: set[int] = field(default_factory=set)
    kv_bytes: int = 0


class AdmissionBudget:
    """Reserve once above model resident memory; never add RSS to Metal.

    A full per-request KV + completion extraction reserve stays charged until
    removal/drain. The model's measured resident footprint is charged once.
    Actual allocator memory is an alternate lower bound, never an extra sum.
    """
    def __init__(self, config, resident_bytes, capacity_bytes):
        self.config = config
        self.resident_bytes = resident_bytes
        self.capacity_bytes = min(capacity_bytes, config.memory_budget_bytes or capacity_bytes)
        self.reservations = {}

    def check(self, key, prepared, max_tokens, actual_bytes=0, available_bytes=None, actual_cache_bytes=0):
        if key in self.reservations:
            raise RuntimeError("Duplicate request reservation")
        context = len(prepared.tokens) + max_tokens
        if len(self.reservations) >= self.config.max_active:
            return "active_capacity"
        if sum(c for _, c, _ in self.reservations.values()) + context > self.config.max_context_tokens:
            return "context_capacity"
        reserved = self.total_reserved(prepared)
        predicted = max(self.resident_bytes + reserved, actual_bytes + prepared.reserve_bytes)
        if predicted + self.config.temporary_reserve_bytes > self.capacity_bytes:
            return "memory_capacity"
        # OS available already excludes materialized cache arrays. Reserve only
        # the not-yet-materialized part plus new work, avoiding double counting.
        future = max(0, reserved - actual_cache_bytes)
        if available_bytes is not None and future + self.config.temporary_reserve_bytes > available_bytes * 0.65:
            return "available_memory"
        return None

    def reserve(self, key, prepared, max_tokens):
        self.reservations[key] = (prepared.reserve_bytes, len(prepared.tokens) + max_tokens, prepared.kv_bytes)

    def total_reserved(self, prepared=None):
        values = list(self.reservations.values())
        if prepared is not None:
            values.append((prepared.reserve_bytes, 0, prepared.kv_bytes))
        if not values:
            return 0
        kvs = [kv for _, _, kv in values]
        # Native batched KV is padded to the longest row. A short newcomer can
        # require another long row's storage; sum of independent contexts alone
        # would under-budget heterogeneous batches. Also cover final extraction.
        padding = 2 * (max(kvs) * len(kvs) - sum(kvs))
        return sum(reserve for reserve, _, _ in values) + padding

    def release(self, key):
        self.reservations.pop(key, None)


class MLXBatchSession:
    """One profile, one generator; all methods called on its owner thread."""
    def __init__(self, engine, profile, config):
        if version("mlx-lm") != "0.32.2":
            raise ValueError("Experimental continuous backend requires audited mlx-lm==0.32.2; use serial backend")
        import mlx.core as mx
        import psutil
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.models.cache import make_prompt_cache

        self.engine, self.profile, self.config = engine, profile, config
        self.mx = mx
        self.closed = False
        self.generator = None
        self.last_tick_ms = None
        self.chunk = min(config.prefill_chunk, config.tick_token_budget - config.max_active)
        self.load_started = time.perf_counter()
        self.model_warm = engine.profile == profile and engine.model is not None
        engine._check_snapshot_unchanged()
        engine._load(profile, False)
        if engine.draft_model is not None:
            engine.unload_draft()
        # Batched completion cache boundaries are owned by upstream Responses.
        # Prefix reuse is intentionally cold in this experimental backend; it
        # does not borrow serial snapshots or infer consumed token boundaries.
        engine.clear_cache()
        mx.clear_cache()
        self.loaded_at = time.perf_counter()
        self.metadata = engine.model_registry.get(profile)
        self.adapter = engine.model_registry.adapter(profile)
        self.model_config = json.loads((Path(engine.paths["target"]) / "config.json").read_text())
        caches = make_prompt_cache(engine.model)
        self.cache_capabilities = self.adapter.cache.capabilities(caches)
        if not all(callable(getattr(cache, "merge", None)) for cache in caches):
            raise ValueError("Continuous backend requires native batch cache operations; use serial backend")
        merged = [cache.merge([cache]) for cache in caches]
        if not all(all(callable(getattr(cache, name, None)) for name in ("filter", "extend", "extract", "prepare")) for cache in merged):
            raise ValueError("Merged cache lacks continuous batch operations; use serial backend")
        del merged
        del caches
        resident = mx.get_active_memory()
        capacity = resident + int(psutil.virtual_memory().available * 0.65)
        if engine.memory_limit_bytes is not None:
            capacity = min(capacity, engine.memory_limit_bytes)
        self.budget = AdmissionBudget(config, resident, capacity)
        self.generator = BatchGenerator(
            engine.model, stop_tokens=[[t] for t in engine.tokenizer.eos_token_ids],
            completion_batch_size=config.max_active, prefill_batch_size=1,
            prefill_step_size=self.chunk,
        )

    def prepare(self, request):
        self.engine._check_snapshot_unchanged()
        tokens = self.engine._tokens(request)
        if not tokens:
            raise ValueError("Tokenized prompt is empty")
        if len(tokens) + request.max_tokens > self.metadata.get("context_limit", 40960):
            raise ValueError("Prompt plus output exceeds model context limit; input is never truncated")
        if len(tokens) + request.max_tokens > self.config.max_context_tokens:
            raise MemoryError("Request exceeds continuous context capacity; reduce context or use serial backend")
        kv = self.adapter.memory_estimate(self.model_config, len(tokens) + request.max_tokens + 512)
        text_cfg = self.model_config.get("text_config", self.model_config)
        # Include native padded prefill logits/activations and final extraction
        # scratch; no target/draft snapshots are shared into this baseline path.
        max_chunk = min(self.config.prefill_chunk, self.config.tick_token_budget - self.config.max_active)
        logits = max_chunk * text_cfg["vocab_size"] * 4
        reserve = 2 * kv + logits
        detokenizer = copy.deepcopy(self.engine.tokenizer.detokenizer)
        detokenizer.reset()
        return Prepared(tokens, reserve, self.engine._identity(request, False), detokenizer,
                        self.engine.model_registry.public_metadata(self.metadata),
                        set(self.engine.tokenizer.eos_token_ids), kv)

    def resources(self):
        import psutil
        return {"actual_bytes": self.mx.get_active_memory(), "available_bytes": psutil.virtual_memory().available,
                "actual_cache_bytes": self.generator.prompt_cache_nbytes}

    def insert(self, prepared, request):
        # This is the total output limit, not a per-step vector verification cap.
        return self.generator.insert_segments([[prepared.tokens]], max_tokens=[request.max_tokens])[0]

    def tick(self):
        start = time.perf_counter()
        # Exactly one upstream step, including pure prefill ticks. Never call
        # next_generated(): it swallows pure prefill ticks until decode occurs.
        result = self.generator.next()
        self.last_tick_ms = (time.perf_counter() - start) * 1000
        if self.config.policy == "interactive":
            # Decrease chunks against observed whole-tick time. Decode itself
            # can dominate; no promise of a hard latency bound or zero blocking.
            if self.last_tick_ms > self.config.target_tick_ms:
                self.chunk = max(1, self.chunk // 2)
            elif self.last_tick_ms < self.config.target_tick_ms / 2:
                self.chunk = min(self.config.prefill_chunk, self.chunk + 16,
                                 self.config.tick_token_budget - self.config.max_active)
            self.generator.prefill_step_size = self.chunk
            self.generator._prompt_batch.prefill_step_size = self.chunk
        return result

    def remove(self, uid):
        self.generator.remove([uid], return_prompt_caches=False)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.generator is not None:
            generator = self.generator
            self.generator = None
            try:
                self.mx.synchronize(generator.stream)
            finally:
                generator.close()
        self.mx.clear_cache()
