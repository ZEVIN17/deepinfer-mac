"""Exact prefix LRU with byte accounting and branch-safe cache cloning."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Entry:
    identity: str
    tokens: tuple[int, ...]
    cache: list
    nbytes: int


def cache_bytes(cache):
    return sum(int(c.nbytes) for c in cache)


@dataclass
class CacheTimings:
    """Host time only; immutable MLX arrays are shared, not copied to the CPU.

    ``write_ms`` includes snapshot cloning, insertion and any complete eviction.
    ``eviction_selection_ms`` is only the ordering-pop subset of that interval;
    native trie removal and entry release are not timed separately.
    """

    lookups: int = 0
    hits: int = 0
    writes: int = 0
    rejected_writes: int = 0
    lookup_ms: float = 0.0
    clone_ms: float = 0.0
    write_ms: float = 0.0
    eviction_selection_ms: float = 0.0


class NativePromptCacheStore:
    """Use MLX-LM's trie/trim implementation without inventing hybrid rollback.

    ``get`` always leaves an input token to evaluate. A recurrent/ArraysCache
    exact hit at the full request cannot be trimmed, so a shorter exact snapshot
    is requested instead. Checkpoints are ordinary inserts; upstream preserves
    them for nontrimmable caches and removes redundant trimmable prefixes.

    Imports remain lazy so portable CLI/API operations do not require MLX.
    This is an independent infrastructure optimization, not a DS mechanism.
    """

    def __init__(self, budget_bytes=512 * 1024**2, *, _runtime=None):
        if budget_bytes < 0:
            raise ValueError("Cache budget cannot be negative")
        self.budget_bytes = budget_bytes
        self.timings = CacheTimings()
        self.evictions = 0
        self._runtime = _runtime
        self._store = None

    def _ensure_store(self):
        if self._store is not None:
            return self._store
        if self._runtime is None:
            from mlx_lm.models import cache as runtime

            self._runtime = runtime
        self._store = self._runtime.LRUPromptCache(max_size=10000, max_bytes=self.budget_bytes)
        # Locked MLX-LM 0.32.2 has no public eviction counter. Count actual
        # capacity pops, not prefix pruning, at its single CacheOrder.pop hook.
        original_pop = self._store._lru.pop

        def tracked_pop():
            begin = time.perf_counter()
            result = original_pop()
            self.evictions += 1
            self.timings.eviction_selection_ms += (time.perf_counter() - begin) * 1000
            return result

        self._store._lru.pop = tracked_pop
        return self._store

    @property
    def nbytes(self):
        return self._store.nbytes if self._store is not None else 0

    def clear(self):
        self._store = None

    def get(self, identity, tokens):
        begin = time.perf_counter()
        self.timings.lookups += 1
        try:
            if self._store is None or len(tokens) < 2:
                return None, 0
            cache, rest = self._store.fetch_nearest_cache(identity, list(tokens))
            hit = len(tokens) - len(rest)
            if cache is not None and not rest:
                if self._runtime.can_trim_prompt_cache(cache):
                    removed = self._runtime.trim_prompt_cache(cache, 1)
                    if removed != 1:
                        raise RuntimeError("Native prompt cache did not rewind the final input token")
                    hit -= 1
                else:
                    cache, rest = self._store.fetch_nearest_cache(identity, list(tokens[:-1]))
                    hit = len(tokens) - 1 - len(rest)
            if cache is None or hit <= 0:
                return None, 0
            if hit >= len(tokens):
                raise RuntimeError("Cache lookup must leave one unconsumed input token")
            self.timings.hits += 1
            return cache, hit
        finally:
            # Includes upstream deepcopy and trim; those are not separately
            # observable without changing upstream execution.
            self.timings.lookup_ms += (time.perf_counter() - begin) * 1000

    def put(self, identity, tokens, cache, *, prefill_ms=None, expected_reuses=1.0, cache_type="assistant"):
        """Snapshot an already consumed prefix within the logical byte budget.

        With measured prefill cost, reject an expected saving smaller than the
        observed mean clone cost. With no estimate, retain upstream's byte-only
        admission. This optional host-cost heuristic is explicit and testable.
        """
        if expected_reuses < 0 or (prefill_ms is not None and prefill_ms < 0):
            raise ValueError("Cache admission estimates must be nonnegative")
        begin = time.perf_counter()
        try:
            nbytes = cache_bytes(cache)
            clone_estimate = self.timings.clone_ms / self.timings.writes if self.timings.writes else 0
            if (
                not tokens
                or not self.budget_bytes
                or nbytes > self.budget_bytes
                or (prefill_ms is not None and expected_reuses * prefill_ms <= clone_estimate)
            ):
                self.timings.rejected_writes += 1
                return False
            store = self._ensure_store()
            clone_begin = time.perf_counter()
            snapshot = copy.deepcopy(cache)
            self.timings.clone_ms += (time.perf_counter() - clone_begin) * 1000
            store.insert_cache(identity, list(tokens), snapshot, cache_type=cache_type)
            self.timings.writes += 1
            return True
        finally:
            self.timings.write_ms += (time.perf_counter() - begin) * 1000


class PromptCacheStore:
    def __init__(self, budget_bytes=512 * 1024**2):
        if budget_bytes < 0:
            raise ValueError("Cache budget cannot be negative")
        self.budget_bytes = budget_bytes
        self.entries = OrderedDict()
        self.evictions = 0

    @property
    def nbytes(self):
        return sum(e.nbytes for e in self.entries.values())

    def clear(self):
        self.entries.clear()

    def get(self, identity, tokens):
        """Reuse only a fully matching stored prefix, always leave one input token.

        Caches are snapshots at the end of prefill, not generated continuations.
        No trimming or tensor slicing of unknown custom cache types is attempted.
        """
        selected = None
        for key, entry in self.entries.items():
            if (
                entry.identity == identity
                and len(entry.tokens) < len(tokens)
                and tuple(tokens[: len(entry.tokens)]) == entry.tokens
                and (selected is None or len(entry.tokens) > len(selected[1].tokens))
            ):
                selected = key, entry
        if selected is None:
            return None, 0
        key, entry = selected
        self.entries.move_to_end(key)
        # MLX arrays are immutable; deepcopy of cache containers protects offsets
        # and subsequent update/assignment on branches (same as upstream server).
        return copy.deepcopy(entry.cache), len(entry.tokens)

    def put(self, identity, tokens, cache, *, prefill_ms=None, expected_reuses=1.0, cache_type="assistant"):
        # Compatibility with the measured-admission native facade. The historic
        # exact-only control intentionally retains its original byte-only rule.
        if not tokens or self.budget_bytes == 0:
            return
        nbytes = cache_bytes(cache)
        if nbytes > self.budget_bytes:
            return
        key = identity, tuple(tokens)
        self.entries.pop(key, None)
        while self.entries and self.nbytes + nbytes > self.budget_bytes:
            self.entries.popitem(last=False)
            self.evictions += 1
        self.entries[key] = Entry(identity, tuple(tokens), copy.deepcopy(cache), nbytes)
