"""Portable tests of the native adapter's ownership and input-tail contract."""

import copy
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from deepinfer_mac.cache import NativePromptCacheStore


class State:
    def __init__(self, offset, trimmable=True, nbytes=8):
        self.offset, self.trimmable, self.nbytes = offset, trimmable, nbytes


class NativeDouble:
    """Only the documented nearest-prefix API, not MLX's trie implementation."""

    def __init__(self, max_size, max_bytes):
        self.entries = OrderedDict()
        self.max_bytes = max_bytes
        self._lru = SimpleNamespace(pop=self.pop)

    @property
    def nbytes(self):
        return sum(sum(c.nbytes for c in value) for value in self.entries.values())

    def pop(self):
        return self.entries.popitem(last=False)

    def insert_cache(self, identity, tokens, cache, *, cache_type="assistant"):
        self.entries[identity, tuple(tokens)] = cache
        while self.nbytes > self.max_bytes:
            self._lru.pop()

    def fetch_nearest_cache(self, identity, tokens):
        candidates = []
        for (key, prefix), cache in self.entries.items():
            if key == identity and tuple(tokens[: len(prefix)]) == prefix:
                candidates.append((len(prefix), cache))
        if not candidates:
            return None, tokens
        hit, cache = max(candidates, key=lambda x: x[0])
        return copy.deepcopy(cache), tokens[hit:]


def can_trim(cache):
    return all(c.trimmable for c in cache)


def trim(cache, count):
    assert can_trim(cache)
    for c in cache:
        c.offset -= count
    return count


RUNTIME = SimpleNamespace(LRUPromptCache=NativeDouble, can_trim_prompt_cache=can_trim, trim_prompt_cache=trim)


def test_native_exact_match_leaves_last_input_and_does_not_mutate_snapshot():
    store = NativePromptCacheStore(_runtime=RUNTIME)
    original = [State(3)]
    assert store.put("a", [1, 2, 3], original)
    original[0].offset = 100
    cache, hit = store.get("a", [1, 2, 3])
    assert hit == 2 and cache[0].offset == 2
    assert store.get("a", [1, 2, 3, 4])[0][0].offset == 3
    assert store.get("b", [1, 2, 3, 4]) == (None, 0)
    assert store.get("a", [1]) == (None, 0)


def test_hybrid_exact_match_replays_suffix_from_earlier_exact_snapshot():
    store = NativePromptCacheStore(_runtime=RUNTIME)
    store.put("a", [1], [State(1, False)])
    store.put("a", [1, 2, 3], [State(3, False)])
    cache, hit = store.get("a", [1, 2, 3])
    assert hit == 1 and cache[0].offset == 1
    assert store.get("a", [1, 2, 9])[1] == 1
    assert store.get("a", [1, 2, 3, 4])[1] == 3
    store.clear()
    store.put("a", [1, 2, 3], [State(3, False)])
    assert store.get("a", [1, 2, 3]) == (None, 0)


def test_native_budget_counts_actual_evictions_and_preserves_branches():
    store = NativePromptCacheStore(16, _runtime=RUNTIME)
    for i in range(3):
        store.put("a", [i], [State(1, False)])
    assert store.nbytes == 16 and store.evictions == 1
    assert store.get("a", [0, 9]) == (None, 0)
    branch, _ = store.get("a", [1, 9])
    branch[0].offset = 99
    assert store.get("a", [1, 8])[0][0].offset == 1
    assert store.timings.eviction_selection_ms >= 0


def test_admission_and_telemetry_are_explicit_host_estimates():
    store = NativePromptCacheStore(8, _runtime=RUNTIME)
    assert not store.put("a", [], [State(0)])
    assert not store.put("a", [1], [State(1, nbytes=9)])
    assert not store.put("a", [1], [State(1)], prefill_ms=10, expected_reuses=0)
    assert store.put("a", [1], [State(1)], prefill_ms=10)
    assert store.get("a", [1, 2])[1] == 1
    assert store.timings.writes == 1 and store.timings.rejected_writes == 3
    assert store.timings.lookups == 1 and store.timings.hits == 1
    assert store.timings.lookup_ms >= 0 and store.timings.write_ms >= store.timings.clone_ms
    with pytest.raises(ValueError):
        store.put("a", [1], [State(1)], prefill_ms=-1)
    with pytest.raises(ValueError):
        NativePromptCacheStore(-1)
    disabled = NativePromptCacheStore(0, _runtime=RUNTIME)
    assert not disabled.put("a", [1], [State(1)]) and disabled._store is None
