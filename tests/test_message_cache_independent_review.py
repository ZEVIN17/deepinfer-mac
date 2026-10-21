"""Independent CPU checks; no inference timings or MLX numerical claims."""

import ast
import copy
import importlib.metadata
import threading
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from test_engine_message_checkpoints import TOKENS, LayerState, model, setup

from deepinfer_mac.cache import NativePromptCacheStore


def installed_cache_runtime():
    """Compile installed unchanged trie/LRU AST without importing Metal runtime."""
    try:
        distribution = importlib.metadata.distribution("mlx-lm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Installed MLX-LM source required for upstream CPU compatibility check")
    path = distribution.locate_file("mlx_lm/models/cache.py")
    tree = ast.parse(path.read_text())
    names = {"PromptTrieResult", "PromptTrie", "LRUPromptCache", "can_trim_prompt_cache", "trim_prompt_cache"}
    selected = [node for node in tree.body if getattr(node, "name", "") in names]
    assert len(selected) == len(names)
    namespace = {"copy": copy, "deque": deque, "dataclass": dataclass, "Any": Any, "List": list, "Optional": Optional}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 -- installed pinned local source, no external payload
    return SimpleNamespace(**{name: namespace[name] for name in (
        "LRUPromptCache", "can_trim_prompt_cache", "trim_prompt_cache"
    )})


class PrefixState(LayerState):
    def is_trimmable(self):
        return self.trimmable


def test_installed_native_priority_accounting_and_branch_ownership_without_metal():
    store = NativePromptCacheStore(10000, _runtime=installed_cache_runtime())
    for prefix, role in [([1], "system"), ([1, 2], "user"), ([1, 2, 3], "assistant")]:
        store.put("model", prefix, [PrefixState(prefix)], cache_type=role)
    stats = store._store.stats_by_type()
    assert {key: value["n_sequences"] for key, value in stats.items()} == {
        "assistant": 1, "user": 1, "system": 1
    }
    assert store.nbytes == sum(value["n_bytes"] for value in stats.values())
    restored, hit = store.get("model", [1, 2, 9, 10])
    assert hit == 2 and restored[0].tokens == [1, 2]
    restored[0].tokens.append(99)
    assert store.get("model", [1, 2, 8])[0][0].tokens == [1, 2]


def test_installed_native_full_hit_hybrid_fallback_and_trimmable_tail_without_metal():
    runtime = installed_cache_runtime()
    store = NativePromptCacheStore(10000, _runtime=runtime)
    store.put("model", [1, 2], [PrefixState([1, 2])], cache_type="system")
    store.put("model", [1, 2, 3, 4], [PrefixState([1, 2, 3, 4])], cache_type="user")
    cache, hit = store.get("model", [1, 2, 3, 4])
    assert hit == 2 and cache[0].tokens == [1, 2]
    trimmed = NativePromptCacheStore(10000, _runtime=runtime)
    trimmed.put("model", [1, 2, 3], [PrefixState([1, 2, 3], trimmable=True)], cache_type="user")
    cache, hit = trimmed.get("model", [1, 2, 3])
    assert hit == 2 and cache[0].tokens == [1, 2]


def test_cancelled_prefill_keeps_only_consumed_independent_snapshot():
    engine, request, cache, _, arrays = setup(True)
    cancel = threading.Event()

    def cancelling_model(chunk, *, cache):
        model(chunk, cache=cache)
        cancel.set()

    engine.model = cancelling_model
    with pytest.raises(InterruptedError, match="cancelled"):
        engine._prefill_prompt(request, TOKENS, cache, cache, [], "model", 0, False, arrays, cancel)
    assert arrays.chunks == [TOKENS[:3]]
    restored, hit = engine.cache.get("model", TOKENS)
    assert hit == 3 and restored[0].tokens == TOKENS[:3]
    cache[0].tokens.append(999)
    assert engine.cache.get("model", TOKENS)[0][0].tokens == TOKENS[:3]
