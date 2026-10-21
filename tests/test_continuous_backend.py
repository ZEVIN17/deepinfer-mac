"""Device-free boundary tests exercise real wrapper calls, not timing claims."""
import json
import sys
import types
from types import SimpleNamespace

import pytest

from deepinfer_mac.continuous import ContinuousConfig, MLXBatchSession
from deepinfer_mac.state import GenerationRequest


class Detokenizer:
    def reset(self):
        self.tokens = []


def test_wrapper_batches_merged_kv_and_calls_one_next_per_tick(monkeypatch, tmp_path):
    calls = []
    class SingleCache:
        # Dense unbatched KVCache need not support filter/extend/prepare.
        def merge(self, caches):
            assert caches == [self]
            calls.append("merge")
            return BatchCache()

    class BatchCache:
        def filter(self, _):
            pass

        def extend(self, _):
            pass

        def extract(self, _):
            pass

        def prepare(self, **kwargs):
            pass

    class Generator:
        def __init__(self, model, **kwargs):
            calls.append(("init", kwargs))
            self._prompt_batch = SimpleNamespace(prefill_step_size=kwargs["prefill_step_size"])
            self.prompt_cache_nbytes = 5
            self.stream = object()

        def insert_segments(self, segments, **kwargs):
            calls.append(("insert_segments", segments, kwargs))
            return [42]

        def next(self):
            calls.append("next")
            return [SimpleNamespace(uid=42, progress=(2, 20), end_of_prompt=False)], []

        def next_generated(self):
            raise AssertionError("must not swallow pure prefill ticks")

        def remove(self, uids, **kwargs):
            calls.append(("remove", uids, kwargs))

        def close(self):
            calls.append("close")

    core = types.ModuleType("mlx.core")
    core.get_active_memory = lambda: 100
    core.clear_cache = lambda: None
    core.synchronize = lambda stream: calls.append("sync")
    mlx = types.ModuleType("mlx")
    mlx.core = core
    generate = types.ModuleType("mlx_lm.generate")
    generate.BatchGenerator = Generator
    cache = types.ModuleType("mlx_lm.models.cache")
    cache.make_prompt_cache = lambda model: [SingleCache()]
    for name, module in (("mlx", mlx), ("mlx.core", core), ("mlx_lm.generate", generate), ("mlx_lm.models.cache", cache)):
        monkeypatch.setitem(sys.modules, name, module)
    from deepinfer_mac import continuous
    monkeypatch.setattr(continuous, "version", lambda package: "0.32.2")
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 100}))
    adapter = SimpleNamespace(cache=SimpleNamespace(capabilities=lambda caches: {"batch": False}),
                              memory_estimate=lambda cfg, tokens: tokens * 4)
    registry = SimpleNamespace(get=lambda profile: {"id": profile, "context_limit": 100},
                               adapter=lambda profile: adapter, public_metadata=lambda metadata: metadata)
    tokenizer = SimpleNamespace(detokenizer=Detokenizer(), eos_token_ids={0})
    engine = SimpleNamespace(profile="tiny", model=object(), draft_model=None, tokenizer=tokenizer,
                             _check_snapshot_unchanged=lambda: None, _load=lambda *args: None,
                             clear_cache=lambda: None, model_registry=registry,
                             paths={"target": str(tmp_path)}, memory_limit_bytes=None,
                             _tokens=lambda request: list(range(1, 21)), _identity=lambda *args: "identity")
    config = ContinuousConfig(max_active=2, prefill_chunk=8, tick_token_budget=10)
    session = MLXBatchSession(engine, "tiny", config)
    prepared = session.prepare(GenerationRequest(prompt="long prompt", max_tokens=7))
    assert prepared.detokenizer is not tokenizer.detokenizer
    assert session.model_warm and session.cache_capabilities["batch"] is False
    assert session.insert(prepared, GenerationRequest(prompt="a", max_tokens=7)) == 42
    prefills, outputs = session.tick()
    assert prefills[0].progress == (2, 20) and not outputs
    assert calls.count("next") == 1
    init = next(call[1] for call in calls if isinstance(call, tuple) and call[0] == "init")
    assert init["prefill_batch_size"] == 1 and init["completion_batch_size"] == 2
    insertion = next(call for call in calls if isinstance(call, tuple) and call[0] == "insert_segments")
    assert insertion[2] == {"max_tokens": [7]}
    assert session.resources()["actual_cache_bytes"] == 5
    session.remove(42)
    assert ("remove", [42], {"return_prompt_caches": False}) in calls
    session.close()
    session.close()
    assert calls[-2:] == ["sync", "close"]


def test_unaudited_upstream_version_rejects_before_loading_model(monkeypatch):
    from deepinfer_mac import continuous
    monkeypatch.setattr(continuous, "version", lambda package: "99.0")
    with pytest.raises(ValueError, match="audited"):
        MLXBatchSession(None, "unused", ContinuousConfig())
