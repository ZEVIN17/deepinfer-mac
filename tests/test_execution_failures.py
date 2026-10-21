"""Portable checks for failure recovery and cached native input ownership."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.policy import DSAdaptivePolicy
from deepinfer_mac.speculative import adaptive_generate_step
from deepinfer_mac.state import GenerationRequest


class Inputs(list):
    def __getitem__(self, key):
        return self if key is None else super().__getitem__(key)


@pytest.fixture
def fake_device(monkeypatch):
    """Inject only the device boundary needed before a failed forward."""
    mx = ModuleType("mlx.core")
    mx.uint32 = int
    mx.array = lambda values, dtype: Inputs(values)
    mx.concatenate = lambda values: Inputs([item for value in values for item in value])
    mx.synchronize = mx.reset_peak_memory = lambda: None
    mx.get_active_memory = mx.get_peak_memory = lambda: 0
    mlx = ModuleType("mlx")
    mlx.core = mx
    cache = ModuleType("mlx_lm.models.cache")
    cache.can_trim_prompt_cache = lambda caches: True
    cache.trim_prompt_cache = lambda caches, n: [c.trim(n) for c in caches]
    cache.make_prompt_cache = lambda model: []
    generate = ModuleType("mlx_lm.generate")
    generate.generate_step = generate.speculative_generate_step = None
    generate.device = mx
    for name, module in {
        "mlx": mlx, "mlx.core": mx, "mlx_lm": ModuleType("mlx_lm"),
        "mlx_lm.models": ModuleType("mlx_lm.models"),
        "mlx_lm.models.cache": cache, "mlx_lm.generate": generate,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return generate


class PrefixState:
    nbytes = 8

    def __init__(self, offset):
        self.offset = offset

    def trim(self, count):
        self.offset -= count


@pytest.mark.parametrize("failed_budget", [0, 1])
def test_failed_forward_does_not_poison_persistent_ds_policy(fake_device, failed_budget):
    policy = DSAdaptivePolicy(candidates=(0, 1), exploration_samples=1)
    if failed_budget:
        assert policy.choose(5) == 0
        policy.observe_cycle(0, 0, 0, 1, 1, cycle_ms=1)
    completed = policy.state.process_cnt

    def fail(*args, **kwargs):
        raise RuntimeError("injected model failure")

    for _ in range(2):
        target = [PrefixState(3)]
        generator = adaptive_generate_step([0], fail, fail, target, [PrefixState(3)], policy, 5, [])
        with pytest.raises(RuntimeError, match="injected model failure"):
            next(generator)
        assert policy.state.schedule_cnt == policy.state.process_cnt == completed
        assert not policy.costs[failed_budget]  # Failed work is not a successful runtime sample.
        assert target[0].offset == 3  # No guessed rewind of an unexecuted target.
    for _ in range(5):
        budget = policy.choose(5)
        policy.observe_cycle(budget, 0, 0, 1, 1, cycle_ms=1)
    assert policy.state.schedule_cnt == policy.state.process_cnt == completed + 5


@pytest.mark.parametrize("explicit_budget", [False, True])
def test_native_cached_request_only_consumes_uncached_suffix(
    fake_device, monkeypatch, tmp_path, explicit_budget,
):
    engine = InferenceEngine(cache_backend="exact")
    if explicit_budget:
        fake_device.device.get_active_memory = lambda: 100
        # 100 active + 2064 conservative KV reserve fits. Counting the 8
        # logical snapshot bytes again would wrongly clear this usable hit.
        engine.memory_limit_bytes = 2168
    engine._load = lambda *args: None
    engine._tokens = lambda request: [1, 2, 3]
    engine.model = SimpleNamespace(layers=[])
    engine.tokenizer = SimpleNamespace(
        chat_template="test", eos_token_ids={4}, detokenizer=SimpleNamespace(reset=lambda: None),
    )
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 1,
    }))
    engine.paths = {"target": str(tmp_path)}
    request = GenerationRequest(prompt="test", mode="native_mlx", max_tokens=1, use_cache=True)
    engine.cache.put(engine._identity(request, False), [1, 2], [PrefixState(2)])

    def stream(model, tokenizer, tokens, **kwargs):
        assert tokens == [3]
        assert kwargs["prompt_cache"][0].offset == 2
        yield SimpleNamespace(token=4, text="")

    fake_device.stream_generate = stream
    monkeypatch.setattr("deepinfer_mac.engine.read_swap_bytes", lambda: 0)
    result = engine.generate(request)
    assert result.metrics["cache_hit_tokens"] == 2
    assert result.token_ids == [4]
    assert engine.cache.get(engine._identity(request, False), [1, 2, 3])[0][0].offset == 2
