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


@pytest.mark.parametrize("profile,mode,expected", [
    ("qwen3-4b", "baseline", None),
    ("qwen3-4b", "fixed_speculative", None),
    ("qwen3-4b", "native_mlx", None),
    ("qwen3-4b", "adaptive", "original"),
    ("qwen3.5-9b-8bit", "baseline", None),
    ("qwen3.5-9b-8bit", "fixed_speculative", None),
    ("qwen3.5-9b-8bit", "adaptive", "lagged_mean"),
])
def test_production_engine_scopes_hybrid_cost_adapter_to_hybrid_adaptive(
    fake_device, monkeypatch, tmp_path, profile, mode, expected,
):
    from deepinfer_mac.ds.recovery_cost import LaggedMeanRecoveryPolicy

    runtime = sys.modules["mlx_lm.models.cache"]
    hybrid = profile.startswith("qwen3.5")
    runtime.can_trim_prompt_cache = lambda caches: not hybrid

    class Layer:
        nbytes = 0
        state = ()

    runtime.make_prompt_cache = lambda model: [Layer()]
    engine = InferenceEngine(cache_budget_bytes=0, adaptive_scheduler="ds", hybrid_cost_adapter="lagged_mean",
                             experimental_hybrid_speculation=True)
    engine._load = lambda *args: None
    engine._tokens = lambda request: [0]
    engine.model = engine.draft_model = SimpleNamespace(layers=[None])
    engine.tokenizer = SimpleNamespace(
        chat_template="fixture", eos_token_ids={4},
        detokenizer=SimpleNamespace(reset=lambda: None, finalize=lambda: None, last_segment=""),
    )
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 1, "num_key_value_heads": 1, "head_dim": 1, "vocab_size": 16,
    }))
    engine.paths = {"target": str(tmp_path), "draft": str(tmp_path)}
    fake_device.generate_step = lambda *args, **kwargs: iter([(4, None)])
    fake_device.speculative_generate_step = lambda *args, **kwargs: iter([(4, None, False)])
    fake_device.stream_generate = lambda *args, **kwargs: iter([SimpleNamespace(token=4, text="")])
    selected = []

    def adapter(prompt, model, draft, target_cache, draft_cache, policy, *args, **kwargs):
        selected.append(policy)
        return iter([(4, False)])

    monkeypatch.setattr("deepinfer_mac.speculative.adaptive_generate_step", adapter)
    monkeypatch.setattr("deepinfer_mac.hybrid_speculative.hybrid_generate_step", adapter)
    result = engine.generate(GenerationRequest(prompt="fixture", profile=profile, mode=mode, max_tokens=1,
                                               use_cache=False))
    assert result.token_ids == [4]
    assert result.metrics["hybrid_cost_adapter_requested"] == "lagged_mean"
    assert result.metrics["hybrid_cost_adapter_effective"] == expected
    assert result.metrics["hybrid_cost_adapter_applicable"] is (hybrid and mode == "adaptive")
    if mode == "adaptive":
        assert isinstance(selected[0], LaggedMeanRecoveryPolicy) is hybrid
        assert isinstance(selected[0], DSAdaptivePolicy)
    else:
        assert not any(isinstance(policy, LaggedMeanRecoveryPolicy) for policy in engine.policies.values())
