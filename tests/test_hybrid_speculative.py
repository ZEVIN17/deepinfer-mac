"""Portable transactional hybrid-state checks, no NumPy/MLX dependency."""

import sys
from types import ModuleType

import pytest

from deepinfer_mac.ds.recovery_cost import LaggedMeanRecoveryPolicy
from deepinfer_mac.hybrid_speculative import hybrid_generate_step, restore_cache, snapshot_cache
from deepinfer_mac.policy import DSAdaptivePolicy, FixedPolicy


class Array(list):
    def __getitem__(self, key):
        if key is None:
            return self
        value = super().__getitem__(key)
        return Array(value) if isinstance(key, slice) else value

    def astype(self, dtype):
        return self

    def tolist(self):
        return list(self)


class Logits:
    def __init__(self, predictions):
        self.predictions = Array(predictions)

    def __getitem__(self, key):
        return Logits(self.predictions[key[1]])

    def squeeze(self, axis):
        return self


@pytest.fixture
def fake_device(monkeypatch):
    mx = ModuleType("mlx.core")
    mx.uint32 = int
    mx.array = lambda values, dtype: Array(values)
    mx.concatenate = lambda arrays: Array([v for array in arrays for v in array])
    mx.argmax = lambda logits, axis: logits.predictions
    mx.eval = mx.async_eval = lambda *args: None
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)


class RecurrentCache:
    def __init__(self):
        self.history = []
        self.recurrent = 0

    @property
    def state(self):
        return self.history, self.recurrent

    @state.setter
    def state(self, state):
        self.history, self.recurrent = state

    def consume(self, token):
        self.history = [*self.history, token]
        self.recurrent = (self.recurrent * 7 + token + 3) % 17

    def trim(self, count):
        raise AssertionError("Hybrid state must never be trimmed")


class Model:
    def __init__(self, reject_at=None, fail=False):
        self.reject_at = reject_at
        self.fail = fail
        self.calls = 0

    def __call__(self, inputs, cache):
        self.calls += 1
        predictions = []
        for token in inputs:
            for state in cache:
                state.consume(token)
            value = (token + cache[0].recurrent) % 16
            if self.reject_at is not None and len(cache[0].history) % 5 == self.reject_at:
                value = (value + 1) % 16
            predictions.append(value)
        if self.fail:
            raise RuntimeError("injected after cache mutation")
        return Logits(predictions)


def reference(limit):
    cache, model, y, output = [RecurrentCache()], Model(), 0, []
    for _ in range(limit):
        y = model(Array([y]), cache).predictions[0]
        output.append(y)
    return output


class Scheduled(FixedPolicy):
    def __init__(self):
        self.steps = 0

    def choose(self, remaining, memory_pressure=False):
        self.steps += 1
        return 0 if self.steps % 7 < 4 else min(4, remaining)


@pytest.mark.parametrize("reject", [None, 0, 1, 3])
@pytest.mark.parametrize("policy_name", ["zero", "one", "four", "scheduled", "ds", "lagged"])
@pytest.mark.parametrize("close_early", [False, True])
def test_hybrid_restores_actual_consumed_history(fake_device, reject, policy_name, close_early):
    policy = {"zero": lambda: FixedPolicy(0), "one": lambda: FixedPolicy(1),
              "four": lambda: FixedPolicy(4), "scheduled": Scheduled,
              "ds": lambda: DSAdaptivePolicy(exploration_samples=1),
              "lagged": lambda: LaggedMeanRecoveryPolicy(min_samples=1)}[policy_name]()
    target, draft, rounds = [RecurrentCache(), RecurrentCache()], [RecurrentCache()], []
    identities = [id(c) for c in target + draft]
    model, draft_model = Model(), Model(reject)
    generator = hybrid_generate_step([0], model, draft_model, target, draft, policy, 70, rounds)
    output = []
    try:
        for token, _ in generator:
            output.append(token)
            if close_early and len(output) == 13:
                break
    finally:
        generator.close()
    assert output == reference(len(output))
    expected = [0, *output[:-1]]
    assert all(c.history == expected for c in target)
    assert draft[0].history == expected[:len(draft[0].history)]
    assert [id(c) for c in target + draft] == identities
    assert sum(r["produced"] for r in rounds) == len(output)
    assert all(r["cycle_ms"] >= r["snapshot_ms"] + r["verify_ms"] + r["recovery_ms"] for r in rounds)
    if policy_name == "zero":
        assert draft_model.calls == 0
    if policy_name in ("ds", "lagged"):
        assert policy.state.process_cnt == policy.state.schedule_cnt == len(rounds)
        if not close_early:
            assert any("scheduler_replay" in r for r in rounds)


@pytest.mark.parametrize("fail_target", [False, True])
def test_failed_hybrid_forward_restores_checkpoint_and_aborts_counter(fake_device, fail_target):
    policy = DSAdaptivePolicy(candidates=(0, 1), exploration_samples=1)
    assert policy.choose(5) == 0
    policy.observe_cycle(0, 0, 0, 1, 1, cycle_ms=1)
    target, draft = [RecurrentCache()], [RecurrentCache()]
    target[0].consume(7)
    draft[0].consume(7)
    snapshots = snapshot_cache(target), snapshot_cache(draft)
    generator = hybrid_generate_step([0], Model(fail=fail_target), Model(fail=not fail_target),
                                     target, draft, policy, 5, [])
    with pytest.raises(RuntimeError, match="injected"):
        next(generator)
    assert snapshot_cache(target) == snapshots[0]
    assert snapshot_cache(draft) == snapshots[1]
    assert policy.state.schedule_cnt == policy.state.process_cnt == 1


def test_snapshots_are_branch_safe_and_preserve_cache_identity():
    cache = [RecurrentCache()]
    cache[0].consume(1)
    snap = snapshot_cache(cache)
    identity = id(cache[0])
    restore_cache(cache, snap)
    cache[0].history.append(3)
    restore_cache(cache, snap)
    assert id(cache[0]) == identity
    assert cache[0].history == [1]
    assert snap[0][0] == [1]


def test_hybrid_attention_metadata_counts_only_full_attention_layers():
    from deepinfer_mac.engine import attention_config

    flat = {"num_hidden_layers": 40, "num_key_value_heads": 8, "head_dim": 128}
    assert attention_config(flat) == flat
    hybrid = {"model_type": "qwen3_5", "text_config": {
        "num_hidden_layers": 32, "num_key_value_heads": 4, "head_dim": 256,
        "full_attention_interval": 4,
    }}
    assert attention_config(hybrid)["num_hidden_layers"] == 8
    assert hybrid["text_config"]["num_hidden_layers"] == 32
    hybrid["text_config"]["layer_types"] = ["linear_attention", "full_attention", "linear_attention"]
    assert attention_config(hybrid)["num_hidden_layers"] == 1


def policy_before_first_budget_four(kind):
    from deepinfer_mac.policy import AdaptivePolicy

    policy = {"legacy": lambda: AdaptivePolicy(exploration_samples=1),
              "ds": lambda: DSAdaptivePolicy(exploration_samples=1),
              "lagged": lambda: LaggedMeanRecoveryPolicy(min_samples=1)}[kind]()
    for _ in range(100):
        budget = policy.choose(100)
        if budget == 4:
            if hasattr(policy, "abort_cycle"):
                policy.abort_cycle()
            return policy
        policy.observe_cycle(budget, budget, 0, 10, budget + 1, cycle_ms=10)
    raise AssertionError("Never reached the first budget-four exploration")


@pytest.mark.parametrize("adapter", ["hybrid", "trimmable"])
@pytest.mark.parametrize("kind", ["legacy", "ds", "lagged"])
@pytest.mark.parametrize("termination", ["close", "consumer_exception"])
def test_partial_real_generator_does_not_poison_future_policy(
    fake_device, monkeypatch, adapter, kind, termination,
):
    import copy

    from deepinfer_mac.speculative import adaptive_generate_step

    policy = policy_before_first_budget_four(kind)
    before_positions = copy.deepcopy(dict(policy.positions))
    before_costs = {k: list(v) for k, v in policy.samples.items()}
    before_cycle_costs = {k: list(v) for k, v in getattr(policy, "costs", {}).items()}
    before_rates = copy.deepcopy(policy.state.accept_rates) if hasattr(policy, "state") else None

    class Trimmable(RecurrentCache):
        def trim(self, count):
            removed = min(count, len(self.history))
            history = self.history[:-removed] if removed else self.history
            self.history, self.recurrent = [], 0
            for token in history:
                self.consume(token)
            return removed

    if adapter == "trimmable":
        runtime = ModuleType("mlx_lm.models.cache")
        runtime.can_trim_prompt_cache = lambda caches: True
        def trim_all(caches, count):
            first = 0
            for index, cache in enumerate(caches):
                removed = cache.trim(count)
                if index == 0:
                    first = removed
            return first

        runtime.trim_prompt_cache = trim_all
        monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
        monkeypatch.setitem(sys.modules, "mlx_lm.models", ModuleType("mlx_lm.models"))
        monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", runtime)
        generator = adaptive_generate_step([0], Model(), Model(), [Trimmable()], [Trimmable()],
                                          policy, 100, [], defer_draft_sync=True)
    else:
        generator = hybrid_generate_step([0], Model(), Model(), [RecurrentCache()], [RecurrentCache()],
                                         policy, 100, [])
    if termination == "consumer_exception":
        with pytest.raises(RuntimeError, match="consumer emit failed"):
            try:
                for _, accepted in generator:
                    assert accepted is True
                    raise RuntimeError("consumer emit failed")
            finally:
                generator.close()
    else:
        assert next(generator)[1] is True
        generator.close()
    assert dict(policy.positions) == before_positions
    assert {k: list(v) for k, v in policy.samples.items()} == before_costs
    assert {k: list(v) for k, v in getattr(policy, "costs", {}).items()} == before_cycle_costs
    assert policy.partial_cycles[-1]["accepted"] == policy.partial_cycles[-1]["produced"] == 1
    if hasattr(policy, "state"):
        assert policy.state.accept_rates == policy.state.prev_accept_rates == before_rates
        assert policy.state.schedule_cnt == policy.state.process_cnt
    assert policy.choose(100) == 4  # Missing budget-four evidence must still be explored.
    policy.observe_cycle(4, 4, 0, 10, 5, cycle_ms=10)
    assert policy.positions[3] == [1, 1]  # Only the later fully observed cycle contributes.
    if hasattr(policy, "state"):
        assert policy.state.prev_accept_rates == before_rates  # N-2 age remains correct.
