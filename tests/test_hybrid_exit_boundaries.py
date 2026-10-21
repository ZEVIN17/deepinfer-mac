"""Independent exit-boundary checks using the portable stateful teacher model."""

import threading
from types import ModuleType

import pytest
from test_hybrid_speculative import Model, RecurrentCache, fake_device, reference

from deepinfer_mac.hybrid_speculative import hybrid_generate_step
from deepinfer_mac.policy import DSAdaptivePolicy, FixedPolicy

__all__ = ["fake_device"]  # pytest fixture is shared, not a second model implementation.


def test_every_small_budget_exit_position_preserves_teacher_prefix(fake_device):
    # Closing at every position crosses accepted output, corrective output,
    # full-cycle and immediately-next-cycle boundaries, rather than only #13.
    for budget in (1, 2, 4):
        for reject_at in (None, 0, 1, 3):
            for stop in range(1, 10):
                target, draft = [RecurrentCache()], [RecurrentCache()]
                teacher = reference(stop)
                generator = hybrid_generate_step(
                    [0], Model(), Model(reject_at), target, draft,
                    FixedPolicy(budget), 30, [],
                )
                output = [next(generator)[0] for _ in range(stop)]
                generator.close()
                assert output == teacher
                assert target[0].history == [0, *teacher[:-1]]
                assert draft[0].history == target[0].history[:len(draft[0].history)]


def test_cancel_between_cycles_aborts_without_another_forward(fake_device):
    cancel = threading.Event()
    model, draft_model = Model(), Model()
    target, draft = [RecurrentCache()], [RecurrentCache()]
    policy = DSAdaptivePolicy(candidates=(0,), exploration_samples=1)
    rounds = []
    generator = hybrid_generate_step(
        [0], model, draft_model, target, draft, policy, 30, rounds, cancel=cancel,
    )
    first = next(generator)[0]
    cancel.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        next(generator)
    assert first == reference(1)[0]
    assert model.calls == 1 and draft_model.calls == 0
    assert target[0].history == [0]
    assert policy.state.schedule_cnt == policy.state.process_cnt == 1
    assert sum(r["produced"] for r in rounds) == 1


def test_consumer_exception_runs_close_and_keeps_last_output_unconsumed(fake_device):
    target, draft = [RecurrentCache()], [RecurrentCache()]
    policy = DSAdaptivePolicy(exploration_samples=1)
    generator = hybrid_generate_step([0], Model(), Model(), target, draft, policy, 30, [])
    output = []
    try:
        with pytest.raises(RuntimeError, match="consumer"):
            for token, _ in generator:
                output.append(token)
                if len(output) == 8:
                    raise RuntimeError("consumer stopped during accepted block")
    finally:
        generator.close()
    assert output == reference(8)
    assert target[0].history == [0, *output[:-1]]
    assert policy.state.schedule_cnt == policy.state.process_cnt


def test_verification_callback_failure_restores_precycle_and_aborts_policy(fake_device):
    target, draft = [RecurrentCache()], [RecurrentCache()]
    policy = DSAdaptivePolicy(candidates=(0, 1), exploration_samples=1)
    policy.choose(30)
    policy.observe_cycle(0, 0, 0, 1, 1, cycle_ms=1)

    def fail(_):
        raise RuntimeError("profiling callback")

    generator = hybrid_generate_step(
        [0], Model(), Model(), target, draft, policy, 30, [], verification_hook=fail,
    )
    with pytest.raises(RuntimeError, match="profiling callback"):
        next(generator)
    assert target[0].history == draft[0].history == []
    assert policy.state.schedule_cnt == policy.state.process_cnt == 1


def test_trimmable_all_accept_collects_steady_costs_and_enters_ds_schedule(fake_device, monkeypatch):
    """A normal two-input draft lag must not permanently classify as catch-up."""
    import sys

    from deepinfer_mac.speculative import adaptive_generate_step

    class TrimCache(RecurrentCache):
        def trim(self, count):
            count = min(len(self.history), count)
            retained = self.history[:-count] if count else self.history[:]
            self.history, self.recurrent = [], 0
            for token in retained:
                self.consume(token)
            return count

    cache_module = ModuleType("mlx_lm.models.cache")
    cache_module.can_trim_prompt_cache = lambda caches: True

    def trim_all(caches, count):
        for cache in caches:
            result = cache.trim(count)
        return result

    cache_module.trim_prompt_cache = trim_all
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)
    policy = DSAdaptivePolicy(candidates=(0, 1, 2), exploration_samples=2)
    rounds = []
    output = list(adaptive_generate_step(
        [0], Model(), Model(), [TrimCache()], [TrimCache()], policy, 70, rounds,
        defer_draft_sync=True,
    ))
    assert [token for token, _ in output] == reference(70)
    assert len(policy.costs[1]) >= 2 and len(policy.costs[2]) >= 2
    assert any("scheduler_replay" in record for record in rounds)
    assert policy.state.schedule_cnt == policy.state.process_cnt == len(rounds)
