"""Tiny deterministic MLX transitions test control flow; these are not benchmarks."""

import platform

import pytest

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() != "arm64", reason="requires Apple Silicon"
    ),
]


class CounterCache:
    def __init__(self):
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, count):
        count = min(count, self.offset)
        self.offset -= count
        return count


class Budget:
    def __init__(self, values):
        self.values = iter(values)
        self.last = 2

    def choose(self, remaining, memory_pressure=False):
        self.last = next(self.values, self.last)
        return min(self.last, remaining)

    def observe(self, *args):
        pass


def transition(reject_input=None):
    import mlx.core as mx

    class Model:
        def __call__(self, x, cache):
            cache[0].offset += x.size
            target = (x + 1) % 16
            if reject_input is not None:
                target = mx.where(x == reject_input, (x + 2) % 16, target)
            return mx.where(mx.arange(16)[None, None, :] == target[..., None], 10.0, -10.0)

    return Model()


@pytest.mark.parametrize("reject", [None, 0, 1])
@pytest.mark.parametrize("limit", [1, 2, 7, 11])
@pytest.mark.parametrize("deferred", [False, True])
def test_adapter_matches_target_all_accept_first_and_middle_reject(reject, limit, deferred):
    from deepinfer_mac.speculative import adaptive_generate_step

    target, draft = [CounterCache()], [CounterCache()]
    rounds = []
    generated = list(
        adaptive_generate_step(
            [0],
            transition(),
            transition(reject),
            target,
            draft,
            Budget([3]),
            limit,
            rounds,
            defer_draft_sync=deferred,
        )
    )
    assert [t for t, accepted in generated] == list(range(1, limit + 1))
    if reject is None and limit >= 7:
        assert any(r["accepted"] == r["budget"] for r in rounds)
    if reject == 0:
        assert rounds[0]["accepted"] == 0
    if reject == 1 and limit >= 2:
        assert rounds[0]["accepted"] == 1


@pytest.mark.parametrize("deferred", [False, True])
def test_zero_budget_catchup_and_mid_yield_close(deferred):
    from deepinfer_mac.speculative import adaptive_generate_step

    target, draft = [CounterCache()], [CounterCache()]
    generator = adaptive_generate_step(
        [0], transition(), transition(), target, draft, Budget([0, 0, 3]), 12, [], defer_draft_sync=deferred
    )
    assert [next(generator)[0] for _ in range(5)] == [1, 2, 3, 4, 5]
    generator.close()
    # As in upstream, full acceptance leaves target one token ahead of draft;
    # the pending last accepted token is consumed on the next draft call.
    assert target[0].offset == 6 and draft[0].offset == 5


@pytest.mark.parametrize("deferred", [False, True])
def test_mid_accepted_close_rewinds_unconsumed_tail(deferred):
    from deepinfer_mac.speculative import adaptive_generate_step

    target, draft = [CounterCache()], [CounterCache()]
    generator = adaptive_generate_step(
        [0], transition(), transition(), target, draft, Budget([4]), 8, [], defer_draft_sync=deferred
    )
    assert next(generator) == (1, True)
    assert next(generator) == (2, True)
    generator.close()  # Equivalent to EOS observed at token 2 by the caller.
    assert target[0].offset == draft[0].offset == 3


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("close_early", [False, True])
def test_ds_scheduler_in_real_adapter_observes_complete_cycles(diagnostics, close_early):
    from deepinfer_mac.policy import DSAdaptivePolicy
    from deepinfer_mac.speculative import adaptive_generate_step

    policy = DSAdaptivePolicy(exploration_samples=1)
    target, draft, rounds = [CounterCache()], [CounterCache()], []
    generator = adaptive_generate_step(
        [0], transition(), transition(), target, draft, policy, 40, rounds,
        defer_draft_sync=True, diagnostics=diagnostics,
    )
    output = []
    try:
        for token, _ in generator:
            output.append(token)
            if close_early and len(output) == 12:
                break
    finally:
        generator.close()
    assert output == [i % 16 for i in range(1, len(output) + 1)]
    assert policy.state.process_cnt == policy.state.schedule_cnt == len(rounds)
    assert sum(r["produced"] for r in rounds) == len(output)
    assert target[0].offset - draft[0].offset in (0, 1)
    if diagnostics:
        assert all(r["cycle_ms"] >= r["draft_ms"] + r["verify_ms"] for r in rounds)
        if not close_early:
            assert any("scheduler_replay" in r for r in rounds)


@pytest.mark.parametrize("failed_budget", [0, 1])
def test_ds_policy_recovers_after_real_device_boundary_forward_failure(failed_budget):
    from deepinfer_mac.policy import DSAdaptivePolicy
    from deepinfer_mac.speculative import adaptive_generate_step

    policy = DSAdaptivePolicy(exploration_samples=1)
    if failed_budget:
        assert policy.choose(40) == 0
        policy.observe_cycle(0, 0, 0, 1, 1, cycle_ms=1)
    before = policy.state.process_cnt

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    generator = adaptive_generate_step(
        [0], fail, fail, [CounterCache()], [CounterCache()], policy, 40, [], defer_draft_sync=True,
    )
    with pytest.raises(RuntimeError, match="injected forward failure"):
        next(generator)
    assert policy.state.schedule_cnt == policy.state.process_cnt == before
    output = list(adaptive_generate_step(
        [0], transition(), transition(), [CounterCache()], [CounterCache()], policy, 40, [],
        defer_draft_sync=True,
    ))
    assert [token for token, _ in output] == [i % 16 for i in range(1, 41)]
    assert policy.state.schedule_cnt == policy.state.process_cnt


@pytest.mark.parametrize("reject", [None, 0, 2])
@pytest.mark.parametrize("close_early", [False, True])
def test_hybrid_actual_arrays_and_kv_cache_recovery(reject, close_early):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from deepinfer_mac.hybrid_speculative import hybrid_generate_step
    from deepinfer_mac.policy import DSAdaptivePolicy

    class HybridTransition:
        def __init__(self, reject_at=None):
            self.reject_at = reject_at

        def __call__(self, x, cache):
            recurrent, kv = cache
            old_sum = recurrent[0] if recurrent[0] is not None else mx.array([0])
            cumulative = mx.cumsum(x, axis=1) + old_sum[:, None]
            recurrent[0] = cumulative[:, -1]
            # A second recurrent tensor and real KV offsets both require restore.
            previous = recurrent[1] if recurrent[1] is not None else mx.zeros((1, 0), mx.uint32)
            recurrent[1] = mx.concatenate([previous, x], axis=1)[:, -2:]
            positions = kv.offset + mx.arange(x.shape[1])
            values = x[:, None, :, None]
            kv.update_and_fetch(values, values)
            prediction = (x + cumulative + 1) % 16
            if self.reject_at is not None:
                prediction = mx.where(positions[None, :] % 5 == self.reject_at,
                                      (prediction + 1) % 16, prediction)
            return mx.where(mx.arange(16)[None, None, :] == prediction[..., None], 10., -10.)

    def cache():
        return [ArraysCache(size=2), KVCache()]

    target, draft = cache(), cache()
    combined = target + draft
    identities = [id(c) for c in combined]
    policy = DSAdaptivePolicy(exploration_samples=1)
    generator = hybrid_generate_step([0], HybridTransition(), HybridTransition(reject),
                                     target, draft, policy, 60, [])
    output = []
    try:
        for token, _ in generator:
            output.append(token)
            if close_early and len(output) == 13:
                break
    finally:
        generator.close()
    baseline, state, y = [], cache(), 0
    for _ in range(len(output)):
        logits = HybridTransition()(mx.array([[y]], mx.uint32), state)
        y = int(mx.argmax(logits[0, -1]).item())
        baseline.append(y)
    assert output == baseline
    assert identities == [id(c) for c in combined]
    assert target[1].offset == len(output)
    assert target[1].keys_and_values()[0].flatten().tolist() == [0, *output[:-1]]
    assert mx.array_equal(target[0][0], state[0][0]).item()
    assert mx.array_equal(target[0][1], state[0][1]).item()
    assert policy.state.schedule_cnt == policy.state.process_cnt
