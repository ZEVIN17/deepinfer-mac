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
def test_adapter_matches_target_all_accept_first_and_middle_reject(reject, limit):
    from deepinfer_mac.speculative import adaptive_generate_step

    target, draft = [CounterCache()], [CounterCache()]
    rounds = []
    generated = list(
        adaptive_generate_step(
            [0], transition(), transition(reject), target, draft, Budget([3]), limit, rounds
        )
    )
    assert [t for t, accepted in generated] == list(range(1, limit + 1))
    if reject is None and limit >= 7:
        assert any(r["accepted"] == r["budget"] for r in rounds)
    if reject == 0:
        assert rounds[0]["accepted"] == 0
    if reject == 1 and limit >= 2:
        assert rounds[0]["accepted"] == 1


def test_zero_budget_catchup_and_mid_yield_close():
    from deepinfer_mac.speculative import adaptive_generate_step

    target, draft = [CounterCache()], [CounterCache()]
    generator = adaptive_generate_step(
        [0], transition(), transition(), target, draft, Budget([0, 0, 3]), 12, []
    )
    assert [next(generator)[0] for _ in range(5)] == [1, 2, 3, 4, 5]
    generator.close()
    # As in upstream, full acceptance leaves target one token ahead of draft;
    # the pending last accepted token is consumed on the next draft call.
    assert target[0].offset == 6 and draft[0].offset == 5


def test_mid_accepted_close_rewinds_unconsumed_tail():
    from deepinfer_mac.speculative import adaptive_generate_step

    target, draft = [CounterCache()], [CounterCache()]
    generator = adaptive_generate_step([0], transition(), transition(), target, draft, Budget([4]), 8, [])
    assert next(generator) == (1, True)
    assert next(generator) == (2, True)
    generator.close()  # Equivalent to EOS observed at token 2 by the caller.
    assert target[0].offset == draft[0].offset == 3
