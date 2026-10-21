"""Independent unchanged-upstream C++ oracle; no MLX, model weights or network."""

import hashlib
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from deepinfer_mac.ds import ForwardRuntimeEstimator, MTPState, MTPVarlen, SamplerRuntimeEstimator
from deepinfer_mac.policy import DSAdaptivePolicy

ROOT = Path(__file__).parent / "reference"


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++17 compiler is required for the independent source oracle")
    executable = tmp_path_factory.mktemp("ds-oracle") / "oracle"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(ROOT / "shim"), str(ROOT / "mtp_oracle.cpp"),
         "-o", str(executable)], check=True, capture_output=True,
    )

    def run(commands):
        result = subprocess.run([str(executable)], input="\n".join(commands) + "\n", text=True,
                                capture_output=True, check=True)
        return result.stdout.splitlines()

    return run


def test_source_hash_unchanged():
    assert hashlib.sha256((ROOT / "vendor/mtp_varlen.hpp").read_bytes()).hexdigest() == (
        "164f45db4cccd6d6909dd2846e7dfffb57754b108af679443bf4bd71ec33785a"
    )


def test_runtime_estimators_against_cpp_including_eviction(oracle):
    rng = random.Random(774)
    sampler, forward = SamplerRuntimeEstimator(), ForwardRuntimeEstimator()
    commands, expected = [], []
    for i in range(5110):
        value = rng.choice([-2.5, -1, 0, 0, 0.125, 0.5, 3, 9])
        graph = rng.choice([1, 2, 4, 8, 512])
        commands.append(f"F {graph} {value}")
        forward.update(graph, value)
        expected.append(str(forward.get()))
        if i < 30:
            commands.append(f"S {value}")
            sampler.update(value)
            expected.append("null" if sampler.get() is None else str(sampler.get()))
    actual = oracle(commands)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == e == "null" or float(a) == pytest.approx(float(e), abs=1e-14)


def serialize_state(s):
    return " ".join(map(str, [len(s.draft_tokens), int(s.has_draft_info), s.process_cnt,
                             s.schedule_cnt, len(s.accept_rates), *s.accept_rates,
                             len(s.prev_accept_rates), *s.prev_accept_rates]))


def test_budget_decisions_against_cpp_randomized(oracle):
    rng = random.Random(424242)
    commands, expected = [], []
    scheduler = MTPVarlen()
    # Includes empty curves, holes between graph sizes, ties, no draft,
    # unpaired process/schedule, negative residual and insufficient histories.
    for case in range(250):
        n = rng.choice([1, 2, 4])
        curve = [] if case % 17 == 0 else [(i, rng.uniform(0.5, 20)) for i in [4, 8, 16, 32]]
        scheduler.set_decode_perf_curve(curve)
        commands.append("C " + str(len(curve)) + " " + " ".join(f"{g} {v}" for g, v in curve))
        if curve:
            for _ in range(rng.randrange(15)):
                graph = rng.choice([4, 8, 16])
                forward_ms, sample_ms = rng.uniform(0, 25), rng.uniform(0, 1)
                scheduler.update_runtime(n, graph, forward_ms, sample_ms)
                commands.append(f"R {n} {graph} {forward_ms} {sample_ms}")
        states = []
        for _ in range(n):
            nd = rng.randrange(6)
            rates = lambda nd=nd: [rng.choice([0, 0.1, 0.5, 0.9, 1]) for _ in range(nd)]
            states.append(MTPState(list(range(nd)), rates() if rng.random() < .8 else [],
                                   rates() if rng.random() < .8 else [], bool(rng.getrandbits(1)),
                                   case, case + rng.randrange(2)))
        commands.append(f"Q {n} " + " ".join(serialize_state(s) for s in states))
        expected.append(scheduler.schedule(states))
    actual = [[int(v) for v in row.split()] for row in oracle(commands)]
    assert actual == expected


def test_adjacent_history_isolation_and_ties():
    scheduler = MTPVarlen([(2, 10), (3, 12), (4, 13)])
    a = MTPState([1], [1], [0], True)
    b = MTPState([1], [0], [0], True)
    assert scheduler.schedule([a, b]) == [1, 1]  # N-1 cannot inflate total budget.
    a.prev_accept_rates = b.prev_accept_rates = [1]
    assert scheduler.schedule([a, b]) == [2, 2]
    scheduler.set_decode_perf_curve([(2, 10), (3, 10.1), (4, 30)])
    b.accept_rates = [1]
    assert scheduler.schedule([a, b]) == [2, 1]  # Equal rates: earlier sequence wins.
    a.schedule_cnt = 1
    a.accept_rates = [0]
    b.prev_accept_rates = [0]
    assert scheduler.schedule([a, b]) == [1, 1]  # Unpaired uses accept_rates as N-2.


def test_platform_validation_and_empty_batch():
    with pytest.raises(ValueError):
        MTPVarlen([(2, 1), (1, 1)])
    scheduler = MTPVarlen([(1, 2)])
    assert scheduler.schedule([]) == []
    with pytest.raises(ValueError):
        scheduler.update_runtime(1, 2, 1, 0)
    with pytest.raises(ValueError):
        scheduler.schedule([MTPState(schedule_cnt=2)])


def test_empirical_adapter_replay_and_expiring_fallback():
    policy = DSAdaptivePolicy(candidates=(0, 1, 2), exploration_samples=1, refresh_steps=5)
    budgets = []
    for _ in range(20):
        k = policy.choose(10)
        budgets.append(k)
        policy.observe_cycle(k, 0, 0, 10 + k * 20, 1, cycle_ms=10 + k * 20)
    assert budgets[:3] == [0, 1, 2]
    assert budgets[3] == 0
    assert 1 in budgets[6:] and 2 in budgets[6:]  # No permanent disabled bucket.
    assert policy.state.schedule_cnt == policy.state.process_cnt == 20
    assert policy.choose(10, memory_pressure=True) == 0
    assert policy.summary()["confidence_head"] is False


def test_lagged_mean_estimated_curves_still_execute_original_cpp_budget_rule(oracle):
    from deepinfer_mac.ds.recovery_cost import LaggedMeanRecoveryPolicy

    # Unit fixture costs are artificial, not model benchmark measurements.
    rng = random.Random(9917)
    policy = LaggedMeanRecoveryPolicy(candidates=(0, 1, 2), min_samples=2, window=4)
    commands, expected = [], []
    for _ in range(150):
        budget = policy.choose(30)
        if policy.last_replay is not None:
            replay = policy.last_replay
            curve = replay["curve"]
            commands.append("C " + str(len(curve)) + " " + " ".join(f"{g} {v}" for g, v in curve))
            commands.append("Q 1 " + serialize_state(MTPState(**replay["state"])))
            expected.append([budget + 1])
            assert all(s["cycle"] <= replay["state"]["process_cnt"] - 1
                       for point in replay["cost_model"]["points"] for s in point["measured_samples"])
        accepted = rng.randrange(budget + 1)
        ms = 10 + budget * 2 + (15 if accepted < budget else 0) + rng.random()
        # Interleave censored consumer closes; they advance generation age but
        # cannot fabricate rejections at positions beyond emitted acceptance.
        partial = accepted > 0 and policy.steps % 13 == 0
        before_positions = {k: list(v) for k, v in policy.positions.items()}
        policy.observe_cycle(budget, accepted, 0, ms, accepted if partial else accepted + 1, cycle_ms=ms)
        if partial:
            assert dict(policy.positions) == before_positions
    assert len(expected) > 100
    assert [[int(v) for v in row.split()] for row in oracle(commands)] == expected
    assert not policy.scheduler.forward.window  # Deliberate direct-cost adapter, no hidden reset.
