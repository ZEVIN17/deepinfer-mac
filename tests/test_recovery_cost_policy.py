"""Cost feedback age, complete-cost accounting and explicit unknown boundaries."""

import copy

import pytest

from deepinfer_mac.ds.recovery_cost import LaggedMeanRecoveryPolicy
from deepinfer_mac.engine import InferenceEngine


def observe(policy, budget, ms, accepted=None):
    accepted = budget if accepted is None else accepted
    policy.observe_cycle(budget, accepted, 0, ms * .5, accepted + 1, cycle_ms=ms,
                         stage_ms={"snapshot_ms": ms * .1, "combined_draft_verify_ms": ms * .5,
                                   "recovery_ms": ms * .3})


def test_unknown_cost_never_fabricated_and_mean_includes_recovery():
    policy = LaggedMeanRecoveryPolicy(candidates=(0,), min_samples=5)
    for ms in [10, 10, 10, 10, 100]:
        assert policy.choose(20) == 0
        observe(policy, 0, ms)
    assert policy.summary()["unknown_graphs"] == [1]
    assert policy.last_replay is None
    assert policy.choose(20) == 0
    observe(policy, 0, 1)
    assert policy.choose(20) == 0
    replay = policy.last_replay
    assert replay["curve"] == [(1, 28.0)]  # E[cost], not median=10 or latest=1.
    assert replay["cost_model"]["feedback_cutoff_cycle"] == 5
    assert [s["cycle"] for s in replay["cost_model"]["points"][0]["measured_samples"]] == [1, 2, 3, 4, 5]
    assert replay["runtime_projection"] == {"forward_slope_ms": 0., "sampler_ms": 0.}


def test_latest_correlated_acceptance_and_cost_cannot_change_total_budget():
    base = LaggedMeanRecoveryPolicy(candidates=(0, 1), min_samples=1)
    for _ in range(8):
        budget = base.choose(20)
        observe(base, budget, 10)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    assert a.choose(20) == b.choose(20) == 1
    observe(a, 1, 200, accepted=0)
    observe(b, 1, 1_000_000, accepted=1)
    assert a.state.accept_rates != b.state.accept_rates  # The N-1 confound exists.
    assert a.state.prev_accept_rates == b.state.prev_accept_rates
    assert a.choose(20) == b.choose(20)
    assert a.last_replay["curve"] == b.last_replay["curve"]
    assert a.last_replay["cost_model"] == b.last_replay["cost_model"]
    assert max(s["cycle"] for p in a.last_cost_curve["points"] for s in p["measured_samples"]) <= 8


def test_partial_close_not_used_as_full_cycle_but_kept_in_actual_diagnostics():
    policy = LaggedMeanRecoveryPolicy(candidates=(0, 1), min_samples=1)
    assert policy.choose(10) == 0
    observe(policy, 0, 10)
    assert policy.choose(10) == 0
    observe(policy, 0, 10)
    assert policy.choose(10) == 1
    policy.observe_cycle(1, 1, 0, 10, 1, cycle_ms=20)  # Closed before final correction.
    assert policy.choose(10) == 1
    observe(policy, 1, 30)
    assert not policy.available[1]
    assert list(policy.costs[1]) == [30]
    assert policy.partial_cycles[-1]["cycle_ms"] == 20
    assert policy.positions[0] == [1, 1]
    assert policy.summary()["unknown_graphs"] == [2]


def test_abort_does_not_publish_pending_feedback_and_fast_path_skips_replay():
    policy = LaggedMeanRecoveryPolicy(candidates=(0,), min_samples=1)
    budget = policy.choose(10)
    observe(policy, budget, 10)
    pending = copy.deepcopy(policy.pending_observation)
    policy.choose(10)
    policy.abort_cycle()
    assert policy.pending_observation == pending and not policy.available[0]
    policy.capture_replay = False
    budget = policy.choose(10)
    observe(policy, budget, 10)
    policy.choose(10)
    assert policy.last_replay is None and policy.last_cost_curve is None
    assert policy.state.schedule_cnt == policy.state.process_cnt + 1


def test_invalid_measurement_and_optin_boundaries():
    with pytest.raises(ValueError):
        LaggedMeanRecoveryPolicy(min_samples=10, window=9)
    with pytest.raises(ValueError):
        InferenceEngine(hybrid_cost_adapter="lagged_mean")  # Legacy cannot silently switch.
    assert InferenceEngine(adaptive_scheduler="ds").hybrid_cost_adapter == "original"
    policy = LaggedMeanRecoveryPolicy(candidates=(0,))
    policy.choose(10)
    with pytest.raises(ValueError):
        observe(policy, 0, float("nan"))
    with pytest.raises(ValueError):
        policy.observe(0, 0, 1, 1, 1)
    assert policy.state.process_cnt == 0
