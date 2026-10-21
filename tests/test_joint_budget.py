import importlib.util
from pathlib import Path

import pytest

from deepinfer_mac.ds.joint_budget import (
    Confidence,
    CostKey,
    EmpiricalConfidence,
    JointBudget,
    RequestKey,
    allocate_cpu,
    remap_keys,
)
from deepinfer_mac.varlen_verify import greedy_result, verification_groups


def c(values, kind="conditional", calibration="v1"):
    return Confidence(tuple(values), kind, "model", calibration, "mechanism_control")


def test_probability_semantics_and_rejection():
    assert c([.5, .6]).cumulative() == pytest.approx((.5, .3))
    assert allocate_cpu([c([.5, .3], "cumulative"), c([.28, 0], "cumulative")], [0, 1], 4, 2) == (3, 1)
    for values in ([float("nan")], [1.01], [-1]):
        with pytest.raises(ValueError):
            c(values)
    with pytest.raises(ValueError):
        c([.2, .3], "cumulative")


def test_empirical_censor_and_atomic_invalid_update():
    predictor = EmpiricalConfidence(4, "model", "v1")
    predictor.observe([True, True, None, None])
    predictor.observe([True, False, None, None])
    assert predictor.counts == [[2, 2], [1, 2], [0, 0], [0, 0]]
    with pytest.raises(ValueError):
        predictor.observe([True, False, True, None])
    assert predictor.counts[0] == [2, 2]
    assert predictor.predict().kind == "conditional"
    assert predictor.diagnostics()["coverage"] == [2, 2, 0, 0]
    assert predictor.diagnostics()["prequential_brier"][2:] == [None, None]


def test_uid_generation_fresh_and_tie():
    a, b, reused = RequestKey("a", 0), RequestKey("b", 0), RequestKey("a", 1)
    assert remap_keys([a, b], [b, reused, a]) == (1, -1, 0)
    assert allocate_cpu([c([0, 0])], [-1, 0], 3, 2) == (2, 1)
    assert allocate_cpu([], [], 0, 2) == ()
    with pytest.raises(ValueError):
        allocate_cpu([], [-2], 1, 2)
    with pytest.raises(ValueError):
        remap_keys([], [a, a])


def test_independent_exhaustive_reference():
    spec = importlib.util.spec_from_file_location("independent", Path(__file__).parent / "reference/phase3_budget_replay.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    import itertools
    for raw in itertools.product((0, .5, 1), repeat=4):
        values = [list(raw[:2]), list(raw[2:])]
        for remap in ([0, 1], [1, 0], [-1, 0], [-1, -1], [0, 0]):
            for q in range(2, 7):
                expected = reference.rank_count_reference(reference.cumulative_from_conditional(values, remap, 2), q)
                assert allocate_cpu([c(row) for row in values], remap, q, 2) == tuple(expected)


def test_history_isolation_cost_key_and_calibration():
    key = CostKey("model", "draft", "8bit", 2, 128, "bucket", "v1")
    scheduler = JointBudget(2, "model")
    a, b = RequestKey("a", 0), RequestKey("b", 0)
    scheduler.set_curve(key, [(2, 1), (6, 1.1)])
    scheduler.commit([(a, c([1, 1])), (b, c([1, 1]))])
    assert scheduler.choose([a, b], key) == (1, 1)  # N-2 still fresh
    scheduler.commit([(a, c([0, 0])), (b, c([1, 1]))])
    assert scheduler.choose([a, b], key) == (3, 3)  # N-2 picks full total
    assert scheduler.previous[a].values == (1, 1)
    with pytest.raises(ValueError):
        scheduler.commit([(a, c([1, 1], calibration="other"))])
    assert scheduler.latest[a].values == (0, 0)


def test_real_groups_and_greedy_censor():
    assert verification_groups([1, 3, 1], "bucket") == [(1, (0, 2)), (3, (1,))]
    assert verification_groups([1, 3, 1], "padded_control") == [(3, (0, 1, 2))]
    assert greedy_result(9, (1, 2, 3), [1, 7, 8, 9], 10, set()) == ((1, 7), (True, False, None))
    assert greedy_result(9, (1, 2, 3), [1, 2, 3, 4], 1, set()) == ((1,), (True, None, None))
    assert greedy_result(9, (1, 2, 3), [1, 2, 3, 4], 9, {2}) == ((1, 2), (True, True, None))


@pytest.mark.parametrize("invalid", [True, 0.5, -1, float("nan"), "1"])
def test_integer_identity_and_depth_contracts(invalid):
    with pytest.raises(ValueError):
        RequestKey("a", invalid)
    with pytest.raises(ValueError):
        JointBudget(invalid, "model")
    with pytest.raises(ValueError):
        EmpiricalConfidence(invalid, "model", "v1")
    with pytest.raises(ValueError):
        CostKey("model", "draft", "8bit", invalid, 128, "bucket", "v1")
    with pytest.raises(ValueError):
        CostKey("model", "draft", "8bit", 1, invalid, "bucket", "v1")


def test_confidence_is_immutable_numeric_not_boolean():
    values = [.5, .25]
    confidence = Confidence(values, "conditional", "model", "v1", "empirical")
    values[0] = 1
    assert confidence.values == (.5, .25)
    for values in ([True], ["0.5"], [None]):
        with pytest.raises(ValueError):
            c(values)


@pytest.mark.parametrize("remaining", [True, 1.5, 0, -1, float("nan")])
def test_remaining_output_budget_must_be_exact_integer(remaining):
    from deepinfer_mac.varlen_verify import VerifyRow
    with pytest.raises(ValueError):
        greedy_result(0, (1, 2), [1, 2, 3], remaining, set())
    with pytest.raises(ValueError):
        VerifyRow(RequestKey("a", 0), [], 0, (1, 2), remaining)


@pytest.mark.parametrize("token", [True, -1, .5, "1"])
def test_token_ids_are_explicit_nonnegative_integers(token):
    from deepinfer_mac.varlen_verify import VerifyRow
    with pytest.raises(ValueError):
        VerifyRow(RequestKey("a", 0), [], token, (), 1)
    with pytest.raises(ValueError):
        greedy_result(0, (), [token], 1, set())
