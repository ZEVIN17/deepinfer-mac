"""Tiny real MLX execution controls; no pretrained model or performance claim."""
import copy

import pytest

pytestmark = pytest.mark.metal


def tiny(seed=7):
    import mlx.core as mx
    from mlx_lm.models.llama import Model, ModelArgs
    mx.random.seed(seed)
    model = Model(ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2,
                            intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                            rms_norm_eps=1e-5, vocab_size=32))
    mx.eval(model.parameters())
    return model


def rows_for(model):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from deepinfer_mac.ds.joint_budget import RequestKey
    from deepinfer_mac.varlen_verify import VerifyRow
    rows = []
    for i, prompt in enumerate(([1, 2], [3, 4, 5, 6], [2, 3])):
        cache = make_prompt_cache(model)
        model(mx.array([prompt[:-1]]), cache=cache)
        mx.eval([c.state for c in cache])
        rows.append(VerifyRow(RequestKey(str(i), 0), cache, prompt[-1], (1, 2, 3), 4))
    return rows


def test_device_rank_matches_independent_cpu_reference():
    # Compare to independently source-derived reference, not allocate_cpu.
    import importlib.util
    import itertools
    from pathlib import Path

    import mlx.core as mx

    from deepinfer_mac.ds.joint_budget import Confidence, allocate_mlx
    spec = importlib.util.spec_from_file_location("oracle", Path(__file__).parent / "reference/phase3_budget_replay.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

    def c(values):
        return Confidence(tuple(values), "conditional", "m", "c", "mechanism_control")
    for raw in itertools.product((0.0, 1.0), repeat=4):
        values = [list(raw[:2]), list(raw[2:])]
        for remap in ([0, 1], [-1, 0], [1, 0]):
            for q in (2, 3, 5, 6):
                actual = allocate_mlx([c(v) for v in values], remap, q, 2)
                mx.eval(actual)
                assert actual.tolist() == oracle.rank_count_reference(oracle.cumulative_from_conditional(values, remap, 2), q)
    cumulative = [Confidence((.5, .3), "cumulative", "m", "c", "mechanism_control"),
                  Confidence((.28, 0), "cumulative", "m", "c", "mechanism_control")]
    assert allocate_mlx(cumulative, [0, 1], 4, 2).tolist() == [3, 1]


def test_actual_dense_lengths_outputs_recovery_and_cancellation():
    import mlx.core as mx

    from deepinfer_mac.varlen_verify import MLXVariableVerifier
    model = tiny()
    original = rows_for(model)
    verifier = MLXVariableVerifier(model)
    bucket = verifier.verify(copy.deepcopy(original), [1, 3, 1])
    padded = verifier.verify(copy.deepcopy(original), [1, 3, 1], strategy="padded_control")
    assert bucket["outputs"] == padded["outputs"]
    assert bucket["executed_verify_q"] == 5
    assert padded["executed_verify_q"] == 9
    assert bucket["padded_q"] == 0 and padded["padded_q"] == 4
    assert [(c["batch"], c["sequence_length"]) for c in bucket["model_calls"] if c["phase"] == "verify"] == [(2, 1), (1, 3)]
    rows = copy.deepcopy(original)
    result = verifier.verify(rows, [1, 3, 1])
    for before, after, output in zip(original, rows, result["outputs"]):
        model(mx.array([[before.tail, *output[:-1]]]), cache=before.cache)
        expected = model(mx.array([[output[-1]]]), cache=before.cache)
        actual = model(mx.array([[output[-1]]]), cache=after.cache)
        assert mx.allclose(expected, actual, atol=1e-6).item()
    rows = rows_for(model)
    snapshots = [copy.deepcopy([c.state for c in r.cache]) for r in rows]
    count = 0

    def cancel():
        nonlocal count
        count += 1
        return count >= 6

    with pytest.raises(InterruptedError):
        verifier.verify(rows, [1, 3, 1], cancel=cancel)
    for row, states in zip(rows, snapshots):
        for cache, state in zip(row.cache, states):
            assert all(mx.array_equal(a, b).item() for a, b in zip(cache.state, state))


def test_real_distinct_draft_session_counts_complete_work():
    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget
    from deepinfer_mac.varlen_verify import JointSpeculativeSession
    target, draft = tiny(7), tiny(19)
    rows, draft_rows = rows_for(target), rows_for(draft)
    budget = JointBudget(3, "target")
    key = CostKey("target", "draft", "float32", 3, 16, "bucket", "test-only")
    session = JointSpeculativeSession(target, draft, budget, key, draft_identity="draft",
                                      calibration_identity="test-only", allocator="mlx")
    result = session.tick(rows, [r.cache for r in draft_rows])
    assert result["full_cycle_ms"] > 0 and result["draft_cost_included"]
    assert result["executed_total_q"] == sum(c["tokens"] for c in result["model_calls"])
    assert len([c for c in result["model_calls"] if c["phase"] == "draft"]) == 9
    assert result["allocator"] == "mlx" and result["draft_source"] == "external_model"
    assert set(budget.latest) == {r.key for r in rows}


def tiny_hybrid():
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    mx.random.seed(23)
    model = TextModel(TextModelArgs(model_type="qwen3_5_text", hidden_size=64,
                                    intermediate_size=128, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2,
                                    head_dim=16, vocab_size=32, linear_num_value_heads=2,
                                    linear_num_key_heads=2, linear_key_head_dim=16,
                                    linear_value_head_dim=16, full_attention_interval=2,
                                    rope_parameters={"type": "default", "rope_theta": 10000,
                                                     "partial_rotary_factor": 1.0}))
    mx.eval(model.parameters())
    return model


def assert_states_equal(rows, expected):
    import mlx.core as mx

    def equal(a, b):
        if isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif hasattr(a, "shape"):
            assert mx.array_equal(a, b).item()
        else:
            assert a == b
    for row, states in zip(rows, expected):
        equal([c.state for c in row.cache], states)


def test_actual_hybrid_single_rows_exact_recovery_eos_and_remaining():
    import mlx.core as mx

    from deepinfer_mac.varlen_verify import MLXVariableVerifier
    model = tiny_hybrid()
    rows = rows_for(model)
    original = copy.deepcopy(rows)
    assert any(type(c).__name__ == "ArraysCache" for c in rows[0].cache)
    # Establish first target token independently, then force EOS there.
    first_logits = model(mx.array([[rows[0].tail]]), cache=copy.deepcopy(rows[0].cache))
    rows[0].eos = frozenset({mx.argmax(first_logits[0, -1]).item()})
    rows[1].remaining = 1
    result = MLXVariableVerifier(model).verify(rows, [3, 4, 3])
    assert len(result["outputs"][0]) == len(result["outputs"][1]) == 1
    assert all(c["batch"] == 1 for c in result["model_calls"])
    for before, after, output in zip(original, rows, result["outputs"]):
        model(mx.array([[before.tail, *output[:-1]]]), cache=before.cache)
        expected = model(mx.array([[output[-1]]]), cache=before.cache)
        actual = model(mx.array([[output[-1]]]), cache=after.cache)
        assert mx.allclose(expected, actual, atol=1e-6).item()


@pytest.mark.parametrize("failure_stage", ["target", "draft_catchup"])
def test_actual_failed_forward_rolls_back_both_networks_and_history(failure_stage):
    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget
    from deepinfer_mac.varlen_verify import JointSpeculativeSession
    target, draft = tiny_hybrid(), tiny(19)
    rows, draft_rows = rows_for(target), rows_for(draft)
    expected_target = [copy.deepcopy([c.state for c in r.cache]) for r in rows]
    expected_draft = [copy.deepcopy([c.state for c in r.cache]) for r in draft_rows]

    class FailAfter:
        def __init__(self, model, count):
            self.model, self.count = model, count

        def __call__(self, *args, **kwargs):
            output = self.model(*args, **kwargs)
            self.count -= 1
            if not self.count:
                raise RuntimeError("injected after real forward")
            return output

    budget = JointBudget(3, "target")
    key = CostKey("target", "draft", "float32", 3, 16, "bucket", "test-only")
    session = JointSpeculativeSession(FailAfter(target, 2) if failure_stage == "target" else target,
                                      FailAfter(draft, 10) if failure_stage == "draft_catchup" else draft,
                                      budget, key, draft_identity="draft", calibration_identity="test-only")
    with pytest.raises(RuntimeError, match="injected"):
        session.tick(rows, [r.cache for r in draft_rows])
    assert_states_equal(rows, expected_target)
    assert_states_equal(draft_rows, expected_draft)
    assert budget.latest == budget.previous == session.predictors == {}


def test_actual_session_dynamic_uid_generation_and_cost_strategy():
    from dataclasses import replace

    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget, RequestKey
    from deepinfer_mac.varlen_verify import JointSpeculativeSession
    target, draft = tiny(), tiny(19)
    rows, draft_rows = rows_for(target), rows_for(draft)
    budget = JointBudget(3, "target")
    key = CostKey("target", "draft", "float32", 3, 16, "bucket", "test-only")
    session = JointSpeculativeSession(target, draft, budget, key, draft_identity="draft",
                                      calibration_identity="test-only", allocator="mlx")
    with pytest.raises(ValueError, match="strategy"):
        session.tick(rows, [r.cache for r in draft_rows], strategy="shared_cap", forced_lengths=[1, 2, 3])
    first = session.tick(rows, [r.cache for r in draft_rows], forced_lengths=[1, 2, 3])
    old_keys = list(budget.latest)
    for row, output in zip(rows, first["outputs"]):
        row.tail = output[-1]
        row.remaining -= len(output)
    # Complete/cancel rows 1/2; reuse UID 1 with a new generation and fresh caches.
    new_target, new_draft = rows_for(target)[1], rows_for(draft)[1]
    new_target.key = RequestKey("1", 1)
    survivor = rows[0]
    survivor.remaining = 4
    active = [new_target, survivor]
    result = session.tick(active, [new_draft.cache, draft_rows[0].cache],
                          cost_key=replace(key, batch_size=2))
    assert len(result["outputs"]) == 2
    assert list(budget.previous) == old_keys
    assert list(budget.latest) == [RequestKey("1", 1), RequestKey("0", 0)]
    assert set(session.predictors) == set(budget.latest)


def test_stage_profile_preserves_outputs_and_committed_cache():
    import mlx.core as mx

    from deepinfer_mac.varlen_verify import MLXVariableVerifier

    model = tiny()
    plain, profiled = rows_for(model), rows_for(model)
    reference = MLXVariableVerifier(model).verify(plain, (1, 2, 4))
    measured = MLXVariableVerifier(model).verify(profiled, (1, 2, 4), profiling=True)
    assert reference['outputs'] == measured['outputs']
    assert reference['model_calls'] == measured['model_calls']
    assert reference['verification_stages_ms'] is None
    assert all(value >= 0 for value in measured['verification_stages_ms'].values())
    for left, right in zip(plain, profiled):
        assert [c.offset for c in left.cache] == [c.offset for c in right.cache]
        token = mx.array([[reference['outputs'][int(left.key.uid)][-1]]])
        a, b = model(token, cache=left.cache), model(token, cache=right.cache)
        mx.eval(a, b)
        assert mx.array_equal(a, b).item()


def test_session_profile_cost_isolation_and_complete_partition():
    from dataclasses import replace

    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget
    from deepinfer_mac.varlen_verify import JointSpeculativeSession, execution_identity

    target, draft = tiny(), tiny(19)
    rows, draft_rows = rows_for(target), rows_for(draft)
    key = CostKey('target', 'draft', 'float32', 3, 16, 'bucket', 'test-only')
    session = JointSpeculativeSession(target, draft, JointBudget(3, 'target'), key,
                                      draft_identity='draft', calibration_identity='test-only', profiling=True)
    with pytest.raises(ValueError, match='strategy'):
        session.tick(rows, [r.cache for r in draft_rows])
    result = session.tick(rows, [r.cache for r in draft_rows], forced_lengths=[1, 2, 4],
                          cost_key=replace(key, execution_strategy=execution_identity('bucket', True)))
    assert result['profiling']
    stages = result['cycle_stages_ms']
    assert stages['verification_total'] >= sum(result['verification_stages_ms'].values())
    assert result['cycle_profile_residual_ms'] >= 0
    assert sum(stages.values()) + result['cycle_profile_residual_ms'] == pytest.approx(result['full_cycle_ms'])
    assert {'draft_forward_argmax_eval', 'draft_argmax_host', 'draft_catchup_eval'} <= stages.keys()
