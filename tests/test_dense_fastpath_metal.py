"""Tiny real device recovery controls; independent execution is required for acceptance."""
import copy
from dataclasses import replace

import pytest
from test_varlen_metal import assert_states_equal, rows_for, tiny, tiny_hybrid

pytestmark = pytest.mark.metal


def session_for(target, draft, batch, *, depth=3, strategy='bucket', draft_execution='serial', recovery='exact_replay'):
    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget
    from deepinfer_mac.varlen_verify import JointSpeculativeSession, execution_identity
    key = CostKey('t', 'd', 'float32', batch, 16,
                  execution_identity(strategy, draft_execution=draft_execution, recovery=recovery), 'test-only')
    return JointSpeculativeSession(target, draft, JointBudget(depth, 't'), key,
                                   draft_identity='d', calibration_identity='test-only',
                                   draft_execution=draft_execution, recovery=recovery), key


def assert_next_state(model, initial, actual, output):
    import mlx.core as mx
    expected = copy.deepcopy(initial.cache)
    model(mx.array([[initial.tail, *output[:-1]]]), cache=expected)
    assert [c.offset for c in expected] == [c.offset for c in actual]
    # Compare only logical consumed prefixes, not allocation capacity or stale tail.
    for a, b in zip(expected, actual):
        assert mx.allclose(a.keys[..., :a.offset, :], b.keys[..., :b.offset, :], atol=2e-6).item()
        assert mx.allclose(a.values[..., :a.offset, :], b.values[..., :b.offset, :], atol=2e-6).item()
    a = model(mx.array([[output[-1]]]), cache=expected)
    b = model(mx.array([[output[-1]]]), cache=copy.deepcopy(actual))
    mx.eval(a, b)
    assert mx.allclose(a, b, atol=2e-6).item()
    assert mx.argmax(a, -1).tolist() == mx.argmax(b, -1).tolist()


@pytest.mark.parametrize('strategy', ['bucket', 'padded_control', 'shared_cap'])
@pytest.mark.parametrize('draft_execution', ['serial', 'batched'])
def test_dense_trim_matches_same_draft_replay_and_next_state(strategy, draft_execution):
    target, draft = tiny(), tiny(19)
    original, original_draft = rows_for(target), rows_for(draft)
    for rows in (original, original_draft):
        rows[0].remaining = 1
    records = []
    for recovery in ('exact_replay', 'dense_trim'):
        rows, draft_rows = copy.deepcopy(original), copy.deepcopy(original_draft)
        ids = [[id(c) for c in r.cache] for r in rows + draft_rows]
        session, _ = session_for(target, draft, 3, strategy=strategy,
                                 draft_execution=draft_execution, recovery=recovery)
        result = session.tick(rows, [r.cache for r in draft_rows], strategy=strategy, forced_lengths=(1, 4, 2))
        assert ids == [[id(c) for c in r.cache] for r in rows + draft_rows]
        for before, before_draft, row, draft_row, output in zip(original, original_draft, rows, draft_rows, result['outputs']):
            assert_next_state(target, before, row.cache, output)
            assert_next_state(draft, before_draft, draft_row.cache, output)
        assert result['model_submissions'] == len(result['model_calls'])
        if recovery == 'dense_trim':
            assert result['executed_replay_q'] == 0
        assert len([c for c in result['model_calls'] if c['phase'] == 'draft']) == (3 if draft_execution == 'batched' else 9)
        records.append(result)
    assert records[0]['outputs'] == records[1]['outputs']


@pytest.mark.parametrize('depth', [0, 3])
@pytest.mark.parametrize('early_stop', ['full', 'eos', 'remaining'])
def test_full_accept_and_truncated_prefix_draft_catchup(depth, early_stop):
    # Distinct objects with identical parameters force acceptance without enabling
    # same-object mechanism shortcuts; these are explicitly tiny mechanics only.
    target, draft = tiny(7), tiny(7)
    original, draft_original = rows_for(target)[:1], rows_for(draft)[:1]
    if early_stop == 'remaining':
        original[0].remaining = 1
    if early_stop == 'eos':
        import mlx.core as mx
        cache = copy.deepcopy(original[0].cache)
        token = mx.argmax(target(mx.array([[original[0].tail]]), cache=cache)[0, -1]).item()
        original[0].eos = frozenset({token})
    rows, draft_rows = copy.deepcopy(original), copy.deepcopy(draft_original)
    session, _ = session_for(target, draft, 1, depth=depth, draft_execution='batched', recovery='dense_trim')
    result = session.tick(rows, [r.cache for r in draft_rows], forced_lengths=(depth + 1,))
    output = result['outputs'][0]
    assert len(output) == (depth + 1 if early_stop == 'full' else 1)
    assert_next_state(target, original[0], rows[0].cache, output)
    assert_next_state(draft, draft_original[0], draft_rows[0].cache, output)
    catches = [c for c in result['model_calls'] if c['phase'] == 'draft_catchup']
    assert len(catches) == int(len(output) > depth)
    assert all(c['tokens'] == 1 for c in catches)


@pytest.mark.parametrize("strategy", ["bucket", "padded_control", "shared_cap"])
def test_empty_tick_is_noop_and_cost_identity_guard(strategy):
    target, draft = tiny(), tiny(19)
    session, key = session_for(target, draft, 0, strategy=strategy, draft_execution='batched', recovery='dense_trim')
    before = copy.deepcopy((session.budget.previous, session.budget.latest, session.predictors))
    result = session.tick([], [], strategy=strategy)
    assert result['model_submissions'] == result['executed_total_q'] == 0
    assert result['outputs'] == []
    assert before == (session.budget.previous, session.budget.latest, session.predictors)
    with pytest.raises(ValueError, match='strategy'):
        session.tick([], [], strategy=strategy, cost_key=replace(key, execution_strategy='bucket'))


@pytest.mark.parametrize('draft_execution,recovery', [('batched', 'exact_replay'), ('serial', 'dense_trim')])
@pytest.mark.parametrize('hybrid_side', ['both', 'target', 'draft'])
def test_hybrid_rejects_unsupported_opt_in(draft_execution, recovery, hybrid_side):
    target = tiny_hybrid() if hybrid_side in ('both', 'target') else tiny()
    draft = tiny_hybrid() if hybrid_side in ('both', 'draft') else tiny(19)
    rows, drafts = rows_for(target), rows_for(draft)
    session, _ = session_for(target, draft, 3, draft_execution=draft_execution, recovery=recovery)
    with pytest.raises(ValueError, match='dense'):
        session.tick(rows, [r.cache for r in drafts])


@pytest.mark.parametrize('failure', ['draft', 'verify', 'catchup', 'cancel'])
def test_failure_rolls_back_every_original_cache_and_history(failure):
    target, draft = tiny(), tiny(7)
    rows, drafts = rows_for(target), rows_for(draft)
    target_states = [copy.deepcopy([c.state for c in r.cache]) for r in rows]
    draft_states = [copy.deepcopy([c.state for c in r.cache]) for r in drafts]
    calls = {'draft': 0, 'target': 0}

    class Wrapped:
        def __init__(self, model, name):
            self.model, self.name = model, name
        def __call__(self, *args, **kwargs):
            result = self.model(*args, **kwargs)
            calls[self.name] += 1
            if ((failure == 'draft' and self.name == 'draft' and calls['draft'] == 2)
                    or (failure == 'verify' and self.name == 'target')
                    or (failure == 'catchup' and self.name == 'draft' and calls['draft'] == 4)):
                raise RuntimeError('injected after real model call')
            return result
    session, _ = session_for(Wrapped(target, 'target'), Wrapped(draft, 'draft'), 3,
                             draft_execution='batched', recovery='dense_trim')
    cancel = (lambda: calls['target'] > 0) if failure == 'cancel' else None
    with pytest.raises((RuntimeError, InterruptedError)):
        session.tick(rows, [r.cache for r in drafts], forced_lengths=(4, 4, 4), cancel=cancel)
    assert_states_equal(rows, target_states)
    assert_states_equal(drafts, draft_states)
    assert session.budget.previous == session.budget.latest == session.predictors == {}


def test_multiple_rounds_row_reorder_and_reused_slot():
    from deepinfer_mac.ds.joint_budget import RequestKey
    target, draft = tiny(), tiny(19)
    rows, drafts = rows_for(target), rows_for(draft)
    session, key = session_for(target, draft, 3, draft_execution='batched', recovery='dense_trim')
    for _ in range(3):
        before, before_draft = copy.deepcopy(rows), copy.deepcopy(drafts)
        result = session.tick(rows, [r.cache for r in drafts], forced_lengths=(1, 2, 4))
        for initial, initial_draft, row, draft_row, output in zip(before, before_draft, rows, drafts, result['outputs']):
            assert_next_state(target, initial, row.cache, output)
            assert_next_state(draft, initial_draft, draft_row.cache, output)
            row.tail = output[-1]
        rows.reverse()
        drafts.reverse()
    fresh, fresh_draft = rows_for(target)[0], rows_for(draft)[0]
    fresh.key = RequestKey('0', 1)
    active, caches = [fresh, rows[0]], [fresh_draft.cache, drafts[0].cache]
    # Make the surviving UID distinct even when previous row order changes.
    assert active[0].key != active[1].key
    session.tick(active, caches, cost_key=replace(key, batch_size=2))
    assert set(session.budget.latest) == {r.key for r in active}
    assert set(session.predictors) == set(session.budget.latest)
