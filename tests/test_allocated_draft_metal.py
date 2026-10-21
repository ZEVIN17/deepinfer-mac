"""Actual variable-depth draft controls, including rollback and censored history."""
import copy

import pytest
from test_dense_fastpath_metal import assert_next_state
from test_varlen_metal import assert_states_equal, rows_for, tiny

from deepinfer_mac.ds.joint_budget import CostKey, JointBudget
from deepinfer_mac.varlen_verify import JointSpeculativeSession, execution_identity

pytestmark = pytest.mark.metal


def session_for(target, draft, mode, recovery, strategy, policy):
    key = CostKey('t', 'd', 'float32', 3, 16, execution_identity(
        strategy, draft_execution=mode, recovery=recovery, draft_budget=policy), 'test')
    return JointSpeculativeSession(target, draft, JointBudget(3, 't'), key,
                                   draft_identity='d', calibration_identity='test',
                                   draft_execution=mode, recovery=recovery, draft_budget=policy)


@pytest.mark.parametrize('mode', ['serial', 'batched'])
@pytest.mark.parametrize('recovery', ['exact_replay', 'dense_trim'])
@pytest.mark.parametrize('strategy', ['bucket', 'padded_control', 'shared_cap'])
def test_allocated_draft_matches_full_budget_state(mode, recovery, strategy):
    target, draft = tiny(), tiny(19)
    initial, initial_draft = rows_for(target), rows_for(draft)
    records = []
    for policy in ('full', 'allocated'):
        rows, drafts = copy.deepcopy(initial), copy.deepcopy(initial_draft)
        session = session_for(target, draft, mode, recovery, strategy, policy)
        result = session.tick(rows, [r.cache for r in drafts], strategy=strategy, forced_lengths=(1, 4, 2))
        for before, before_draft, row, draft_row, output in zip(initial, initial_draft, rows, drafts, result['outputs']):
            assert_next_state(target, before, row.cache, output)
            assert_next_state(draft, before_draft, draft_row.cache, output)
        expected = 9 if policy == 'full' or strategy == 'shared_cap' else 4
        assert result['executed_draft_q'] == expected
        if policy == 'allocated' and strategy != 'shared_cap':
            assert session.predictors[rows[0].key].diagnostics()['coverage'] == [0, 0, 0]
            assert session.predictors[rows[2].key].diagnostics()['coverage'][1:] == [0, 0]
        records.append(result)
    assert records[0]['outputs'] == records[1]['outputs']


@pytest.mark.parametrize('mode', ['serial', 'batched'])
def test_all_zero_proposals_preserves_pending_tail(mode):
    target, draft = tiny(), tiny(7)
    rows, drafts = rows_for(target), rows_for(draft)
    initial, initial_draft = copy.deepcopy(rows), copy.deepcopy(drafts)
    session = session_for(target, draft, mode, 'dense_trim', 'bucket', 'allocated')
    result = session.tick(rows, [r.cache for r in drafts], forced_lengths=(1, 1, 1))
    assert result['executed_draft_q'] == 0
    assert len(result['outputs']) == 3
    for before, before_draft, row, draft_row, output in zip(initial, initial_draft, rows, drafts, result['outputs']):
        assert len(output) == 1
        assert_next_state(target, before, row.cache, output)
        assert_next_state(draft, before_draft, draft_row.cache, output)


@pytest.mark.parametrize('mode', ['serial', 'batched'])
def test_cancel_after_allocated_draft_rolls_back(mode):
    target, draft = tiny(), tiny(19)
    rows, drafts = rows_for(target), rows_for(draft)
    states = [copy.deepcopy([c.state for c in r.cache]) for r in rows]
    draft_states = [copy.deepcopy([c.state for c in r.cache]) for r in drafts]
    calls = 0

    class Wrapped:
        def __call__(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return draft(*args, **kwargs)

    session = session_for(target, Wrapped(), mode, 'exact_replay', 'bucket', 'allocated')
    with pytest.raises(InterruptedError):
        session.tick(rows, [r.cache for r in drafts], forced_lengths=(1, 4, 2), cancel=lambda: calls > 1)
    assert_states_equal(rows, states)
    assert_states_equal(drafts, draft_states)
    assert session.predictors == session.budget.latest == session.budget.previous == {}
