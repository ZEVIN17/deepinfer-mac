from dataclasses import replace

import pytest

from deepinfer_mac.cycle_cost import CycleCostTable, CycleShape, ExecutionPolicy
from deepinfer_mac.ds.joint_budget import CostKey


def shape():
    return CycleShape(CostKey('target', 'draft', 'q4', 2, 64, 'bucket'), ('KVCache',),
                      (32, 64), (('draft', 2, 1), ('verify', 2, 4)))


def test_same_total_tokens_different_bucket_shape_never_reuses_cost():
    table = CycleCostTable()
    first = shape()
    second = replace(first, calls=(('draft', 2, 1), ('verify', 1, 4), ('verify', 1, 4)))
    for _ in range(3):
        table.observe(first, 10, 4)
    assert table.estimate(first)['ms_per_emitted_token'] == 2.5
    assert table.estimate(second) is None
    assert table.estimate(replace(first, contexts=(64, 32))) is None
    assert table.estimate(replace(first, identity=replace(first.identity, quantization='q8'))) is None


def test_observation_storage_is_bounded_and_old_samples_expire():
    table = CycleCostTable(max_keys=2, window=3)
    first = shape()
    for value in (1000, 10, 10, 10):
        table.observe(first, value, 1)
    assert table.estimate(first) == {'samples': 3, 'full_cycle_ms': 10, 'ms_per_emitted_token': 10}
    for context in (65, 66):
        table.observe(replace(first, contexts=(32, context)), 1, 1)
    assert len(table.samples) == 2 and table.estimate(first) is None


@pytest.mark.parametrize('elapsed,tokens', [(float('nan'), 1), (0, 1), (1, 0), (True, 1)])
def test_invalid_cost_not_committed(elapsed, tokens):
    table = CycleCostTable()
    with pytest.raises(ValueError):
        table.observe(shape(), elapsed, tokens)
    assert not table.samples


def test_native_cold_start_quality_gate_hysteresis_and_identity_reset():
    policy = ExecutionPolicy()
    assert policy.choose('model-a', {'joint': 1}) == 'native_batch'
    for _ in range(3):
        assert policy.choose('model-a', {'joint': 1, 'native_batch': 10}) == 'native_batch'
    costs = {'joint': 5, 'native_batch': 10}
    assert policy.choose('model-a', costs, quality_admitted={'joint'}) == 'native_batch'
    assert policy.choose('model-a', costs, quality_admitted={'joint'}) == 'joint'
    assert policy.choose('model-a', {'joint': 10, 'native_batch': 9}, quality_admitted={'joint'}) == 'joint'
    assert policy.choose('model-b', costs, quality_admitted={'joint'}) == 'native_batch'
    assert policy.choose('model-b', costs, quality_admitted={'joint'}) == 'joint'
    assert policy.choose('model-b', costs) == 'native_batch'  # Quality admission revoked.
