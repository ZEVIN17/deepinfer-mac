"""Timing bookkeeping contracts only; these are not performance measurements."""
from unittest.mock import patch

from deepinfer_mac.varlen_verify import StageTimer, execution_identity


def test_disabled_timer_never_reads_clock():
    with patch('deepinfer_mac.varlen_verify.time.perf_counter', side_effect=AssertionError):
        timer = StageTimer(False)
        timer.mark('ignored')
    assert timer.stages == {}


def test_intervals_are_disjoint_and_repeated_labels_accumulate():
    with patch('deepinfer_mac.varlen_verify.time.perf_counter', side_effect=[1, 2, 4, 7]):
        timer = StageTimer(True)
        timer.mark('forward')
        timer.mark('host')
        timer.mark('forward')
    assert timer.stages == {'forward': 4000, 'host': 2000}


def test_profile_identity_cannot_reuse_old_cost_key():
    assert execution_identity('bucket') == 'bucket'
    assert execution_identity('bucket', True) != 'bucket'
    assert execution_identity('bucket', True) != execution_identity('shared_cap', True)


def calibration_fixture():
    # Synthetic timing values are used exclusively for validator contracts.
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('pair_cost_guard', Path('benchmarks/experiments/phase3_pair_cost.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = {'stage': 'development', 'accepted': True, 'calibration_frozen': True,
             'source_sha256': {}, 'model_content_sha256': ['t', 'd'], 'calibration_identity': 'c',
             'execution_identity': {'draft': 'serial', 'recovery': 'exact_replay', 'profiling': False},
             'curves': [{'key': {'model_identity': 't', 'draft_identity': 'd',
                                'quantization': 'target6bit-g64/draft8bit-g64', 'batch_size': b,
                                'context_bucket': 512, 'execution_strategy': 'bucket', 'calibration_identity': 'c'},
                         'points': [[q, 1] for q in range(b, b * 4 + 1)]} for b in (1, 2)]}
    return module.validate_calibration, value


def test_calibration_requires_complete_unprofiled_identity():
    import copy

    import pytest
    validate, valid = calibration_fixture()
    validate(valid, {}, ['t', 'd'], 'c')
    mutations = [lambda x: x['execution_identity'].update(profiling=True),
                 lambda x: x.pop('execution_identity'),
                 lambda x: x['curves'].pop(),
                 lambda x: x['curves'].append(copy.deepcopy(x['curves'][0])),
                 lambda x: x['curves'][1]['key'].update(execution_strategy='bucket|profile=wall_v1'),
                 lambda x: x['curves'][0]['points'].pop(),
                 lambda x: x['curves'][0]['points'][0].__setitem__(1, float('nan'))]
    for mutate in mutations:
        invalid = copy.deepcopy(valid)
        mutate(invalid)
        with pytest.raises(ValueError):
            validate(invalid, {}, ['t', 'd'], 'c')
