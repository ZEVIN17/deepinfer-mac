"""Protect scientific pairing, aggregation, and release-safe evidence boundaries."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'quality_compare', Path(__file__).resolve().parents[1]/'benchmarks/quality/compare_models.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
compare = _module.compare


@pytest.fixture
def reports():
    source = {'source_tree_sha256': 'runtime-hash'}
    cases = [{'setting': 'greedy_engine', 'task_id': f'mmlu:{i}', 'benchmark': 'mmlu', 'split': 'dev',
                  'max_tokens': 100, 'seed': 42, 'enable_thinking': False, 'sampling': {'temperature': 0},
                  'finish_reason': 'stop' if i == 1 else 'length', 'output_token_ids': [1]*i,
                  'output': 'PRIVATE RAW RESPONSE', 'messages': [{'content': 'PRIVATE TASK'}],
                  'quality': {'pass': i == 1}, 'metrics': {'total_ms': i*10, 'peak_memory_bytes': i*100}}
             for i in (1, 2)]
    left = {'profile': 'left', 'status': 'measured', 'corpus': {'public_tasks_sha256': 'corpus-hash'},
                'frozen_source': {'tree_sha256': 'frozen-hash'}, 'source_start': source,
                'source_end': copy.deepcopy(source), 'cases': cases}
    right = copy.deepcopy(left)
    right['profile'] = 'right'
    right['cases'][0]['quality']['pass'] = False
    right['cases'][1]['quality']['pass'] = True
    right['cases'][1]['finish_reason'] = 'stop'
    return [left, right]


def test_real_pairing_completion_resource_aggregation_and_privacy(reports):
    result = compare(reports)
    left = result['summaries']['left']['greedy_engine/mmlu']
    assert (left['passed'], left['scored'], left['tasks']) == (1, 2, 2)
    assert (left['normal_stop'], left['length_stop'], left['normal_stop_rate']) == (1, 1, .5)
    assert result['summaries']['right']['greedy_engine/mmlu']['normal_stop_rate'] == 1
    assert left['diagnostic_resources_not_fair_performance']['peak_memory_bytes'] == {'median': 150, 'max': 200}
    assert 0 < left['wilson_95'][0] < .5 < left['wilson_95'][1] < 1
    pair = result['paired'][0]
    assert (pair['left_only_pass'], pair['right_only_pass']) == (1, 1)
    assert pair['exact_paired_discordance_p_unadjusted'] == 1
    assert pair['tasks'][0]['left_pass'] is True and pair['tasks'][0]['right_pass'] is False
    assert pair['tasks'][1]['left_finish'] == 'length'
    assert 'PRIVATE' not in json.dumps(result)
    assert result['quantization_equivalence_established'] is False


def test_duplicate_and_missing_cases_rejected(reports):
    reports[1]['cases'].append(copy.deepcopy(reports[1]['cases'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        compare(reports)
    reports[1]['cases'] = reports[1]['cases'][:1]
    with pytest.raises(ValueError, match='same settings/tasks'):
        compare(reports)


@pytest.mark.parametrize('field,value', [('max_tokens', 101), ('seed', 43), ('enable_thinking', True),
                                        ('sampling', {'temperature': .7}), ('split', 'heldout')])
def test_budget_seed_sampler_and_split_mismatch_rejected(reports, field, value):
    reports[1]['cases'][0][field] = value
    with pytest.raises(ValueError, match='Nonidentical'):
        compare(reports)


@pytest.mark.parametrize('field,key', [('corpus', 'public_tasks_sha256'), ('frozen_source', 'tree_sha256'),
                                      ('source_start', 'source_tree_sha256')])
def test_source_and_corpus_mismatch_rejected(reports, field, key):
    reports[1][field][key] = 'different'
    with pytest.raises(ValueError, match='identical frozen source'):
        compare(reports)


def test_setting_mismatch_rejected(reports):
    reports[1]['cases'][0]['setting'] = 'official_engine'
    with pytest.raises(ValueError, match='same settings/tasks'):
        compare(reports)
