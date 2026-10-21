"""Budget controls must preserve all18cases and distinguish prefix/cap effects."""
import copy
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'budget_analysis', Path(__file__).resolve().parents[1]/'benchmarks/quality/dev_budget_sensitivity.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.fixture
def reports():
    originals, extended = [], []
    for profile in sorted(_module.PROFILES):
        cases = []
        for setting in sorted(_module.SETTINGS):
            for task in sorted(_module.TASK_IDS):
                cases.append({'setting': setting, 'task_id': task, 'model': profile, 'revision': 'pin',
                              'split': 'dev', 'max_tokens': 1024, 'sampling': {}, 'seed': 42,
                              'enable_thinking': False, 'messages': [{'content': 'PRIVATE'}], 'prompt_token_ids': [4],
                              'output_token_ids': [1, 2], 'finish_reason': 'length', 'elapsed_seconds': 1,
                              'quality': {'pass': False, 'status': 'unparseable_answer'}, 'metrics': {}})
        old = {'profile': profile, 'status': 'measured', 'cases': cases,
               'source_start': {'source_tree_sha256': 'runtime'}, 'source_end': {'source_tree_sha256': 'runtime'},
               'corpus': {'public_tasks_sha256': 'corpus'}, 'frozen_source': {'tree_sha256': 'old'}}
        new = copy.deepcopy(old)
        new['frozen_source']['tree_sha256'] = 'extended'
        for case in new['cases']:
            case.update(max_tokens=4096, output_token_ids=[1, 2, 3], finish_reason='stop', elapsed_seconds=2)
            case['quality'] = {'pass': True, 'status': 'pass'}
        originals.append(old)
        extended.append(new)
    return originals, extended


def test_all18_prefixes_budgets_quality_and_longer_cost_retained(reports):
    output = _module.analyze(*reports)
    assert output['planned_responses'] == output['actual_responses'] == 18
    assert all(row['full_original_sequence_is_prefix'] and row['shared_prefix_exact'] for row in output['results'])
    assert all(row['original_pass'] is False and row['extended_pass'] is True for row in output['results'])
    assert all(row['extended_seconds'] > row['original_seconds'] for row in output['results'])
    assert 'PRIVATE' not in str(output)
    assert reports[0][0]['cases'][0]['quality']['pass'] is False


def test_divergent_prefix_is_disclosed_without_dropping_failure(reports):
    reports[1][0]['cases'][0]['output_token_ids'] = [8, 9]
    reports[1][0]['cases'][0]['quality']['pass'] = False
    row = _module.analyze(*reports)['results'][0]
    assert row['shared_prefix_exact'] is False and row['full_original_sequence_is_prefix'] is False
    assert row['extended_pass'] is False


def test_missing_model_case_and_changed_sampler_rejected(reports):
    with pytest.raises(ValueError, match='Distinct matching'):
        _module.analyze(reports[0], reports[1][:-1])
    reports[1][0]['cases'][0]['sampling'] = {'temperature': .7}
    with pytest.raises(ValueError, match='Only the output budget'):
        _module.analyze(*reports)
    reports[1][0]['cases'].pop()
    with pytest.raises(ValueError, match='all six preregistered'):
        _module.analyze(*reports)


def test_original_heldout_case_cannot_be_relabeled_development(reports):
    reports[0][0]['cases'][0]['split'] = 'heldout'
    with pytest.raises(ValueError, match='Only preregistered development'):
        _module.analyze(*reports)
