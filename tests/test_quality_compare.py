"""Protect scientific pairing, aggregation, and release-safe evidence boundaries."""
import copy
import hashlib
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


@pytest.fixture
def bridged(reports):
    before_script = ("    from mlx_lm.generate import generate_step\n"
                     "        report['cache_types'] = sorted({type(c).__name__ for c in engine.model.make_cache()})\n"
                     "        report['status'] = 'measured'\n    finally:\n")
    after_script = _module.metadata_audit_patch(before_script)
    bridge = {'allowed_file': _module.RUNNER_PATH, 'before_script': before_script, 'after_script': after_script}
    for prefix, script, report in zip(('before', 'after'), (before_script, after_script), reports):
        digest = hashlib.sha256(script.encode()).hexdigest()
        files = {_module.RUNNER_PATH: digest, 'src/runtime.py': 'same', 'benchmarks/quality/runner.py': 'same-scorer'}
        tree = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        bridge.update({f'{prefix}_files': files, f'{prefix}_tree_sha256': tree, f'{prefix}_script_sha256': digest})
        report['frozen_source'] = {'files': copy.deepcopy(files), 'tree_sha256': tree}
    return reports, bridge


def test_explicit_exact_metadata_bridge_is_required_and_disclosed(bridged):
    reports, bridge = bridged
    with pytest.raises(ValueError, match='verified metadata-only harness bridge'):
        compare(reports)
    result = compare(reports, harness_bridge=bridge)
    assert result['frozen_source_tree_sha256'] is None
    assert len(result['frozen_source_tree_sha256_values']) == 2
    assert result['harness_bridge']['before_tree_sha256'] != result['harness_bridge']['after_tree_sha256']
    assert result['paired'][0]['left_only_pass'] == 1


@pytest.mark.parametrize('path', ['src/runtime.py', 'benchmarks/quality/runner.py'])
def test_bridge_rejects_runtime_or_scorer_changes(bridged, path):
    reports, bridge = bridged
    bridge['after_files'][path] = 'changed'
    with pytest.raises(ValueError, match='runtime, scoring'):
        compare(reports, harness_bridge=bridge)


@pytest.mark.parametrize('extra', ['\ntemperature = .8\n', '\nscore = True\n'])
def test_bridge_rejects_sampler_or_scoring_edits_inside_runner(bridged, extra):
    reports, bridge = bridged
    bridge['after_script'] += extra
    with pytest.raises(ValueError, match='unapproved generation/sampling/scoring'):
        compare(reports, harness_bridge=bridge)


def test_bridge_cannot_bless_unlisted_report_source(bridged):
    reports, bridge = bridged
    reports[1]['frozen_source']['files']['src/runtime.py'] = 'unlisted'
    with pytest.raises(ValueError, match='outside the verified'):
        compare(reports, harness_bridge=bridge)
