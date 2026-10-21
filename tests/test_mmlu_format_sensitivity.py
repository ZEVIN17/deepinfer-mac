"""Markdown diagnostics must not silently rewrite the original strict scores."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'mmlu_format_analysis', Path(__file__).resolve().parents[1]/'benchmarks/quality/mmlu_format_sensitivity.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def fixture_report(text, strict_pass=False):
    return {'status': 'measured', 'profile': 'fixture', 'cases': [
        {'task_id': 'mmlu:1', 'benchmark': 'mmlu', 'split': 'dev', 'setting': 'native',
         'output': text, 'quality': {'pass': strict_pass}, 'finish_reason': 'stop',
         'output_token_ids': [1], 'max_tokens': 100, 'sampling': {}, 'seed': 42}]}


TASKS = [{'id': 'mmlu:1', 'scoring': {'type': 'multiple_choice', 'answer': 'A'}}]


def test_correct_markdown_answer_changes_only_sensitivity_not_original():
    report = fixture_report('The answer is **A**.<|im_end|>')
    output = _module.analyze([report], TASKS)
    row = output['results'][0]
    assert row['strict_pass'] is False and row['markdown_normalized_pass'] is True
    assert report['cases'][0]['quality']['pass'] is False
    assert output['mmlu_response_count'] == output['input_response_count'] == 1
    assert 'The answer' not in str(output)


def test_normalization_does_not_turn_wrong_or_missing_letter_correct():
    for text in ('The answer is **B**.', 'No final selection provided.'):
        output = _module.analyze([fixture_report(text)], TASKS)
        assert output['results'][0]['markdown_normalized_pass'] is False


def test_strict_score_drift_rejected():
    with pytest.raises(ValueError, match='disagrees with the retained'):
        _module.analyze([fixture_report('The answer is (A).')], TASKS)


def test_other_benchmarks_are_counted_without_normalizing_constraints():
    report = fixture_report('The answer is (A).', strict_pass=True)
    report['cases'].append({'benchmark': 'ifeval', 'output': '**must remain unchanged**'})
    output = _module.analyze([report], TASKS)
    assert output['input_response_count'] == 2
    assert output['mmlu_response_count'] == output['other_benchmark_responses_unmodified'] == 1
