"""Portable checks for answer parsing and avoiding benchmark-harness false scores."""
import runpy
from pathlib import Path

import pytest

QUALITY = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'benchmarks/experiments/phase2_model_quality.py'))


def test_thinking_opened_in_prompt_and_unfinished_output():
    answer = QUALITY['final_answer']
    assert answer('Reasoning already opened by template</think>\n391', thinking=True) == '391'
    assert answer('Only unfinished reasoning', thinking=True) is None
    assert answer('<think>Reasoning</think>\n391') == '391'
    assert answer('391') == '391'


def test_code_harness_supports_pure_builtins_and_distinguishes_own_limits():
    task = {'kind': 'code', 'function': 'f', 'tests': [{'args': [3], 'expected': 3.0}]}
    score = QUALITY['score_smoke']
    assert score(task, 'def f(x):\n    return float(x)')['passed'] is True
    assert score(task, 'def f(x):\n    return nonexistent_symbol(x)')['passed'] is False
    blocked = score(task, 'def f(x):\n    return open(x)')
    assert blocked['passed'] is None and blocked['status'] == 'harness_error'


def test_public_raw_output_cannot_enter_distributable_reports(tmp_path):
    output = QUALITY['quality_output_path']
    assert output('model', True).parts[:2] == ('results', 'quality-cache')
    assert output('model', True, tmp_path / 'results/quality-cache/raw.json').name == 'raw.json'
    with pytest.raises(ValueError, match='Public raw reports'):
        output('model', True, tmp_path / 'benchmarks/reports/raw.json')
