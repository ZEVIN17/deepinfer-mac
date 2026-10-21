"""Portable deterministic quality scoring; never loads a model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_checker_spec = importlib.util.spec_from_file_location('deepinfer_ifeval_checkers', ROOT / 'vendor' / 'ifeval_checkers.py')
ifeval_checkers = importlib.util.module_from_spec(_checker_spec)
_checker_spec.loader.exec_module(ifeval_checkers)
DEFAULT_TASKS = ROOT.parents[1] / 'results' / 'quality-cache' / 'tasks.json'
CHECKERS = {
    'punctuation:no_comma': ifeval_checkers.CommaChecker,
    'detectable_content:number_placeholders': ifeval_checkers.PlaceholderChecker,
    'detectable_format:number_bullet_lists': ifeval_checkers.BulletListChecker,
    'keywords:existence': ifeval_checkers.KeywordChecker,
    'detectable_format:json_format': ifeval_checkers.JsonFormat,
    'startend:end_checker': ifeval_checkers.EndChecker,
    'detectable_format:title': ifeval_checkers.TitleChecker,
}


def load_tasks(split=None, path=DEFAULT_TASKS):
    tasks = json.loads(Path(path).read_text())
    return [t for t in tasks if split is None or t['split'] == split]


def _run_code(code, case, timeout):
    """Process/time/resource isolation, not a security boundary for hostile code."""
    if case.get('testtype') != 'stdin':
        return {'status': 'unsupported_test_type', 'pass': None}
    launcher = (
        'import resource,sys\n'
        'resource.setrlimit(resource.RLIMIT_CPU,(3,3))\n'
        'resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))\n'
        'resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))\n'
        'source=open(sys.argv[1]).read()\n'
        'exec(compile(source,sys.argv[1],"exec"),{"__name__":"__main__"})\n'
    )
    with tempfile.TemporaryDirectory(prefix='deepinfer-quality-') as directory:
        path = Path(directory)
        (path / 'solution.py').write_text(code)
        with (path / 'stdout').open('w+') as stdout, (path / 'stderr').open('w+') as stderr:
            process = subprocess.Popen(
                [sys.executable, '-I', '-S', '-c', launcher, str(path / 'solution.py')],
                stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
                cwd=path, env={'PATH': '/usr/bin:/bin', 'PYTHONHASHSEED': '0'},
                start_new_session=True,
            )
            try:
                process.communicate(case['input'], timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if process.poll() is None:
                        process.kill()
                process.communicate()
                return {'status': 'timeout', 'pass': False}
            finally:
                # Stop children that survived the solution's main process.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            stdout.seek(0)
            stderr.seek(0)
            actual = stdout.read(1048576)
            errors = stderr.read(4096)
        passed = process.returncode == 0 and actual.split() == case['output'].split()
        return {'status': 'pass' if passed else 'runtime_or_output_failure', 'pass': passed,
                'returncode': process.returncode, 'actual': actual[:4096], 'expected': case['output'],
                'stderr': errors}


def score_task(task, output):
    scoring = task['scoring']
    kind = scoring['type']
    if not isinstance(output, str) or not output.strip():
        return {'pass': False, 'status': 'empty_output', 'details': {}}
    if kind == 'ifeval_strict':
        results = []
        for name, kwargs in zip(scoring['instruction_ids'], scoring['kwargs'], strict=True):
            if name not in CHECKERS:
                return {'pass': None, 'status': 'unsupported_instruction', 'details': {'name': name}}
            checker = CHECKERS[name]()
            checker.build_description(**kwargs)
            results.append({'instruction': name, 'pass': bool(checker.check_following(output))})
        passed = all(r['pass'] for r in results)
        return {'pass': passed, 'status': 'pass' if passed else 'constraint_failure', 'details': results}
    if kind == 'multiple_choice':
        # Predeclared deterministic answer extraction; no random guessing fallback.
        answers = re.findall(r'(?:the\s+)?answer\s+is\s*\(?([A-J])\)?', output, re.IGNORECASE)
        if not answers and re.fullmatch(r'\s*\(?[A-J]\)?[.\s]*', output, re.IGNORECASE):
            answers = re.findall(r'[A-J]', output, re.IGNORECASE)
        answer = answers[-1].upper() if answers else None
        passed = answer == scoring['answer']
        return {'pass': passed, 'status': 'pass' if passed else 'wrong_answer' if answer else 'unparseable_answer',
                'details': {'predicted': answer, 'expected': scoring['answer']}}
    if kind == 'python_stdin':
        blocks = re.findall(r'```(?:python|python3)?\s*\n(.*?)```', output, re.DOTALL | re.IGNORECASE)
        code = blocks[-1] if blocks else output.strip()
        results = [_run_code(code, case, scoring.get('timeout_seconds', 3)) for case in scoring['cases']]
        if any(result['pass'] is None for result in results):
            return {'pass': None, 'status': 'unsupported_test_type', 'details': results}
        passed = all(result['pass'] for result in results)
        return {'pass': passed, 'status': 'pass' if passed else 'test_failure', 'details': results}
    return {'pass': None, 'status': 'unsupported_scoring_type', 'details': {'type': kind}}


def wilson(passed, count):
    if not count:
        return None
    z = 1.959963984540054
    p = passed / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    spread = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - spread), min(1, center + spread)]


def score_responses(responses, tasks=None):
    tasks = load_tasks() if tasks is None else tasks
    by_id = {r['task_id']: r for r in responses}
    if len(by_id) != len(responses):
        raise ValueError('Duplicate response task_id; separate model/configuration runs before scoring.')
    rows = []
    groups = defaultdict(list)
    for task in tasks:
        response = by_id.get(task['id'])
        result = score_task(task, response['output']) if response is not None else {'pass': None, 'status': 'missing_response'}
        row = {'task_id': task['id'], 'benchmark': task['benchmark'], 'split': task['split'], **result}
        if response:
            row['finish_reason'] = response.get('finish_reason')
        rows.append(row)
        groups[task['benchmark']].append(row)
    summaries = {}
    for benchmark, rows_in_group in groups.items():
        scored = [r for r in rows_in_group if r['pass'] is not None]
        passed = sum(r['pass'] for r in scored)
        summaries[benchmark] = {'passed': passed, 'scored': len(scored), 'expected': len(rows_in_group),
                                'pass_rate': passed / len(scored) if scored else None,
                                'wilson_95': wilson(passed, len(scored)),
                                'complete': len(scored) == len(rows_in_group)}
    return {'schema_version': 1, 'scope': 'fixed local subsets; no full-benchmark/leaderboard claim',
            'results': rows, 'summary': summaries,
            'gates': {'all_expected_tasks_scored': all(v['complete'] for v in summaries.values()),
                      'no_combined_capability_score': True,
                      'quantization_equivalence_established': False}}


def score_model_report(report, tasks=None):
    """Group native/engine/sampling settings without pooling their samples."""
    tasks = load_tasks() if tasks is None else tasks
    groups = defaultdict(list)
    for case in report['cases']:
        if case['task_id'] not in {task['id'] for task in tasks}:
            continue
        output = case['output']
        if case.get('enable_thinking') or '<think>' in output:
            output = output.rsplit('</think>', 1)[1] if '</think>' in output else ''
        for marker in ('<|im_end|>', '<|endoftext|>'):
            output = output.replace(marker, '')
        groups[case.get('setting', 'default')].append(dict(case, output=output))
    summaries = {}
    for setting, cases in groups.items():
        scored = score_responses(cases, tasks)
        # Release-safe derived evidence: IDs/booleans, no source problem text,
        # expected test outputs, generated code, or full benchmark responses.
        summaries[setting] = {
            'summary': scored['summary'], 'gates': scored['gates'],
            'results': [{key: row[key] for key in ('task_id', 'benchmark', 'split', 'pass', 'status')} for row in scored['results']],
            'timings': [{'task_id': case['task_id'], 'elapsed_seconds': case.get('elapsed_seconds'),
                         'ttft_seconds': case.get('ttft_seconds'), 'finish_reason': case.get('finish_reason'),
                         'output_tokens': len(case.get('output_token_ids', []))} for case in cases],
            'configurations': [{key: case.get(key) for key in ('task_id', 'model', 'revision', 'quantization', 'sampling', 'seed', 'enable_thinking', 'max_tokens')} for case in cases],
        }
    return {'schema_version': 1, 'scope': 'fixed local subsets; no leaderboard/full-benchmark claim',
            'settings': summaries, 'manifest_sha256': hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest(),
            'provenance': {
                'profile': report.get('profile'), 'run_status': report.get('status'),
                'runtime_source_tree_sha256': report.get('source_start', {}).get('source_tree_sha256'),
                'runtime_source_unchanged_during_run': report.get('source_start') == report.get('source_end'),
                'frozen_source_tree_sha256': (report.get('frozen_source') or {}).get('tree_sha256'),
                'public_tasks_sha256': report.get('corpus', {}).get('public_tasks_sha256'),
                'additional_control': report.get('additional_control'),
            }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['score', 'summary', 'selftest'])
    parser.add_argument('--responses', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=['dev', 'heldout'])
    args = parser.parse_args()
    if args.command == 'selftest':
        selftest()
        return
    responses = json.loads(args.responses.read_text())
    if args.command == 'summary':
        report = score_model_report(responses, load_tasks(args.split))
    else:
        if isinstance(responses, dict):
            responses = responses['responses']
        report = score_responses(responses, load_tasks(args.split))
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


def selftest():
    for name, cls in CHECKERS.items():
        checker = cls()
        kwargs, positive, negative = {
            'punctuation:no_comma': ({}, 'hello world', 'hello, world'),
            'detectable_content:number_placeholders': ({'num_placeholders': 2}, '[one] [two]', '[one]'),
            'detectable_format:number_bullet_lists': ({'num_bullets': 2}, '* one\n- two', '* one'),
            'keywords:existence': ({'keywords': ['red', 'blue']}, 'red blue', 'red'),
            'detectable_format:json_format': ({}, '```json\n{"a":1}\n```', 'not json'),
            'startend:end_checker': ({'end_phrase': 'The end.'}, 'This is The end.', 'The end. extra'),
            'detectable_format:title': ({}, '<<A title>>', 'no title'),
        }[name]
        checker.build_description(**kwargs)
        assert checker.check_following(positive), name
        assert not checker.check_following(negative), name
    task = {'scoring': {'type': 'python_stdin', 'cases': [{'input': '2 3\n', 'output': '25', 'testtype': 'stdin'}]}}
    assert score_task(task, '```python\na,b=map(int,input().split())\nprint((a+b)**2)\n```')['pass']
    assert not score_task(task, 'print(0)')['pass']
    assert not score_task(task, 'while True: pass')['pass']
    unsupported = {'scoring': {'type': 'python_stdin', 'cases': [{'testtype': 'functional'}]}}
    assert score_task(unsupported, 'print(1)')['pass'] is None
    choice = {'scoring': {'type': 'multiple_choice', 'answer': 'B'}}
    assert score_task(choice, 'The answer is (B).')['pass']
    assert not score_task(choice, 'The answer is (A).')['pass']
    assert not score_task(choice, 'perhaps')['pass']
    print('quality scorer selftest: passed (7 official IFEval checkers, MCQ, subprocess pass/fail/timeout/unsupported)')


if __name__ == '__main__':
    main()
