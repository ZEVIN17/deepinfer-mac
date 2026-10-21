"""Uniform offline Markdown-only MMLU sensitivity; never replaces strict scores."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location('public_quality_scorer', Path(__file__).with_name('runner.py'))
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
load_tasks, score_task = _runner.load_tasks, _runner.score_task


def normalize_markdown(text):
    return re.sub(r'[*_`]', '', text)


def final_text(text):
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    return text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()


def analyze(reports, tasks):
    by_id = {t['id']: t for t in tasks}
    rows = []
    grouped = defaultdict(list)
    for report in reports:
        if report['status'] != 'measured':
            raise ValueError('Use completed reports only')
        for case in report['cases']:
            if case['benchmark'] != 'mmlu':
                continue
            task = by_id[case['task_id']]
            text = final_text(case['output'])
            strict = score_task(task, text)
            if strict['pass'] != case['quality']['pass']:
                raise ValueError('Current strict scorer disagrees with the retained score')
            normalized = score_task(task, normalize_markdown(text))
            row = {'profile': report['profile'], 'setting': case['setting'],
                   'sampling_family': report.get('sampling_family', 'shared_qwen35_protocol'),
                   'task_id': case['task_id'], 'split': case['split'],
                   'strict_pass': strict['pass'], 'strict_status': strict['status'],
                   'markdown_normalized_pass': normalized['pass'],
                   'markdown_normalized_status': normalized['status'],
                   'finish_reason': case['finish_reason'], 'output_tokens': len(case['output_token_ids']),
                   'max_tokens': case['max_tokens'], 'sampling': case['sampling'], 'seed': case['seed']}
            rows.append(row)
            grouped[(report['profile'], case['setting'])].append(row)
    summaries = [{'profile': key[0], 'setting': key[1], 'tasks': len(items),
                  'strict_passed': sum(r['strict_pass'] is True for r in items),
                  'markdown_normalized_passed': sum(r['markdown_normalized_pass'] is True for r in items),
                  'changed_task_ids': [r['task_id'] for r in items if r['strict_pass'] != r['markdown_normalized_pass']]}
                 for key, items in grouped.items()]
    return {'schema_version': 1,
            'scope': 'Post-observation offline format sensitivity, uniformly applied to every retained MMLU response; not a new generation or replacement official benchmark score.',
            'normalization': 'Delete only Markdown emphasis/code delimiter characters *, _, and backtick before the unchanged answer regex.',
            'input_response_count': sum(len(report['cases']) for report in reports),
            'mmlu_response_count': len(rows),
            'other_benchmark_responses_unmodified': sum(len(report['cases']) for report in reports)-len(rows),
            'limitations': ['The task requested an exact final-answer format; strict format failures remain failures in the primary report.',
                            'This isolates parser/format sensitivity from answer-letter correctness; no cross-protocol aggregate ranking.'],
            'summaries': summaries, 'results': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.reports]
    result = analyze(reports, load_tasks())
    result['raw_report_sha256'] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reports]
    result['scorer_sha256'] = hashlib.sha256((Path(__file__).parent/'runner.py').read_bytes()).hexdigest()
    result['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
