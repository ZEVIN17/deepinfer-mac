"""Preregistered development-only budget extension; original scores stay intact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASK_IDS = {'mmlu:1471', 'mmlu:9423', 'lcb:abc387_c'}
SETTINGS = {'greedy_engine', 'official_engine'}
PROFILES = {'qwen3.5-9b-8bit', 'qwen3.5-9b-4bit', 'qwen3-14b'}


def analyze(originals, extended):
    original_map = {r['profile']: r for r in originals}
    if (len(originals) != 3 or len(extended) != 3 or set(original_map) != PROFILES
            or {r['profile'] for r in extended} != PROFILES):
        raise ValueError('Distinct matching original/extended model profiles required')
    all_reports = originals+extended
    if (len({r['source_start']['source_tree_sha256'] for r in all_reports}) != 1
            or len({r['corpus']['public_tasks_sha256'] for r in all_reports}) != 1
            or len({r['frozen_source']['tree_sha256'] for r in extended}) != 1):
        raise ValueError('All models require one unchanged runtime/corpus and one extended frozen source')
    rows = []
    for report in extended:
        original = original_map[report['profile']]
        if report['status'] != 'measured' or original['status'] != 'measured':
            raise ValueError('Both original and extended reports must be complete')
        if (report['source_start'] != report['source_end'] or original['source_start'] != original['source_end']
                or report['source_start']['source_tree_sha256'] != original['source_start']['source_tree_sha256']
                or report['corpus']['public_tasks_sha256'] != original['corpus']['public_tasks_sha256']):
            raise ValueError('Runtime/source stability or corpus mismatch')
        cases = {(c['setting'], c['task_id']): c for c in report['cases']}
        expected = {(setting, task) for setting in SETTINGS for task in TASK_IDS}
        if len(cases) != len(report['cases']) or set(cases) != expected:
            raise ValueError('Require all six preregistered setting/task pairs without duplicates')
        before = {(c['setting'], c['task_id']): c for c in original['cases']}
        if len(before) != len(original['cases']) or not expected <= set(before):
            raise ValueError('Original report has duplicate or missing setting/task pairs')
        for key, after in cases.items():
            old = before[key]
            if (after['split'] != 'dev' or old['split'] != 'dev' or after['max_tokens'] != 4096
                    or old['max_tokens'] >= 4096 or after['enable_thinking'] or after['seed'] != 42):
                raise ValueError('Only preregistered development4096 controls are accepted')
            if any(after[field] != old[field] for field in ('model', 'revision', 'sampling', 'seed', 'enable_thinking', 'messages', 'prompt_token_ids')):
                raise ValueError('Only the output budget may change; model/prompt/sampler controls differ')
            old_ids, new_ids = old['output_token_ids'], after['output_token_ids']
            shared = min(len(old_ids), len(new_ids))
            rows.append({'profile': report['profile'], 'setting': key[0], 'task_id': key[1], 'split': 'dev',
                         'original_budget': old['max_tokens'], 'extended_budget': 4096,
                         'original_tokens': len(old_ids), 'extended_tokens': len(new_ids),
                         'shared_prefix_tokens': shared, 'shared_prefix_exact': old_ids[:shared] == new_ids[:shared],
                         'full_original_sequence_is_prefix': new_ids[:len(old_ids)] == old_ids,
                         'same_complete_token_sequence': new_ids == old_ids,
                         'original_finish': old['finish_reason'], 'extended_finish': after['finish_reason'],
                         'original_pass': old['quality']['pass'], 'extended_pass': after['quality']['pass'],
                         'original_status': old['quality']['status'], 'extended_status': after['quality']['status'],
                         'original_seconds': old['elapsed_seconds'], 'extended_seconds': after['elapsed_seconds'],
                         'extended_peak_metal_bytes': after['metrics'].get('peak_memory_bytes'),
                         'extended_rss_bytes': after['metrics'].get('rss_bytes'),
                         'extended_system_swap_delta_bytes': after['metrics'].get('swap_delta_bytes'),
                         'sampling': after['sampling'], 'seed': after['seed']})
    return {'schema_version': 1, 'scope': 'Preregistered three-development-task budget sensitivity, not replacement primary27scores, held-out evidence, or framework performance.',
            'limits': ['Tasks follow exactly the initial9B8development cap-failure rule and were fixed before all additional4096-token runs; no held-out task selected.',
                       'All three models/two common settings use4096outputs; original strict scores remain unchanged.',
                       'Longer responses and their total elapsed cost are retained; timings are diagnostic, not a fair framework speed comparison.'],
            'planned_responses': 18, 'actual_responses': len(rows), 'results': rows,
            'runtime_source_tree_sha256': extended[0]['source_start']['source_tree_sha256'],
            'public_tasks_sha256': extended[0]['corpus']['public_tasks_sha256'],
            'original_frozen_trees': {r['profile']: r['frozen_source']['tree_sha256'] for r in originals},
            'extended_frozen_trees': {r['profile']: r['frozen_source']['tree_sha256'] for r in extended}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--originals', type=Path, nargs='+', required=True)
    parser.add_argument('--extended', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze([json.loads(p.read_text()) for p in args.originals], [json.loads(p.read_text()) for p in args.extended])
    result['raw_report_sha256'] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in args.originals+args.extended]
    result['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
