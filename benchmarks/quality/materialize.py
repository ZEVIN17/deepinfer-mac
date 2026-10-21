"""Materialize a fixed, outcome-independent official benchmark subset."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parents[1] / 'results' / 'quality-cache'
CACHE = DATA_ROOT / 'sources'
SEED = 'deepinfer-phase2-quality-v1'
SOURCES = {
    'ifeval': {'repo': 'google/IFEval', 'revision': '966cd89545d6b6acfd7638bc708b98261ca58e84', 'file': 'ifeval_input_data.jsonl', 'local': 'ifeval.jsonl', 'license': 'Apache-2.0'},
    'mmlu': {'repo': 'TIGER-Lab/MMLU-Pro', 'revision': 'b189ec765aa7ed75c8acfea42df31fdae71f97be', 'file': 'data/test-00000-of-00001.parquet', 'local': 'mmlu.parquet', 'license': 'MIT'},
    'lcb': {'repo': 'livecodebench/code_generation_lite', 'revision': '0fe84c3912ea0c4d4a78037083943e8f0c4dd505', 'file': 'test6.jsonl', 'local': 'lcb-v6.jsonl', 'license': 'dataset metadata: cc (variant unspecified); original problem rights retained'},
}
EXPECTED_SOURCE_SHA256 = {'ifeval': '6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2', 'mmlu': '0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8', 'lcb': 'bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5'}
SUPPORTED = {
    'punctuation:no_comma', 'detectable_content:number_placeholders',
    'detectable_format:number_bullet_lists', 'keywords:existence',
    'detectable_format:json_format', 'startend:end_checker', 'detectable_format:title',
}


def digest(value):
    return hashlib.sha256((SEED + ':' + str(value)).encode()).hexdigest()


def fetch(source):
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / source['local']
    if not target.exists():
        url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}"
        with urllib.request.urlopen(url, timeout=180) as response:
            target.write_bytes(response.read())
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != EXPECTED_SOURCE_SHA256[next(key for key, value in SOURCES.items() if value == source)]:
        raise ValueError(f'Source SHA256 mismatch: {target}')
    return target


def task(source_key, row, sample_id, prompt, scoring, max_tokens, split):
    source = dict(SOURCES[source_key], sample_id=str(sample_id))
    source.pop('local')
    return {'id': f'{source_key}:{sample_id}', 'benchmark': source_key, 'split': split,
            'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens,
            'enable_thinking': False, 'scoring': scoring, 'source': source,
            'metadata': {k: row[k] for k in ('category', 'difficulty', 'contest_date', 'question_title') if k in row}}


def main():
    from pyarrow import parquet

    rows = [json.loads(line) for line in fetch(SOURCES['ifeval']).read_text().splitlines()]
    eligible = sorted((r for r in rows if set(r['instruction_id_list']) <= SUPPORTED), key=lambda r: digest(r['key']))
    selected = []
    for family in sorted(SUPPORTED):
        chosen = next(r for r in eligible if family in r['instruction_id_list'] and r not in selected)
        selected.append(chosen)
    selected.append(next(r for r in eligible if r not in selected))
    tasks = [task('ifeval', r, r['key'], r['prompt'], {'type': 'ifeval_strict', 'instruction_ids': r['instruction_id_list'], 'kwargs': r['kwargs']}, 768, 'dev' if i % 2 == 0 else 'heldout') for i, r in enumerate(selected)]

    rows = parquet.read_table(fetch(SOURCES['mmlu'])).to_pylist()
    for i, category in enumerate(sorted({r['category'] for r in rows})):
        r = min((r for r in rows if r['category'] == category), key=lambda r: digest(r['question_id']))
        choices = '\n'.join(f'{chr(65+j)}. {option}' for j, option in enumerate(r['options']))
        prompt = f"The following is a multiple choice question about {category}.\n\n{r['question']}\n{choices}\n\nThink step by step and finish with exactly: The answer is (X). Replace X with the correct option letter."
        tasks.append(task('mmlu', r, r['question_id'], prompt, {'type': 'multiple_choice', 'answer': r['answer'], 'num_options': len(r['options'])}, 1024, 'dev' if i % 2 == 0 else 'heldout'))

    # First two easy and first two medium stdin tasks in the pinned v6 increment,
    # chosen before any model output; do not deserialize private pickle payloads.
    selected_lcb = []
    counts = {'easy': 0, 'medium': 0}
    with fetch(SOURCES['lcb']).open() as stream:
        for line in stream:
            r = json.loads(line)
            difficulty = r['difficulty']
            cases = json.loads(r['public_test_cases'])
            if difficulty in counts and counts[difficulty] < 2 and not r.get('starter_code') and all(c['testtype'] == 'stdin' for c in cases):
                counts[difficulty] += 1
                selected_lcb.append(r)
            if counts == {'easy': 2, 'medium': 2}:
                break
    for i, r in enumerate(selected_lcb):
        prompt = r['question_content'] + '\n\nWrite a complete Python 3 solution reading standard input and writing standard output. Put the solution in one ```python code block.'
        tasks.append(task('lcb', r, r['question_id'], prompt, {'type': 'python_stdin', 'cases': json.loads(r['public_test_cases']), 'test_scope': 'official_public_tests_only', 'timeout_seconds': 3}, 2048, 'dev' if i % 2 == 0 else 'heldout'))

    repair_row = next(r for r in selected_lcb if r['question_id'] == 'abc387_a')
    prompt = repair_row['question_content'] + '\n\nFix this incorrect Python solution. Return the full corrected program in one ```python block.\n```python\na, b = map(int, input().split())\nprint((a + b) * 2)\n```'
    repair = task('lcb', repair_row, 'abc387_a:repair', prompt, {'type': 'python_stdin', 'cases': json.loads(repair_row['public_test_cases']), 'test_scope': 'official_public_tests_only', 'timeout_seconds': 3}, 1024, 'heldout')
    repair['benchmark'] = 'auxiliary_repair'
    repair['metadata']['modification'] = 'Project-authored bug injection; derived from LCB task; excluded from LCB generation aggregate.'
    tasks.append(repair)

    (DATA_ROOT / 'tasks.json').write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + '\n')
    manifest = {'schema_version': 1, 'seed': SEED, 'selection_before_model_outputs': True,
                'selection': 'IFEval first hash-ranked eligible task per supported family plus one; MMLU first hash-ranked ID per category; LCB first two easy and first two medium stdin tasks in test6 file order; alternating dev/heldout within each benchmark; repair follows its source task heldout split.',
                'sources': {name: dict(src, sha256=hashlib.sha256(fetch(src).read_bytes()).hexdigest()) for name, src in SOURCES.items()},
                'tasks': [{k: t[k] for k in ('id', 'benchmark', 'split', 'source', 'metadata', 'max_tokens')} for t in tasks],
                'task_file_sha256': hashlib.sha256((DATA_ROOT / 'tasks.json').read_bytes()).hexdigest()}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Materialized {len(tasks)} tasks; sha256={manifest["task_file_sha256"]}')


if __name__ == '__main__':
    main()
