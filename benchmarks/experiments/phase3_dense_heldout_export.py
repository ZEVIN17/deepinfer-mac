"""Reproducible private-output redaction and prompt-clustered diagnostic summary."""
import argparse
import hashlib
import json
import statistics
from itertools import pairwise
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(value):
    return {'count': len(value), 'sha256': hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()}


def export(raw):
    public = {k: v for k, v in raw.items() if k not in ('cases', 'serial_quality_oracle', 'failures')}
    public['serial_quality_oracle'] = [tokens(out) for out in raw.get('serial_quality_oracle', [])]
    public['failures'] = [{'type': str(f.get('type', 'unknown')), 'detail_sha256':
                          hashlib.sha256(str(f.get('detail', '')).encode()).hexdigest()} for f in raw.get('failures', [])]
    public['cases'] = []
    for case in raw['cases']:
        record = {k: v for k, v in case.items() if k not in ('outputs', 'cycles')}
        record['outputs'] = [tokens(out) for out in case['outputs']]
        record['cycle_count'] = len(case['cycles'])
        record['decode_model_input_positions'] = sum(t['executed_total_q'] for t in case['cycles']) if case['cycles'] else None
        record['decode_model_submissions'] = sum(t['model_submissions'] for t in case['cycles']) if case['cycles'] else None
        record['active_batch_sizes'] = [len(t['request_ids']) for t in case['cycles']]
        public['cases'].append(record)
    return public


def summarize(reports):
    if len(reports) != 3 or sorted(r['round'] for r in reports) != [0, 1, 2]:
        raise ValueError('Exactly the three declared independent rounds are required')
    if any(r['source_sha256'] != reports[0]['source_sha256'] or r['protocol'] != reports[0]['protocol'] for r in reports):
        raise ValueError('Frozen source/protocol must match all rounds')
    controls = ('native_batch', 'old_replay', 'fast_trim')
    if any({c['control'] for c in r['cases']} != set(controls) for r in reports):
        raise ValueError('Every declared control must remain represented')
    result = {'schema_version': 1, 'scope': 'Three-process heldout diagnostic; two prompt clusters; no performance acceptance.',
              'accepted': all(r['accepted'] for r in reports), 'performance_claim': False,
              'functional_output_pass': all(r['functional_output_pass'] for r in reports),
              'process_count': len(reports), 'prompt_cluster_count': len(reports[0]['workload_manifest']['splits']['heldout']),
              'native_matched_requests': sum(r['gate_counts']['native_matched_requests'] for r in reports),
              'swap_growth_bytes': [r['swap_after_bytes'] - r['swap_before_bytes'] for r in reports],
              'metal_peak_bytes': [r['metal_peak_bytes'] for r in reports], 'controls': {}, 'prompt_clusters': []}
    for control in controls:
        cases = [next(c for c in r['cases'] if c['control'] == control) for r in reports]
        times = [c['complete_task_ms'] for c in cases]
        result['controls'][control] = {'complete_task_ms_by_process': times, 'median_complete_task_ms': statistics.median(times)}
    for i, task in enumerate(reports[0]['workload_manifest']['splits']['heldout']):
        cluster = {'id': task['id'], 'controls': {}}
        for control in controls:
            cases = [next(c for c in r['cases'] if c['control'] == control) for r in reports]
            arrivals = [c['arrival_ms'][i] for c in cases]
            cluster['controls'][control] = {
                'completion_ms_by_process': [c['request_completion_ms'][i] for c in cases],
                'ttft_ms_by_process': [a[0] for a in arrivals],
                'itl_ms_by_process': [[b - a for a, b in pairwise(arrival)] for arrival in arrivals]}
        result['prompt_clusters'].append(cluster)
    result['numeric_groups'] = {}
    for oracle in ('whole_prefix', 'cycle_partition'):
        result['numeric_groups'][oracle] = {}
        for control in ('old_replay', 'fast_trim'):
            states = [s[label][oracle] for r in reports for c in r['cases'] if c['control'] == control
                      for s in c['next_state'] for label in ('target', 'draft')]
            result['numeric_groups'][oracle][control] = {'checked': len(states), **{
                key: sum(s[key] for s in states) for key in ('allclose_1e5', 'argmax_equal', 'offsets_equal')},
                'max_abs_errors': [s['max_abs_error'] for s in states]}
    result['limits'] = ['Per-prompt TTFT/ITL/completion retain three process observations; tokens and cycles are not independent samples.',
                        'Speculative tokens arrive together at verified-cycle completion; zero within-cycle ITL is retained.',
                        'Only two prompts, fixed short ignored-EOS budgets and one dense small pair; not general serving or oMLX evidence.',
                        'Strict numerical tolerance is unchanged; no release promotion while its gate fails.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs=3, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(p.read_text()) for p in args.inputs]
    public_reports = []
    for path, raw in zip(args.inputs, reports):
        out = args.output_dir / f"phase3-dense-heldout-round{raw['round']}.json"
        if out.exists():
            raise FileExistsError(out)
        public = export(raw)
        public['raw_report_sha256'] = sha(path)
        out.write_text(json.dumps(public, indent=2) + '\n')
        public_reports.append({'file': out.name, 'sha256': sha(out)})
    summary = summarize(reports)
    summary['reports'] = public_reports
    out = args.output_dir / 'phase3-dense-heldout-summary.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(summary, indent=2) + '\n')
