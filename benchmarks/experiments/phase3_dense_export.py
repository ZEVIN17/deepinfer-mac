"""Export source-bound dense development summaries without raw generated tokens."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def export(raw):
    result = {k: raw[k] for k in ('schema_version', 'stage', 'depth', 'source_sha256', 'workload_manifest',
              'models', 'versions', 'platform', 'available_before_bytes', 'required_admission_bytes',
              'swap_before_bytes', 'swap_after_bytes', 'metal_peak_bytes', 'source_unchanged', 'coverage',
              'accepted', 'performance_claim', 'process_count', 'scope', 'failures')}
    result['native'] = [{k: n[k] for k in ('arrival_ms', 'complete_task_ms')} |
                        {'output_count': len(n['tokens']), 'output_sha256': digest(n['tokens'])} for n in raw['native']]
    result['cases'] = []
    for case in raw['cases']:
        item = {k: case[k] for k in ('draft_execution', 'recovery', 'strategy', 'allocation', 'arrival_ms',
                'prefill_ms', 'complete_task_ms', 'next_state', 'matches_native', 'budget_complete', 'cost_key') if k in case}
        item.update(outputs=[{'count': len(x), 'sha256': digest(x)} for x in case['outputs']],
                    cycle_count=len(case['cycles']),
                    executed_model_input_positions=sum(t['executed_total_q'] for t in case['cycles']),
                    model_submissions=sum(t['model_submissions'] for t in case['cycles']),
                    active_batch_sizes=[len(t['request_ids']) for t in case['cycles']])
        result['cases'].append(item)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads(args.input.read_text())
    public = export(raw)
    public['raw_report_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(public, indent=2) + '\n')
