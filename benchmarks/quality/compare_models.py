"""Release-safe paired quality comparison; raw responses never enter the output."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

RUNNER_PATH = 'benchmarks/experiments/phase2_model_quality.py'


def metadata_audit_patch(script):
    """The entire permitted bridge: no generation/scoring edits can hide here."""
    replacements = [
        ('    from mlx_lm.generate import generate_step\n',
         '    from mlx_lm.generate import generate_step\n    from mlx_lm.models.cache import make_prompt_cache\n'),
        ("        report['cache_types'] = sorted({type(c).__name__ for c in engine.model.make_cache()})",
         "        report['cache_types'] = sorted({type(c).__name__ for c in make_prompt_cache(engine.model)})"),
        ("        report['status'] = 'measured'\n    finally:",
         "        report['status'] = 'measured'\n    except BaseException as exc:\n        report['status'] = 'failed'\n        report['error'] = f'{type(exc).__name__}: {exc}'\n        raise\n    finally:"),
    ]
    for old, new in replacements:
        if script.count(old) != 1:
            raise ValueError('Harness bridge does not match the exact permitted metadata patch')
        script = script.replace(old, new)
    return script


def validate_bridge(bridge, reports):
    if bridge.get('allowed_file') != RUNNER_PATH:
        raise ValueError('Harness bridge may only change runner audit metadata')
    before, after = bridge['before_files'], bridge['after_files']
    changed = {p for p in set(before) | set(after) if before.get(p) != after.get(p)}
    if changed != {RUNNER_PATH}:
        raise ValueError('Harness bridge changes runtime, scoring, or other unapproved files')
    if metadata_audit_patch(bridge['before_script']) != bridge['after_script']:
        raise ValueError('Harness bridge includes unapproved generation/sampling/scoring changes')
    accepted = {}
    for prefix, files in [('before', before), ('after', after)]:
        script_hash = hashlib.sha256(bridge[f'{prefix}_script'].encode()).hexdigest()
        tree_hash = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if (files[RUNNER_PATH] != script_hash or bridge[f'{prefix}_script_sha256'] != script_hash
                or bridge[f'{prefix}_tree_sha256'] != tree_hash):
            raise ValueError('Harness bridge content/hash mismatch')
        accepted[tree_hash] = files
    for report in reports:
        frozen = report['frozen_source']
        if frozen['tree_sha256'] not in accepted or frozen['files'] != accepted[frozen['tree_sha256']]:
            raise ValueError('Report is outside the verified harness bridge')
    return {'allowed_file': RUNNER_PATH, 'before_tree_sha256': bridge['before_tree_sha256'],
            'after_tree_sha256': bridge['after_tree_sha256'],
            'before_script_sha256': bridge['before_script_sha256'], 'after_script_sha256': bridge['after_script_sha256'],
            'manifest_sha256': hashlib.sha256(json.dumps(bridge, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'scope': 'Only cache-type audit factory and exception-status recording differ; generation/scoring/runtime unchanged.'}


def wilson(passed, total):
    if not total:
        return None
    z = 1.959963984540054
    rate = passed / total
    denominator = 1 + z*z/total
    center = (rate + z*z/(2*total)) / denominator
    margin = z*math.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    return [max(0, center-margin), min(1, center+margin)]


def summarize(cases):
    scored = [c for c in cases if c['quality'].get('pass') is not None]
    passed = sum(c['quality']['pass'] is True for c in scored)
    stopped = sum(c['finish_reason'] == 'stop' for c in cases)
    metrics = {}
    for key in ('ttft_ms', 'total_ms', 'decode_tps', 'peak_memory_bytes', 'rss_bytes', 'swap_delta_bytes'):
        values = [c['metrics'][key] for c in cases if c['metrics'].get(key) is not None]
        metrics[key] = {'median': statistics.median(values), 'max': max(values)} if values else None
    return {'passed': passed, 'scored': len(scored), 'tasks': len(cases),
            'wilson_95': wilson(passed, len(scored)), 'normal_stop': stopped,
            'length_stop': sum(c['finish_reason'] == 'length' for c in cases),
            'normal_stop_rate': stopped/len(cases) if cases else None,
            'diagnostic_resources_not_fair_performance': metrics}


def compare(reports, harness_bridge=None):
    if len(reports) < 2:
        raise ValueError('At least two completed distinct model reports are required')
    if any(report['status'] != 'measured' for report in reports):
        raise ValueError('All comparison reports must have completed, including failures')
    if len({r['profile'] for r in reports}) != len(reports):
        raise ValueError('One completed report per distinct profile is required')
    public_sha = {r.get('corpus', {}).get('public_tasks_sha256') for r in reports}
    source_sha = {(r.get('frozen_source') or {}).get('tree_sha256') for r in reports}
    runtime_sha = {r.get('source_start', {}).get('source_tree_sha256') for r in reports}
    if (len(public_sha) != 1 or None in public_sha or None in source_sha
            or len(runtime_sha) != 1 or None in runtime_sha
            or any(r.get('source_start') != r.get('source_end') for r in reports)):
        raise ValueError('Shared comparisons require identical frozen source and public corpus')
    bridge_evidence = validate_bridge(harness_bridge, reports) if harness_bridge else None
    if len(source_sha) != 1 and bridge_evidence is None:
        raise ValueError('Shared comparisons require identical frozen source or a verified metadata-only harness bridge')
    indexed = []
    for report in reports:
        if not report['cases']:
            raise ValueError('Empty case lists cannot establish paired quality')
        cases = {(c['setting'], c['task_id']): c for c in report['cases']}
        if len(cases) != len(report['cases']):
            raise ValueError('Duplicate setting/task keys must not be silently collapsed')
        indexed.append(cases)
    if any(set(indexed[0]) != set(item) for item in indexed[1:]):
        raise ValueError('Every model must complete the same settings/tasks')
    for key in indexed[0]:
        controls = {(c[key]['max_tokens'], c[key]['seed'], c[key]['enable_thinking'],
                     c[key]['benchmark'], c[key]['split'],
                     json.dumps(c[key]['sampling'], sort_keys=True)) for c in indexed}
        if len(controls) != 1:
            raise ValueError(f'Nonidentical budget/seed/sampling controls: {key}')
    groups = sorted({(c['setting'], c['benchmark']) for c in reports[0]['cases']})
    summaries = {}
    for report in reports:
        summaries[report['profile']] = {
            f'{setting}/{benchmark}': summarize([c for c in report['cases']
                                                if c['setting'] == setting and c['benchmark'] == benchmark])
            for setting, benchmark in groups}
    pairs = []
    for (left, a), (right, b) in itertools.combinations(zip(reports, indexed), 2):
        for setting, benchmark in groups:
            rows = []
            for key, first in a.items():
                if first['setting'] != setting or first['benchmark'] != benchmark:
                    continue
                second = b[key]
                rows.append({'task_id': key[1], 'split': first['split'],
                             'max_tokens': first['max_tokens'], 'seed': first['seed'],
                             'enable_thinking': first['enable_thinking'], 'sampling': first['sampling'],
                             'left_pass': first['quality'].get('pass'), 'right_pass': second['quality'].get('pass'),
                             'left_finish': first['finish_reason'], 'right_finish': second['finish_reason'],
                             'left_tokens': len(first['output_token_ids']), 'right_tokens': len(second['output_token_ids'])})
            left_only = sum(r['left_pass'] is True and r['right_pass'] is False for r in rows)
            right_only = sum(r['left_pass'] is False and r['right_pass'] is True for r in rows)
            discordant = left_only + right_only
            p = min(1, 2*sum(math.comb(discordant, k) for k in range(min(left_only, right_only)+1))/2**discordant) if discordant else 1
            pairs.append({'left': left['profile'], 'right': right['profile'], 'setting': setting,
                          'benchmark': benchmark, 'left_only_pass': left_only, 'right_only_pass': right_only,
                          'exact_paired_discordance_p_unadjusted': p, 'tasks': rows})
    return {'schema_version': 1, 'protocol': 'shared_qwen35_protocol',
            'scope': 'Small fixed subsets, finite output budgets; LCB public tests only. No aggregate capability ranking.',
            'limitations': ['Wide small-sample uncertainty; nonsignificance does not establish quantization equivalence.',
                            'Diagnostic timings/resources are not a fair framework performance comparison.',
                            'Length stops are finite-budget completion failures, not demonstrated intrinsic inability.',
                            'Paired p-values are descriptive, unadjusted for multiple comparisons.'],
            'frozen_source_tree_sha256': next(iter(source_sha)) if len(source_sha) == 1 else None,
            'frozen_source_tree_sha256_values': sorted(source_sha), 'harness_bridge': bridge_evidence,
            'public_tasks_sha256': next(iter(public_sha)),
            'additional_controls': {r['profile']: r.get('additional_control') for r in reports},
            'summaries': summaries, 'paired': pairs, 'quantization_equivalence_established': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--harness-bridge', type=Path, help='Explicit audited metadata-only source bridge; default requires identical archives')
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    bridge = json.loads(args.harness_bridge.read_text()) if args.harness_bridge else None
    output = compare(reports, harness_bridge=bridge)
    output['raw_report_sha256'] = {r['profile']: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for r, p in zip(reports, args.reports)}
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
