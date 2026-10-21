"""Independent bounded dense fast-path probe; explicit --execute and GPU baton required."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'benchmarks/experiments/phase3_dense_workload_manifest.json'
MODEL_PATHS = [ROOT / 'results/phase3-models' / n for n in (
    'smollm2-360m-instruct-6bit', 'smollm2-135m-instruct-8bit')]
IDENTITIES = ['40e44f0e35876a3e846c856f830ecca4208db73319256ff4f493434d2723950d',
              'a4f7f71502edccf61c70717a7edb5368266a104f0c45cc1c0ed72cfb7ea27981']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = [Path(__file__), MANIFEST, ROOT / 'uv.lock', ROOT / 'pyproject.toml',
             *sorted((ROOT / 'src/deepinfer_mac').rglob('*.py'))]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def load_workload(path):
    manifest = json.loads(MANIFEST.read_text())
    if sha(path) != manifest['workload_sha256']:
        raise ValueError('Workload differs from predeclared manifest')
    data = json.loads(path.read_text())
    for split in ('development', 'heldout'):
        observed = [{'id': x['id'], 'context_tokens': x['context_tokens'],
                     'prompt_sha256': hashlib.sha256(x['prompt'].encode()).hexdigest()} for x in data[split]]
        if observed != manifest['splits'][split]:
            raise ValueError('Split differs from manifest')
    return manifest, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--workload', type=Path, default=ROOT / 'artifacts/phase3-dense-fastpath/workloads-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, choices=(1, 2, 3), default=3)
    # Heldout intentionally unavailable until the development result is frozen.
    args = parser.parse_args()
    manifest, data = load_workload(args.workload)
    if not args.execute:
        print(json.dumps({'execution': False, 'manifest': manifest, 'depth': args.depth,
                          'requires_root_gpu_baton': True}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    import mlx.core as mx
    import psutil
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    from deepinfer_mac.adapters import ModelAdapter, inspect_snapshot
    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget, RequestKey
    from deepinfer_mac.models import verify_pair
    from deepinfer_mac.varlen_verify import JointSpeculativeSession, VerifyRow, execution_identity

    audits = [inspect_snapshot(p) for p in MODEL_PATHS]
    if [a['content_sha256'] for a in audits] != IDENTITIES:
        raise ValueError('Model content identity mismatch')
    if not all(verify_pair(*MODEL_PATHS)['explicit_thinking_template_tokens_equal'].values()):
        raise ValueError('Pair template mismatch')
    reserve = 3 * 2 * sum(ModelAdapter({}).memory_estimate(a['config'], 1024) for a in audits)
    required = sum(a['weight_bytes'] for a in audits) + 2 * 1024**3 + reserve
    available, swap = psutil.virtual_memory().available, psutil.swap_memory().used
    if required > available * .85 or reserve > available * .65:
        raise MemoryError('Conservative pair admission failed')
    report = {'schema_version': 1, 'stage': 'development', 'depth': args.depth,
              'source_sha256': sources(), 'workload_manifest': manifest,
              'models': [{'content_sha256': a['content_sha256'], 'weight_bytes': a['weight_bytes'],
                          'quantization': a['config'].get('quantization')} for a in audits],
              'versions': {p: version(p) for p in ('mlx', 'mlx-lm')}, 'platform': platform.mac_ver()[0],
              'available_before_bytes': available, 'required_admission_bytes': required,
              'swap_before_bytes': swap, 'cases': [], 'native': [], 'failures': [], 'accepted': False,
              'performance_claim': False, 'process_count': 1,
              'scope': 'Development mechanism only; simultaneous paired arrival, cold KV, resident warmed pair; not API E2E.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    save()
    deadline = time.monotonic() + 300

    def guard():
        if time.monotonic() > deadline:
            raise TimeoutError('Five-minute invocation limit')
        if psutil.swap_memory().used > swap or reserve > psutil.virtual_memory().available * .65:
            raise MemoryError('Swap growth or future reserve guard')

    try:
        begin = time.perf_counter()
        target, tokenizer = load(MODEL_PATHS[0])
        draft, _ = load(MODEL_PATHS[1])
        mx.eval(target.parameters(), draft.parameters())
        report['resident_pair_load_ms'] = (time.perf_counter() - begin) * 1000

        def prefill(model, tokens):
            cache = make_prompt_cache(model)
            model(mx.array([tokens[:-1]], dtype=mx.uint32), cache=cache)
            mx.eval([c.state for c in cache])
            return cache

        begin = time.perf_counter()
        for model in (target, draft):
            cache = prefill(model, [1, 2, 3, 4])
            del cache
        report['warmup_ms'] = (time.perf_counter() - begin) * 1000
        mx.reset_peak_memory()
        inputs = []
        for task in data['development']:
            question = tokenizer.encode(task['prompt'])
            padding = tokenizer.encode(data['padding_text'] * 150)
            if len(question) >= task['context_tokens']:
                raise ValueError('Question exceeds fixed context')
            inputs.append(padding[:task['context_tokens'] - len(question)] + question)
        report['input_token_counts'] = [len(x) for x in inputs]
        for tokens, count in zip(inputs, data['output_tokens']):
            guard()
            begin = time.perf_counter()
            output, arrivals = [], []
            for token, _ in generate_step(mx.array(tokens, dtype=mx.uint32), target, max_tokens=count):
                output.append(int(token))
                arrivals.append((time.perf_counter() - begin) * 1000)
            report['native'].append({'tokens': output, 'arrival_ms': arrivals,
                                     'complete_task_ms': (time.perf_counter() - begin) * 1000})
        calibration = manifest['workload_sha256']
        for draft_mode, recovery in (('serial', 'exact_replay'), ('batched', 'exact_replay'),
                                     ('serial', 'dense_trim'), ('batched', 'dense_trim')):
            policies = [('bucket', 'full'), ('shared_cap', 'full')]
            if (draft_mode, recovery) == ('batched', 'dense_trim'):
                policies += [('bucket', 'heterogeneous'), ('shared_cap', 'heterogeneous')]
            for strategy, allocation in policies:
                guard()
                case = {'draft_execution': draft_mode, 'recovery': recovery, 'strategy': strategy,
                        'allocation': allocation, 'outputs': [[] for _ in inputs], 'arrival_ms': [[] for _ in inputs],
                        'cycles': [], 'next_state': []}
                report['cases'].append(case)

                def key(batch, strategy=strategy, draft_mode=draft_mode, recovery=recovery):
                    return CostKey(IDENTITIES[0], IDENTITIES[1], 'target6bit-g64/draft8bit-g64', batch, 512,
                                   execution_identity(strategy, draft_execution=draft_mode, recovery=recovery), calibration)

                begin = time.perf_counter()
                session = JointSpeculativeSession(target, draft, JointBudget(args.depth, IDENTITIES[0]), key(len(inputs)),
                                                 draft_identity=IDENTITIES[1], calibration_identity=calibration,
                                                 allocator='mlx', draft_execution=draft_mode, recovery=recovery)
                rows = [VerifyRow(RequestKey(str(i), 0), prefill(target, tokens), tokens[-1], (), count)
                        for i, (tokens, count) in enumerate(zip(inputs, data['output_tokens']))]
                caches = [prefill(draft, tokens) for tokens in inputs]
                case['prefill_ms'] = (time.perf_counter() - begin) * 1000
                completed = []
                while rows:
                    guard()
                    tick = session.tick(rows, caches, strategy=strategy,
                                        forced_lengths=tuple(1 if allocation == 'heterogeneous' and row.key.uid == '0' else args.depth + 1 for row in rows),
                                        cost_key=key(len(rows)))
                    tick['request_ids'] = [row.key.uid for row in rows]
                    case['cycles'].append(tick)
                    elapsed = (time.perf_counter() - begin) * 1000
                    kept_rows, kept_caches = [], []
                    for row, cache, emitted in zip(rows, caches, tick['outputs']):
                        i = int(row.key.uid)
                        if not emitted or len(emitted) > row.remaining:
                            raise ValueError('Empty emission or budget overshoot')
                        case['outputs'][i].extend(emitted)
                        case['arrival_ms'][i].extend([elapsed] * len(emitted))
                        row.remaining -= len(emitted)
                        row.tail = emitted[-1]
                        if row.remaining:
                            kept_rows.append(row)
                            kept_caches.append(cache)
                        else:
                            completed.append((row, cache))
                    rows, caches = kept_rows, kept_caches
                case['complete_task_ms'] = (time.perf_counter() - begin) * 1000
                # Independent full serial replay oracle, outside timed task.
                for row, cache in completed:
                    i = int(row.key.uid)
                    consumed = inputs[i] + case['outputs'][i][:-1]
                    checks = {'request_id': row.key.uid}
                    for label, model, actual_cache in (('target', target, row.cache), ('draft', draft, cache)):
                        oracle_cache = prefill(model, consumed + [row.tail])
                        expected = model(mx.array([[row.tail]], dtype=mx.uint32), cache=oracle_cache)
                        actual = model(mx.array([[row.tail]], dtype=mx.uint32), cache=actual_cache)
                        mx.eval(expected, actual)
                        checks[label] = {'argmax_equal': mx.argmax(expected, axis=-1).tolist() == mx.argmax(actual, axis=-1).tolist(),
                                         'logits_allclose_1e5': bool(mx.allclose(expected, actual, atol=1e-5, rtol=1e-5).item()),
                                         'max_abs_error': float(mx.max(mx.abs(expected - actual)).item()),
                                         'offsets_equal': [c.offset for c in actual_cache] == [c.offset for c in oracle_cache]}
                    case['next_state'].append(checks)
                case['matches_native'] = [x == n['tokens'] for x, n in zip(case['outputs'], report['native'])]
                case['budget_complete'] = [len(x) == n for x, n in zip(case['outputs'], data['output_tokens'])]
                case['cost_key'] = asdict(key(len(inputs)))
                del session, completed, rows, caches
                gc.collect()
                save()
    except Exception as exc:  # noqa: BLE001 -- preserve failures rather than retrying to select a good run
        report['failures'].append({'type': type(exc).__name__, 'detail': str(exc)})
    report.update(source_unchanged=sources() == report['source_sha256'], swap_after_bytes=psutil.swap_memory().used,
                  metal_peak_bytes=mx.get_peak_memory(), available_after_bytes=psutil.virtual_memory().available)
    cases = report['cases']
    report['coverage'] = {'completed_controls': sum('budget_complete' in c for c in cases),
                          'native_matched_requests': sum(sum(c.get('matches_native', [])) for c in cases),
                          'observed_output_tokens': sum(len(x) for c in cases for x in c['outputs'])}
    report['accepted'] = (not report['failures'] and report['source_unchanged'] and report['swap_after_bytes'] == swap
                          and len(cases) == 10 and all(all(c.get('matches_native', [False])) and all(c.get('budget_complete', [False]))
                          and len(c['next_state']) == len(inputs) and all(all(v[k] for k in ('argmax_equal', 'logits_allclose_1e5', 'offsets_equal'))
                          for s in c['next_state'] for v in (s['target'], s['draft'])) for c in cases))
    save()
    print(json.dumps({'accepted': report['accepted'], 'coverage': report['coverage'], 'failures': report['failures']}))
    if not report['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
