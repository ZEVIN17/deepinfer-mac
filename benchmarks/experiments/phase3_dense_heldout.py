"""Frozen three-process heldout diagnostic; no tuning or performance promotion."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

from phase3_dense_fastpath_v2 import IDENTITIES, MANIFEST, MODEL_PATHS, ROOT, load_workload, sha

PROTOCOL = ROOT / 'benchmarks/experiments/phase3_dense_heldout_protocol.json'


def sources():
    paths = [Path(__file__), ROOT / 'benchmarks/experiments/phase3_dense_fastpath_v2.py',
             PROTOCOL, MANIFEST, ROOT / 'uv.lock', ROOT / 'pyproject.toml',
             *sorted((ROOT / 'src/deepinfer_mac').rglob('*.py'))]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--round', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--workload', type=Path, default=ROOT / 'artifacts/phase3-dense-fastpath/workloads-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    manifest, workload = load_workload(args.workload)
    if sha(Path(__file__)) != protocol['harness_sha256']:
        raise ValueError('Harness differs from frozen protocol')
    current_sources = sources()
    if any(current_sources.get(path) != expected for path, expected in protocol['runtime_source_sha256'].items()):
        raise ValueError('Runtime differs from frozen development selection')
    if not args.execute:
        print(json.dumps({'execute': False, 'protocol': protocol, 'round': args.round}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    import mlx.core as mx
    import psutil
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator, generate_step
    from mlx_lm.models.cache import make_prompt_cache

    from deepinfer_mac.adapters import ModelAdapter, inspect_snapshot
    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget, RequestKey
    from deepinfer_mac.models import verify_pair
    from deepinfer_mac.varlen_verify import JointSpeculativeSession, VerifyRow, execution_identity

    audits = [inspect_snapshot(path) for path in MODEL_PATHS]
    if [a['content_sha256'] for a in audits] != IDENTITIES:
        raise ValueError('Audited model identity mismatch')
    if not all(verify_pair(*MODEL_PATHS)['explicit_thinking_template_tokens_equal'].values()):
        raise ValueError('Pair templates differ')
    reserve = 6 * sum(ModelAdapter({}).memory_estimate(a['config'], 1024) for a in audits)
    required = sum(a['weight_bytes'] for a in audits) + 2 * 1024**3 + reserve
    available, swap = psutil.virtual_memory().available, psutil.swap_memory().used
    if required > .85 * available or reserve > .65 * available:
        raise MemoryError('Conservative admission failed')
    report = {'schema_version': 1, 'stage': 'heldout_diagnostic', 'round': args.round,
              'source_sha256': sources(), 'protocol': protocol, 'workload_manifest': manifest,
              'models': [{'content_sha256': a['content_sha256'], 'weight_bytes': a['weight_bytes'],
                          'quantization': a['config'].get('quantization')} for a in audits],
              'versions': {name: version(name) for name in ('mlx', 'mlx-lm')},
              'platform': platform.mac_ver()[0], 'available_before_bytes': available,
              'required_admission_bytes': required, 'swap_before_bytes': swap,
              'cases': [], 'failures': [], 'accepted': False, 'performance_claim': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    save()
    deadline = time.monotonic() + 300

    def guard():
        if time.monotonic() > deadline:
            raise TimeoutError('Five-minute process bound')
        if psutil.swap_memory().used > swap or reserve > .65 * psutil.virtual_memory().available:
            raise MemoryError('Swap growth or future reserve guard')

    try:
        start = time.perf_counter()
        target, tokenizer = load(MODEL_PATHS[0])
        draft, _ = load(MODEL_PATHS[1])
        mx.eval(target.parameters(), draft.parameters())
        report['resident_pair_load_ms'] = (time.perf_counter() - start) * 1000

        def prefill(model, tokens):
            cache = make_prompt_cache(model)
            model(mx.array([tokens[:-1]], dtype=mx.uint32), cache=cache)
            mx.eval([c.state for c in cache])
            return cache

        start = time.perf_counter()
        for model in (target, draft):
            cache = prefill(model, [1, 2, 3, 4])
            del cache
        report['warmup_ms'] = (time.perf_counter() - start) * 1000
        inputs = []
        for task in workload['heldout']:
            question = tokenizer.encode(task['prompt'])
            padding = tokenizer.encode(workload['padding_text'] * 150)
            if len(question) >= task['context_tokens']:
                raise ValueError('Question exceeds fixed context')
            inputs.append(padding[:task['context_tokens'] - len(question)] + question)
        budgets = workload['output_tokens']
        report['input_token_counts'] = list(map(len, inputs))
        # Untimed serial target quality oracle is fixed before all measured controls.
        native = []
        for tokens, count in zip(inputs, budgets):
            guard()
            native.append([int(token) for token, _ in generate_step(mx.array(tokens, dtype=mx.uint32), target, max_tokens=count)])
        report['serial_quality_oracle'] = native
        mx.reset_peak_memory()
        for control in protocol['control_orders'][args.round]:
            guard()
            case = {'control': control, 'outputs': [[], []], 'arrival_ms': [[], []],
                    'request_completion_ms': [None, None], 'cycles': [], 'next_state': []}
            report['cases'].append(case)
            start = time.perf_counter()
            if control == 'native_batch':
                generator = BatchGenerator(target, completion_batch_size=2, prefill_batch_size=2,
                                           prefill_step_size=256, stop_tokens=[])
                try:
                    uids = generator.insert(inputs, max_tokens=budgets)
                    indices = {uid: i for i, uid in enumerate(uids)}
                    done = set()
                    while len(done) < len(inputs):
                        guard()
                        _, responses = generator.next()
                        elapsed = (time.perf_counter() - start) * 1000
                        for response in responses:
                            i = indices[response.uid]
                            case['outputs'][i].append(int(response.token))
                            case['arrival_ms'][i].append(elapsed)
                            if response.finish_reason is not None:
                                case['request_completion_ms'][i] = elapsed
                                done.add(i)
                    case['complete_task_ms'] = (time.perf_counter() - start) * 1000
                finally:
                    generator.close()
                del generator
            else:
                draft_mode, recovery = ('serial', 'exact_replay') if control == 'old_replay' else ('batched', 'dense_trim')

                def key(batch, draft_mode=draft_mode, recovery=recovery):
                    return CostKey(IDENTITIES[0], IDENTITIES[1], 'target6bit-g64/draft8bit-g64', batch, 512,
                                   execution_identity('bucket', draft_execution=draft_mode, recovery=recovery), manifest['workload_sha256'])

                session = JointSpeculativeSession(target, draft, JointBudget(3, IDENTITIES[0]), key(2),
                                                 draft_identity=IDENTITIES[1], calibration_identity=manifest['workload_sha256'],
                                                 allocator='mlx', draft_execution=draft_mode, recovery=recovery)
                rows = [VerifyRow(RequestKey(str(i), 0), prefill(target, tokens), tokens[-1], (), count)
                        for i, (tokens, count) in enumerate(zip(inputs, budgets))]
                caches = [prefill(draft, tokens) for tokens in inputs]
                case['prefill_ms'] = (time.perf_counter() - start) * 1000
                completed, transcripts = [], [[], []]
                while rows:
                    guard()
                    tick = session.tick(rows, caches, forced_lengths=tuple(4 for _ in rows), cost_key=key(len(rows)))
                    tick['request_ids'] = [row.key.uid for row in rows]
                    case['cycles'].append(tick)
                    elapsed = (time.perf_counter() - start) * 1000
                    retained_rows, retained_caches = [], []
                    for row, cache, emitted in zip(rows, caches, tick['outputs']):
                        i = int(row.key.uid)
                        if not emitted or len(emitted) > row.remaining:
                            raise ValueError('Empty emission or output budget overshoot')
                        transcripts[i].append([row.tail, *emitted[:-1]])
                        case['outputs'][i].extend(emitted)
                        case['arrival_ms'][i].extend([elapsed] * len(emitted))
                        row.remaining -= len(emitted)
                        row.tail = emitted[-1]
                        if row.remaining:
                            retained_rows.append(row)
                            retained_caches.append(cache)
                        else:
                            case['request_completion_ms'][i] = elapsed
                            completed.append((row, cache))
                    rows, caches = retained_rows, retained_caches
                case['complete_task_ms'] = (time.perf_counter() - start) * 1000
                # Untimed next-state diagnostics: full-prefix and cycle-partition replay.
                for row, cache in completed:
                    i = int(row.key.uid)
                    checks = {'request_index': i}
                    for label, model, actual_cache in (('target', target, row.cache), ('draft', draft, cache)):
                        actual = model(mx.array([[row.tail]], dtype=mx.uint32), cache=copy.deepcopy(actual_cache))
                        mx.eval(actual)
                        checks[label] = {}
                        for oracle in ('whole_prefix', 'cycle_partition'):
                            guard()
                            if oracle == 'whole_prefix':
                                oracle_cache = prefill(model, inputs[i] + case['outputs'][i])
                            else:
                                oracle_cache = prefill(model, inputs[i])
                                for consumed in transcripts[i]:
                                    model(mx.array([consumed], dtype=mx.uint32), cache=oracle_cache)
                                    mx.eval([c.state for c in oracle_cache])
                            offset_equal = [c.offset for c in actual_cache] == [c.offset for c in oracle_cache]
                            expected = model(mx.array([[row.tail]], dtype=mx.uint32), cache=oracle_cache)
                            mx.eval(expected)
                            checks[label][oracle] = {
                                'offsets_equal': offset_equal,
                                'argmax_equal': mx.argmax(actual, -1).tolist() == mx.argmax(expected, -1).tolist(),
                                'allclose_1e5': bool(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()),
                                'max_abs_error': float(mx.max(mx.abs(actual - expected)).item())}
                    case['next_state'].append(checks)
                del completed, session, rows, caches
            case['matches_native'] = [out == ref for out, ref in zip(case['outputs'], native)]
            case['budget_complete'] = [len(out) == count for out, count in zip(case['outputs'], budgets)]
            case['first_divergence'] = [next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)
                                        for out, ref in zip(case['outputs'], native)]
            save()
    except Exception as exc:  # noqa: BLE001 -- preserve every failed diagnostic
        report['failures'].append({'type': type(exc).__name__, 'detail': str(exc)})
    report.update(source_unchanged=sources() == report['source_sha256'], swap_after_bytes=psutil.swap_memory().used,
                  metal_peak_bytes=mx.get_peak_memory(), available_after_bytes=psutil.virtual_memory().available)
    cases = report['cases']
    report['gate_counts'] = {'submitted_requests': sum(len(c['outputs']) for c in cases),
                            'observed_output_requests': sum(bool(out) for c in cases for out in c['outputs']),
                            'native_matched_requests': sum(sum(c.get('matches_native', [])) for c in cases),
                            'budget_complete_requests': sum(sum(c.get('budget_complete', [])) for c in cases)}
    for oracle in ('whole_prefix', 'cycle_partition'):
        checks = [s[label][oracle] for c in cases for s in c['next_state'] for label in ('target', 'draft')]
        report['gate_counts'][oracle] = {'checked': len(checks), **{
            key: sum(c[key] for c in checks) for key in ('allclose_1e5', 'argmax_equal', 'offsets_equal')}}
    base = (not report['failures'] and report['source_unchanged'] and report['swap_after_bytes'] == swap
            and len(cases) == 3 and all(all(c.get('matches_native', [False])) and all(c.get('budget_complete', [False])) for c in cases))
    report['functional_output_pass'] = base
    report['accepted'] = base and all(report['gate_counts'][name]['checked'] == 8 and
        all(report['gate_counts'][name][key] == 8 for key in ('allclose_1e5', 'argmax_equal', 'offsets_equal'))
        for name in ('whole_prefix', 'cycle_partition'))
    save()
    print(json.dumps({'accepted': report['accepted'], 'counts': report['gate_counts'], 'failures': report['failures']}))
    if not report['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
