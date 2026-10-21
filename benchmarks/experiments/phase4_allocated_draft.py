"""Three-process development ablation; public fresh prompts, natural EOS."""
from __future__ import annotations

import argparse
import copy
import json
import time
from importlib.metadata import version
from pathlib import Path

from phase3_dense_fastpath import IDENTITIES, MODEL_PATHS, ROOT, sha

TASKS = [
    'Write a Python function that returns the first duplicate in a list, or None if all values are unique.',
    'Explain to a new developer why a cancelled HTTP request must release its database connection. Use three sentences.',
]
ORDERS = [('native', 'full', 'allocated'), ('full', 'allocated', 'native'), ('allocated', 'native', 'full')]


def sources():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in
            [Path(__file__), *sorted((ROOT / 'src/deepinfer_mac').rglob('*.py'))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--round', type=int, choices=range(3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'tasks': TASKS, 'orders': ORDERS, 'depth': 3, 'limits': [32, 32],
                          'lengths': [1, 4], 'scope': 'development; not heldout or competitive acceptance'}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    import mlx.core as mx
    import psutil
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import make_prompt_cache

    from deepinfer_mac.adapters import ModelAdapter, inspect_snapshot
    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget, RequestKey
    from deepinfer_mac.varlen_verify import JointSpeculativeSession, VerifyRow, execution_identity

    audits = [inspect_snapshot(p) for p in MODEL_PATHS]
    if [a['content_sha256'] for a in audits] != IDENTITIES:
        raise ValueError('Model identity mismatch')
    reserve = 8 * sum(ModelAdapter({}).memory_estimate(a['config'], 1024) for a in audits)
    swap = psutil.swap_memory().used
    required = sum(a['weight_bytes'] for a in audits) + 2 * 1024**3 + reserve
    if required > .85 * psutil.virtual_memory().available:
        raise MemoryError('Admission guard')
    report = {'scope': 'development_allocated_draft_ablation', 'round': args.round,
              'tasks': TASKS, 'output_limits': [32, 32], 'natural_eos': True,
              'models': IDENTITIES, 'versions': {p: version(p) for p in ('mlx', 'mlx-lm')},
              'source_sha256': sources(), 'controls': [], 'failures': [],
              'swap_before_bytes': swap, 'performance_acceptance': False,
              'timing_scope': 'resident pair; includes prompt prefill and decode, excludes load and shared warmup'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 180

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def guard():
        if time.monotonic() > deadline:
            raise TimeoutError('Three minute process limit')
        if psutil.swap_memory().used > swap or reserve > .65 * psutil.virtual_memory().available:
            raise MemoryError('Swap growth or future reserve guard')

    def prefill(model, tokens):
        cache = make_prompt_cache(model)
        model(mx.array([tokens[:-1]], dtype=mx.uint32), cache=cache)
        mx.eval([c.state for c in cache])
        return cache

    save()
    try:
        target, tokenizer = load(MODEL_PATHS[0])
        draft, _ = load(MODEL_PATHS[1])
        mx.eval(target.parameters(), draft.parameters())
        inputs = [tokenizer.encode(tokenizer.apply_chat_template(
            [{'role': 'user', 'content': task}], tokenize=False, add_generation_prompt=True)) for task in TASKS]
        eos = frozenset(tokenizer.eos_token_ids)
        report['input_tokens'] = inputs
        # Equal explicit tiny warmup before rotated controls; not tuned per arm.
        for model in (target, draft):
            prefill(model, [1, 2, 3, 4])
        mx.reset_peak_memory()
        for control in ORDERS[args.round]:
            guard()
            outputs, cycles, completed = [[], []], [], [None, None]
            begin = time.perf_counter()
            if control == 'native':
                generator = BatchGenerator(target, stop_tokens=[[t] for t in eos],
                                           completion_batch_size=2, prefill_batch_size=2, prefill_step_size=256)
                try:
                    uids = generator.insert(inputs, max_tokens=[32, 32])
                    mapping = {uid: i for i, uid in enumerate(uids)}
                    while any(x is None for x in completed):
                        guard()
                        _, generated = generator.next()
                        for r in generated:
                            i = mapping[r.uid]
                            outputs[i].append(r.token)
                            if r.finish_reason:
                                completed[i] = (time.perf_counter() - begin) * 1000
                finally:
                    mx.synchronize(generator.stream)
                    generator.close()
            else:
                target_caches = [prefill(target, tokens) for tokens in inputs]
                drafts = [prefill(draft, tokens) for tokens in inputs]
                rows = [VerifyRow(RequestKey(str(i), 1), cache, tokens[-1], (), 32, eos)
                        for i, (cache, tokens) in enumerate(zip(target_caches, inputs))]
                key = CostKey(IDENTITIES[0], IDENTITIES[1], json.dumps([a['config'].get('quantization') for a in audits]),
                              2, max(len(t) for t in inputs), execution_identity('bucket', draft_budget=control), 'development')
                session = JointSpeculativeSession(target, draft, JointBudget(3, IDENTITIES[0]), key,
                                                 draft_identity=IDENTITIES[1], calibration_identity='development',
                                                 draft_budget=control)
                active = [0, 1]
                while active:
                    guard()
                    from dataclasses import replace
                    result = session.tick([rows[i] for i in active], [drafts[i] for i in active],
                                          forced_lengths=tuple((1, 4)[i] for i in active),
                                          cost_key=replace(key, batch_size=len(active)))
                    cycles.append({k: copy.deepcopy(result[k]) for k in
                                   ('full_cycle_ms', 'executed_total_q', 'executed_draft_q', 'seqlens_q', 'model_calls')})
                    for i, emitted in zip(active, result['outputs']):
                        outputs[i].extend(emitted)
                        rows[i].remaining -= len(emitted)
                        rows[i].tail = emitted[-1]
                        if emitted[-1] in eos or rows[i].remaining == 0:
                            completed[i] = (time.perf_counter() - begin) * 1000
                    active = [i for i in active if completed[i] is None]
            report['controls'].append({'control': control, 'task_ms': (time.perf_counter() - begin) * 1000,
                                       'completed_ms': completed, 'outputs': outputs, 'cycles': cycles})
            save()
        native = next(c['outputs'] for c in report['controls'] if c['control'] == 'native')
        report['outputs_match_native'] = all(c['outputs'] == native for c in report['controls'])
        report['source_unchanged'] = sources() == report['source_sha256']
        report['swap_after_bytes'] = psutil.swap_memory().used
        report['peak_metal_bytes'] = mx.get_peak_memory()
    except BaseException as exc:
        report['failures'].append({'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        save()
    print(json.dumps({'round': args.round, 'outputs_match_native': report['outputs_match_native'],
                      'times_ms': {c['control']: c['task_ms'] for c in report['controls']}}))


if __name__ == '__main__':
    main()
