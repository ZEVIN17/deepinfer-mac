"""Fresh development controls separating cache operations from forward shapes."""
from __future__ import annotations

import argparse
import copy
import json
import time
from importlib.metadata import version
from pathlib import Path

from phase3_dense_fastpath import IDENTITIES, MODEL_PATHS, ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print('Fresh deterministic token histories; no heldout tuning; use --execute.')
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    import mlx.core as mx
    import psutil
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from deepinfer_mac.adapters import ModelAdapter, inspect_snapshot
    from deepinfer_mac.varlen_verify import trim_cache

    audits = [inspect_snapshot(p) for p in MODEL_PATHS]
    if [a['content_sha256'] for a in audits] != IDENTITIES:
        raise ValueError('Model identity changed')
    reserve = 8 * sum(ModelAdapter({}).memory_estimate(a['config'], 1024) for a in audits)
    available, swap = psutil.virtual_memory().available, psutil.swap_memory().used
    required = sum(a['weight_bytes'] for a in audits) + 2 * 1024**3 + reserve
    if required > .85 * available or reserve > .65 * available:
        raise MemoryError('Resource admission failed')
    report = {'schema_version': 1, 'scope': 'development_cache_shape_diagnosis',
                  'models': IDENTITIES, 'versions': {n: version(n) for n in ('mlx', 'mlx-lm')},
                  'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                                 [Path(__file__), ROOT / 'src/deepinfer_mac/varlen_verify.py']},
                  'tolerance': {'atol': 2e-6, 'rtol': 1e-5}, 'cases': [], 'failures': [],
                  'performance_claim': False, 'swap_before_bytes': swap}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 180

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def guard():
        if time.monotonic() > deadline:
            raise TimeoutError('Three minute diagnostic limit')
        if psutil.swap_memory().used > swap or reserve > .65 * psutil.virtual_memory().available:
            raise MemoryError('Swap growth or insufficient future reserve')

    def forward(model, tokens, cache):
        guard()
        logits = model(mx.array(tokens, dtype=mx.uint32), cache=cache)
        mx.eval(logits, [c.state for c in cache])
        return logits

    def compare(model, left, right, tail):
        offsets = [c.offset for c in left] == [c.offset for c in right]
        errors = []
        for a, b in zip(left, right):
            errors.append(max(float(mx.max(mx.abs(x[..., :a.offset, :] - y[..., :b.offset, :])).item())
                              for x, y in ((a.keys, b.keys), (a.values, b.values))))
        a = forward(model, [[tail]], copy.deepcopy(left))
        b = forward(model, [[tail]], copy.deepcopy(right))
        return {'offset_equal': offsets, 'cache_max_abs': max(errors), 'layer_max_abs': errors,
                    'logits_max_abs': float(mx.max(mx.abs(a - b)).item()),
                    'logits_allclose': bool(mx.allclose(a, b, atol=2e-6, rtol=1e-5).item()),
                    'next_token_equal': mx.argmax(a, -1).tolist() == mx.argmax(b, -1).tolist()}

    save()
    try:
        for identity, path in zip(IDENTITIES, MODEL_PATHS):
            guard()
            model, _ = load(path)
            mx.eval(model.parameters())
            initial = []
            # Self-authored synthetic diagnostic histories, distinct from Phase 3 tasks.
            for length in (33, 65):
                cache = make_prompt_cache(model)
                forward(model, [[11 + (i * 7) % 211 for i in range(length)]], cache)
                initial.append(cache)
            inputs = [[13, 27, 41, 55], [17, 31, 45, 59]]
            merged_initial = [c.merge([row[k] for row in initial]) for k, c in enumerate(initial[0])]
            roundtrip = [[c.extract(i) for c in merged_initial] for i in range(2)]
            merged_full = copy.deepcopy(merged_initial)
            forward(model, inputs, merged_full)
            extracted_full = [[c.extract(i) for c in merged_full] for i in range(2)]
            for i in range(2):
                controls = {'merge_extract_no_forward': compare(model, initial[i], roundtrip[i], inputs[i][0])}
                serial_full = copy.deepcopy(initial[i])
                forward(model, [inputs[i]], serial_full)
                for consumed in (1, 2, 4):
                    replay = copy.deepcopy(initial[i])
                    forward(model, [inputs[i][:consumed]], replay)
                    trimmed = copy.deepcopy(serial_full)
                    trim_cache(trimmed, 4 - consumed)
                    batched_trim = copy.deepcopy(extracted_full[i])
                    trim_cache(batched_trim, 4 - consumed)
                    # Oracle retains the exact same forward shape and slices its
                    # logical state directly, independently of trim().
                    sliced = copy.deepcopy(serial_full)
                    for c in sliced:
                        end = initial[i][0].offset + consumed
                        c.state = (c.keys[..., :end, :], c.values[..., :end, :], end)
                    controls[f'q4_trim_vs_q{consumed}_replay'] = compare(model, trimmed, replay, 73)
                    controls[f'q4_trim_vs_same_shape_slice_k{consumed}'] = compare(model, trimmed, sliced, 73)
                    controls[f'batch2_vs_batch1_q4_k{consumed}'] = compare(model, batched_trim, trimmed, 73)
                report['cases'].append({'model_identity': identity, 'context': initial[i][0].offset, 'controls': controls})
                save()
            del model, initial, merged_initial, merged_full, extracted_full, roundtrip
            mx.clear_cache()
        report['swap_after_bytes'] = psutil.swap_memory().used
        report['metal_peak_bytes'] = mx.get_peak_memory()
    except BaseException as exc:
        report['failures'].append({'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        save()
    print(json.dumps({'cases': len(report['cases']), 'failures': report['failures'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
