"""Native MLX execution with fixed versus adaptive prefill; development trace."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

from phase3_dense_fastpath import IDENTITIES, MODEL_PATHS, ROOT, sha

# All arms use the same native generator, model, queue, arrival trace and EOS.
CONTROLS = [('fixed64', 'upstream', 64), ('fixed256', 'upstream', 256),
            ('fixed1024', 'upstream', 1024), ('adaptive1024', 'interactive', 1024)]
TRACE = [(0, 'chat', 0), (10, 'document', 250), (20, 'chat', 0),
         (30, 'document', 150), (40, 'chat', 0), (50, 'chat', 0)]


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    p = (len(values) - 1) * fraction
    lo = int(p)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (p - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--round', type=int, choices=range(3), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-prefill-chunk', type=int, default=32)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'controls': CONTROLS, 'trace': TRACE, 'max_active': 2,
                          'target_tick_ms': 10, 'output_limit': 64, 'scope': 'development'}))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    import mlx.core as mx
    import psutil

    from deepinfer_mac.adapters import ModelRegistry, ModelSpec, inspect_snapshot
    from deepinfer_mac.continuous import ContinuousConfig, MLXBatchSession
    from deepinfer_mac.engine import InferenceEngine
    from deepinfer_mac.state import GenerationRequest
    audit = inspect_snapshot(MODEL_PATHS[0])
    if audit['content_sha256'] != IDENTITIES[0]:
        raise ValueError('Model identity mismatch')
    swap = psutil.swap_memory().used
    if audit['weight_bytes'] + 2 * 1024**3 > .85 * psutil.virtual_memory().available:
        raise MemoryError('Native admission guard')
    source_paths = [Path(__file__), *sorted((ROOT / 'src/deepinfer_mac').rglob('*.py'))]
    sources = lambda: {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    report = {'scope': 'development_native_arrivals', 'round': args.round, 'trace': TRACE,
              'model_identity': IDENTITIES[0], 'controls': [], 'failures': [],
              'source_sha256': sources(), 'swap_before_bytes': swap, 'performance_acceptance': False,
              'timing_scope': 'resident warmed target; submit-to-emit includes tokenization/admission/ticks',
              'min_prefill_chunk': args.min_prefill_chunk, 'target_tick_ms': 10, 'max_active': 2, 'output_limit': 64, 'natural_eos': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: args.output.write_text(json.dumps(report, indent=2) + '\n')
    spec = ModelSpec('native-development', str(MODEL_PATHS[0]), context_limit=4096)
    engine = InferenceEngine(model_registry=ModelRegistry([spec]), cache_budget_bytes=0)
    deadline = time.monotonic() + 180
    session = None

    def guard():
        if time.monotonic() > deadline:
            raise TimeoutError('Three minute trace limit')
        if psutil.swap_memory().used > swap:
            raise MemoryError('Swap growth')

    save()
    try:
        # Explicit shared warmup. Each arm still starts with cold prompt caches.
        engine.generate(GenerationRequest(profile=spec.id, prompt='Explain a queue.', max_tokens=4, use_cache=False))
        order = CONTROLS[args.round:] + CONTROLS[:args.round]
        for name, policy, chunk in order:
            guard()
            session = MLXBatchSession(engine, spec.id, ContinuousConfig(
                max_active=2, tick_token_budget=1026, prefill_chunk=chunk,
                max_context_tokens=8192, target_tick_ms=10, policy=policy, min_prefill_chunk=args.min_prefill_chunk))
            requests = [GenerationRequest(profile=spec.id, prompt=(
                'The project stores each change in a queue. ' * repetitions +
                'Explain how a queue helps a busy team finish work fairly.'), max_tokens=64, use_cache=False)
                for _, _, repetitions in TRACE]
            pending, active, prepared = list(range(len(TRACE))), {}, {}
            emitted, outputs, done, admissions, chunks = {}, {}, {}, {}, []
            begin = time.perf_counter()
            while len(done) < len(TRACE):
                guard()
                now_ms = (time.perf_counter() - begin) * 1000
                for i in list(pending):
                    if TRACE[i][0] > now_ms:
                        break
                    if len(active) >= 2:
                        break
                    if i not in prepared:
                        prepared[i] = session.prepare(requests[i])
                    reason = session.budget.check(str(i), prepared[i], 64, **session.resources())
                    if reason:
                        if not active:
                            raise MemoryError('Trace admission rejected: ' + reason)
                        break
                    session.budget.reserve(str(i), prepared[i], 64)
                    uid = session.insert(prepared[i], requests[i])
                    active[uid] = i
                    admissions[i] = (time.perf_counter() - begin) * 1000
                    emitted[i], outputs[i] = [], []
                    pending.remove(i)
                if not active:
                    time.sleep(.001)
                    continue
                _, generated = session.tick()
                chunks.append(session.chunk)
                timestamp = (time.perf_counter() - begin) * 1000
                for r in generated:
                    i = active[r.uid]
                    emitted[i].append(timestamp)
                    outputs[i].append(r.token)
                    if r.finish_reason:
                        done[i] = timestamp
                        session.budget.release(str(i))
                        del active[r.uid]
            elapsed = (time.perf_counter() - begin) * 1000
            gaps = [b - a for times in emitted.values() for a, b in itertools.pairwise(times)]
            ttfts = [emitted[i][0] - TRACE[i][0] for i in range(len(TRACE))]
            report['controls'].append({'control': name, 'task_ms': elapsed,
                'ttft_p50_ms': percentile(ttfts, .5), 'ttft_p95_ms': percentile(ttfts, .95),
                'itl_p95_ms': percentile(gaps, .95), 'max_itl_ms': max(gaps),
                'tokens_per_second': sum(map(len, outputs.values())) / (elapsed / 1000),
                'prefill_chunks': sorted(set(chunks)), 'admitted_ms': admissions,
                'prompt_tokens': [len(prepared[i].tokens) for i in range(len(TRACE))],
                'request_ttft_ms': ttfts, 'completed_ms': done,
                'output_sha256': [hashlib.sha256(json.dumps(outputs[i]).encode()).hexdigest() for i in range(len(TRACE))]})
            session.close()
            session = None
            save()
        report['outputs_equal'] = all(c['output_sha256'] == report['controls'][0]['output_sha256'] for c in report['controls'])
        report['source_unchanged'] = sources() == report['source_sha256']
    except BaseException as exc:
        report['failures'].append({'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        if session is not None:
            session.close()
        engine.close()
        report['swap_after_bytes'] = psutil.swap_memory().used
        report['metal_peak_bytes'] = mx.get_peak_memory()
        save()
    print(json.dumps({'round': args.round, 'outputs_equal': report['outputs_equal'], 'results': [
        {k: c[k] for k in ('control', 'task_ms', 'ttft_p95_ms', 'itl_p95_ms')} for c in report['controls']]}))


if __name__ == '__main__':
    main()
