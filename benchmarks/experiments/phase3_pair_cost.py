"""Bounded distinct pretrained-pair development/held-out probe; no downloads.

Without --execute this validates the private workload and prints the frozen scope.
Development and held-out are separate invocations. All outputs are private raw
artifacts until independently redacted. A failed strict gate never deletes data.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

ACTIVE_REPORT = None
MANIFEST = Path('benchmarks/experiments/phase3_pair_workload_manifest.json')
TARGET = Path('results/phase3-models/smollm2-360m-instruct-6bit')
DRAFT = Path('results/phase3-models/smollm2-135m-instruct-8bit')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = [Path(__file__), MANIFEST, Path('uv.lock'), Path('pyproject.toml'),
             *[Path('src/deepinfer_mac') / p for p in ('varlen_verify.py', 'adapters.py', 'models.py',
                                                     'hybrid_speculative.py', 'ds/joint_budget.py', 'ds/mtp_varlen.py')]]
    return {p.resolve().relative_to(Path.cwd().resolve()).as_posix(): sha(p) for p in sorted(paths)}


def workload(path):
    manifest, data = json.loads(MANIFEST.read_text()), json.loads(path.read_text())
    if sha(path) != manifest['workload_sha256']:
        raise ValueError('Private workload differs from the predeclared manifest')
    if (data['depth'], data['batch_size'], data['output_tokens'], data['context_bucket']) != (3, 2, 32, 512):
        raise ValueError('This bounded protocol requires depth 3, batch 2, output 32, context bucket 512')
    ids = [x['id'] for split in ('development', 'heldout') for x in data[split]]
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError('Two disjoint development and two held-out tasks required')
    return manifest, data


def validate_calibration(frozen, source, identities, calibration_id):
    """Fail closed before device loading; never turn a wrong curve into fallback."""
    import math

    expected_execution = {"draft": "serial", "recovery": "exact_replay", "profiling": False}
    if (frozen.get('stage') != 'development' or frozen.get('accepted') is not True
            or frozen.get('calibration_frozen') is not True
            or frozen.get('source_sha256') != source
            or frozen.get('model_content_sha256') != identities
            or frozen.get('calibration_identity') != calibration_id
            or frozen.get('execution_identity') != expected_execution):
        raise ValueError('Calibration must match frozen development source/model/workload and unprofiled execution identity')
    expected = [{"model_identity": identities[0], "draft_identity": identities[1],
                 "quantization": "target6bit-g64/draft8bit-g64", "batch_size": batch,
                 "context_bucket": 512, "execution_strategy": "bucket",
                 "calibration_identity": calibration_id} for batch in (1, 2)]
    curves = frozen.get('curves', [])
    if len(curves) != len(expected) or sorted(
            json.dumps(c.get('key'), sort_keys=True) for c in curves) != sorted(
            json.dumps(k, sort_keys=True) for k in expected):
        raise ValueError('Calibration requires exactly all expected CostKeys, without duplicates or foreign strategies')
    for curve in curves:
        batch = curve['key']['batch_size']
        points = curve.get('points', [])
        if (not all(isinstance(p, list) and len(p) == 2 for p in points)
                or [p[0] for p in points] != list(range(batch, batch * 4 + 1))
                or any(type(p[1]) not in (int, float) or not math.isfinite(p[1]) or p[1] <= 0 for p in points)):
            raise ValueError('Calibration requires positive finite costs for every legal budget')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('development', 'heldout'), default='development')
    parser.add_argument('--workload', type=Path, default=Path('artifacts/phase3-pair-cost/workloads.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--profile-stages', action='store_true')
    args = parser.parse_args()
    if args.profile_stages and args.stage != "development":
        raise ValueError("Profiling is development-only; held-out workload stays frozen")
    manifest, data = workload(args.workload)
    if not args.execute:
        print(json.dumps({'stage': args.stage, 'manifest': manifest,
                          'gpu_execution': False, 'requires_root_gpu_baton': True}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.stage == 'heldout' and args.calibration is None:
        raise ValueError('Held-out requires a frozen development artifact')
    # Import device libraries only behind explicit execution.
    import mlx.core as mx
    import psutil
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    from deepinfer_mac.adapters import ModelAdapter, inspect_snapshot
    from deepinfer_mac.ds.joint_budget import (
        CostKey,
        EmpiricalConfidence,
        JointBudget,
        RequestKey,
        allocate_cpu,
    )
    from deepinfer_mac.models import verify_pair
    from deepinfer_mac.varlen_verify import JointSpeculativeSession, VerifyRow, execution_identity

    def read_swap_bytes():
        return psutil.swap_memory().used

    audits = [inspect_snapshot(p) for p in (TARGET, DRAFT)]
    pair = verify_pair(TARGET, DRAFT)
    if not all(pair['explicit_thinking_template_tokens_equal'].values()):
        raise ValueError('Chat template mismatch')
    if any(a['config']['model_type'] != 'llama' for a in audits) or sum(a['weight_bytes'] for a in audits) > 450_000_000:
        raise ValueError('Only the audited bounded dense pair is permitted')
    identities = [a['content_sha256'] for a in audits]
    if identities != ['40e44f0e35876a3e846c856f830ecca4208db73319256ff4f493434d2723950d',
                      'a4f7f71502edccf61c70717a7edb5368266a104f0c45cc1c0ed72cfb7ea27981']:
        raise ValueError('Exact audited distinct pretrained pair required')
    adapter = ModelAdapter({})
    # Same 85% load / 65% future-KV safety thresholds as production; reserve
    # additionally covers all rows and three copies for exact replay snapshots.
    kv_reserve = 3 * data['batch_size'] * sum(adapter.memory_estimate(a['config'], 1024) for a in audits)
    required = sum(a['weight_bytes'] for a in audits) + 2 * 1024**3 + kv_reserve
    available = psutil.virtual_memory().available
    if required > available * .85 or kv_reserve > available * .65:
        raise MemoryError(f'Conservative pair and snapshot admission requires {required} bytes')
    source = sources()
    calibration_id = hashlib.sha256(json.dumps([identities, manifest['workload_sha256']], sort_keys=True).encode()).hexdigest()
    frozen = None
    if args.stage == 'heldout':
        frozen = json.loads(args.calibration.read_text())
        validate_calibration(frozen, source, identities, calibration_id)
    report = {'schema_version': 1, 'stage': args.stage, 'source_sha256': source,
              'execution_identity': {'draft': 'serial', 'recovery': 'exact_replay', 'profiling': args.profile_stages},
              'model_content_sha256': identities, 'calibration_identity': calibration_id,
              'calibration_artifact_sha256': sha(args.calibration) if frozen else None,
              'workload_manifest': manifest, 'process_count': 1, 'versions': {p: version(p) for p in ('mlx', 'mlx-lm')},
              'platform': platform.mac_ver()[0], 'available_before_bytes': available,
              'required_admission_bytes': required, 'swap_before_bytes': read_swap_bytes(),
              'cases': [], 'curves': [], 'failures': [], 'accepted': False,
              'performance_claim': False, 'limits': [
                  'One process per stage; not three independent performance repetitions.',
                  'Fixed 32-token budget ignores EOS equally for all controls; not a chat-quality evaluation.',
                  'Cold refers to empty KV caches with resident warmed weights; model load is separate.',
                  'Development forces q: measured cycle includes draft/verify/recovery/catch-up but excludes normal JointBudget.choose and MLX rank allocation; adaptive held-out full_cycle_ms includes both.',
                  'Complete-cycle curve is a representative development estimate for one context bucket; actual shapes are retained.',
                  'Small held-out calibration coverage is diagnostic, not confidence-model acceptance.',
                  'Same draft depth and exact replay/catch-up are used across controls; padded states are discarded.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        global ACTIVE_REPORT
        ACTIVE_REPORT = (args.output, report)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    deadline = time.monotonic() + 300
    def guard():
        if time.monotonic() > deadline:
            raise TimeoutError('Five-minute stage bound reached; partial measurements retained')
        if read_swap_bytes() > report['swap_before_bytes']:
            raise MemoryError('Swap growth detected; stop without relaxing resource gate')
        if kv_reserve > psutil.virtual_memory().available * .65:
            raise MemoryError('Future snapshot/KV reserve no longer fits')
    start = time.perf_counter()
    target, tokenizer = load(TARGET)
    draft, _ = load(DRAFT)
    mx.eval(target.parameters(), draft.parameters())
    report['resident_pair_load_ms'] = (time.perf_counter() - start) * 1000
    prefill_calls = []
    def prefill(model, tokens):
        cache = make_prompt_cache(model)
        model(mx.array([tokens[:-1]], dtype=mx.uint32), cache=cache)
        mx.eval([c.state for c in cache])
        prefill_calls.append({"phase": "prefill", "model": "target" if model is target else "draft",
                              "batch": 1, "sequence_length": len(tokens) - 1, "tokens": len(tokens) - 1})
        return cache
    # Explicit separate warm-up, never excluded from an alleged cold-start claim.
    start = time.perf_counter()
    for model in (target, draft):
        cache = prefill(model, [1, 2, 3, 4])
        del cache
    report['pair_warmup_ms'] = (time.perf_counter() - start) * 1000
    report['pair_warmup_model_calls'] = list(prefill_calls)
    mx.reset_peak_memory()
    tokenized = []
    for task in data[args.stage]:
        # Preserve the question at the end while adding deterministic context.
        question = tokenizer.encode(task['prompt'])
        padding = tokenizer.encode(data['padding_text'] * 150)
        n = task['context_tokens']
        if len(question) >= n:
            raise ValueError('Question exceeds declared input size')
        tokens = padding[:n - len(question)] + question
        tokenized.append(tokens)
    report['input_token_counts'] = list(map(len, tokenized))
    def key(batch, strategy):
        return CostKey(identities[0], identities[1], 'target6bit-g64/draft8bit-g64', batch, 512, execution_identity(strategy, args.profile_stages), calibration_id)
    def setup(tokens, strategy, curve=False):
        budget = JointBudget(3, identities[0])
        if curve:
            for item in frozen['curves']:
                budget.set_curve(CostKey(**item['key']), item['points'])
        session = JointSpeculativeSession(target, draft, budget, key(len(tokens), strategy),
                                         draft_identity=identities[1], calibration_identity=calibration_id, allocator='mlx', profiling=args.profile_stages)
        rows = [VerifyRow(RequestKey(str(i), 0), prefill(target, t), t[-1], (), 32) for i, t in enumerate(tokens)]
        caches = [prefill(draft, t) for t in tokens]
        if frozen:
            for row in rows:
                predictor = EmpiricalConfidence(3, identities[0], calibration_id)
                predictor.counts = copy.deepcopy(frozen['frozen_confidence']['counts'])
                session.predictors[row.key] = predictor
        return session, rows, caches
    aggregate = EmpiricalConfidence(3, identities[0], calibration_id)
    try:
        if args.stage == 'development':
            # A short full-depth prelude observes only verified prefixes. It is
            # development-only; held-out tokens cannot change this prior/curve.
            prefill_mark = len(prefill_calls)
            session, rows, caches = setup(tokenized, 'bucket')
            report['development_prelude_prefill_calls'] = prefill_calls[prefill_mark:]
            prelude = []
            for _ in range(3):
                guard()
                tick = session.tick(rows, caches, forced_lengths=(4, 4))
                prelude.append(tick)
                for row, emitted, outcomes in zip(rows, tick['outputs'], tick['outcomes']):
                    row.tail = emitted[-1]
                    aggregate.observe(outcomes)
            report['development_prelude'] = prelude
            del session, rows, caches
            for batch in (1, 2):
                samples = []
                for total in range(batch, batch * 4 + 1):
                    lengths = allocate_cpu([aggregate.predict()] * batch, list(range(batch)), total, 3)
                    for repetition in range(data['development_repetitions']):
                        guard()
                        prefill_mark = len(prefill_calls)
                        start = time.perf_counter()
                        session, rows, caches = setup(tokenized[:batch], 'bucket')
                        prefill_ms = (time.perf_counter() - start) * 1000
                        tick = session.tick(rows, caches, forced_lengths=lengths)
                        samples.append({'total_q': total, 'lengths': lengths, 'repetition': repetition,
                                        'prefill_ms': prefill_ms, 'prefill_model_calls': prefill_calls[prefill_mark:], 'tick': tick,
                                        'complete_task_ms': (time.perf_counter() - start) * 1000})
                        del session, rows, caches
                        gc.collect()
                report['curves'].append({'key': asdict(key(batch, 'bucket')), 'samples': samples,
                                         'points': [[q, statistics.median(s['tick']['full_cycle_ms'] for s in samples if s['total_q'] == q)]
                                                    for q in range(batch, batch * 4 + 1)]})
                save()
            report['frozen_confidence'] = {'counts': aggregate.counts, 'conditional': aggregate.predict().values,
                                           'diagnostics': aggregate.diagnostics()}
        else:
            native = []
            for tokens in tokenized:
                guard()
                start = time.perf_counter()
                emitted, distributions, arrivals = [], [], []
                for token, logprobs in generate_step(mx.array(tokens, dtype=mx.uint32), target, max_tokens=32):
                    emitted.append(int(token))
                    distributions.append(logprobs)
                    arrivals.append((time.perf_counter() - start) * 1000)
                native_ms = (time.perf_counter() - start) * 1000
                # Diagnostic sorting is outside the measured native interval.
                margins = []
                for probabilities in distributions:
                    top = mx.sort(probabilities.astype(mx.float32))[-2:]
                    mx.eval(top)
                    margins.append((top[-1] - top[-2]).item())
                native.append({'tokens': emitted, 'top_two_logprob_margins': margins,
                               'native_api': 'mlx_lm.generate.generate_step', 'first_token_ms': arrivals[0],
                               'arrival_ms': arrivals, 'complete_task_ms': native_ms})
                del distributions
            report['native_serial'] = native
            policies = [('cost_bucket', 'bucket', None), ('fixed_full_bucket', 'bucket', 'full'),
                        ('heterogeneous_bucket', 'bucket', 'hetero'), ('heterogeneous_padded', 'padded_control', 'hetero'),
                        ('fixed_shared_cap', 'shared_cap', 'full')]
            for name, strategy, allocation in policies:
                guard()
                prefill_mark = len(prefill_calls)
                start = time.perf_counter()
                session, rows, caches = setup(tokenized, strategy, curve=name == 'cost_bucket')
                outputs = [[], []]
                case = {'policy': name, 'strategy': strategy, 'prefill_ms': (time.perf_counter() - start) * 1000,
                        'prefill_model_calls': prefill_calls[prefill_mark:],
                        'cycles': [], 'outputs': outputs, 'frozen_confidence_observations': []}
                report['cases'].append(case)
                while rows:
                    guard()
                    forced = None if allocation is None else tuple(4 if allocation == 'full' else (1, 4)[int(row.key.uid)] for row in rows)
                    tick = session.tick(rows, caches, strategy=strategy, forced_lengths=forced, cost_key=key(len(rows), strategy))
                    tick['request_ids'] = [row.key.uid for row in rows]
                    case['cycles'].append(tick)
                    retained_rows, retained_caches = [], []
                    for row, cache, emitted, observed in zip(rows, caches, tick['outputs'], tick['outcomes']):
                        index = int(row.key.uid)
                        outputs[index].extend(emitted)
                        case['frozen_confidence_observations'].append({'request_id': row.key.uid, 'outcomes': observed})
                        row.remaining -= len(emitted)
                        row.tail = emitted[-1]
                        if row.remaining:
                            retained_rows.append(row)
                            retained_caches.append(cache)
                    rows, caches = retained_rows, retained_caches
                case['complete_task_ms'] = (time.perf_counter() - start) * 1000
                case['all_output_budgets_complete'] = all(len(out) == 32 for out in outputs)
                case['matches_native'] = [out == ref['tokens'] for out, ref in zip(outputs, native)]
                case['first_native_divergences'] = []
                for out, ref in zip(outputs, native):
                    mismatch = next((i for i, (a, b) in enumerate(zip(out, ref['tokens'])) if a != b), None)
                    case['first_native_divergences'].append(None if mismatch is None else {
                        'index': mismatch, 'actual': out[mismatch], 'native': ref['tokens'][mismatch],
                        'native_top_two_logprob_margin': ref['top_two_logprob_margins'][mismatch]})
                calibration = []
                for position, prediction in enumerate(frozen['frozen_confidence']['conditional']):
                    values = [o['outcomes'][position] for o in case['frozen_confidence_observations'] if o['outcomes'][position] is not None]
                    calibration.append({'position': position, 'prediction': prediction, 'coverage': len(values),
                                        'observed_acceptance': sum(values) / len(values) if values else None,
                                        'brier': statistics.mean((prediction - int(v))**2 for v in values) if values else None})
                case['heldout_frozen_calibration'] = calibration
                case['decode_model_input_positions'] = sum(t['executed_total_q'] for t in case['cycles'])
                case['total_model_input_positions'] = case['decode_model_input_positions'] + sum(c['tokens'] for c in case['prefill_model_calls'])
                case['model_submissions'] = sum(t['model_submissions'] for t in case['cycles']) + len(case['prefill_model_calls'])
                del session, rows, caches
                gc.collect()
                save()
            report['bucket_padded_outputs_equal'] = report['cases'][2]['outputs'] == report['cases'][3]['outputs']
    except Exception as exc:  # noqa: BLE001 -- keep every bounded experiment failure
        report['failures'].append({'error_type': type(exc).__name__, 'message': str(exc)})
    report.update({'source_unchanged': sources() == source, 'swap_after_bytes': read_swap_bytes(),
                   'metal_peak_bytes': mx.get_peak_memory(), 'rss_snapshot_bytes': psutil.Process().memory_info().rss,
                   'available_after_bytes': psutil.virtual_memory().available})
    resource_ok = report['swap_after_bytes'] == report['swap_before_bytes']
    base_ok = not report['failures'] and report['source_unchanged'] and resource_ok
    if args.stage == 'development':
        report['calibration_frozen'] = base_ok and len(report['curves']) == 2 and all(len(x['samples']) == 2 * (x['key']['batch_size'] * 3 + 1) for x in report['curves'])
        report['accepted'] = report['calibration_frozen']
    else:
        report['mechanism_complete'] = base_ok and len(report['cases']) == 5 and all(c.get('all_output_budgets_complete') for c in report['cases'])
        report['accepted'] = report['mechanism_complete'] and report['bucket_padded_outputs_equal'] and all(all(c['matches_native']) for c in report['cases'])
    save()
    print(json.dumps({'stage': args.stage, 'accepted': report['accepted'], 'failures': report['failures']}))
    if not report['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ACTIVE_REPORT is not None:
            path, partial = ACTIVE_REPORT
            partial['accepted'] = False
            partial['calibration_frozen'] = False
            partial['failures'].append({'error_type': type(exc).__name__, 'message': str(exc), 'phase': 'outer_load_or_setup'})
            path.write_text(json.dumps(partial, indent=2) + '\n')
        raise
