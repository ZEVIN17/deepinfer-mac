"""Independent local <=200MB same-target draft mechanism control, never a speed claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', type=Path, default=Path('results/phase3-joint/probe.json'))
    a = p.parse_args()
    if not a.execute:
        print('Existing Smol135M only; same-target draft mechanism control; no speed claim.')
        return
    if a.output.exists():
        raise FileExistsError(a.output)
    import mlx.core as mx
    import psutil
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from deepinfer_mac.adapters import inspect_snapshot
    from deepinfer_mac.ds.joint_budget import CostKey, JointBudget, RequestKey
    from deepinfer_mac.engine import read_swap_bytes
    from deepinfer_mac.varlen_verify import JointSpeculativeSession, VerifyRow
    source = Path('results/phase3-models/smollm2-135m-instruct-8bit')
    audit = inspect_snapshot(source)
    available = psutil.virtual_memory().available
    if audit['weight_bytes'] > 200_000_000 or available < 2 * 1024**3:
        raise RuntimeError('Bounded resource preflight rejected')
    paths = [Path(__file__), Path('src/deepinfer_mac/varlen_verify.py'),
             Path('src/deepinfer_mac/ds/joint_budget.py'), Path('src/deepinfer_mac/adapters.py'),
             Path('src/deepinfer_mac/ds/mtp_varlen.py')]
    def hashes():
        return {x.resolve().relative_to(Path.cwd().resolve()).as_posix():
                hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    report = {'purpose': 'Independent same-target draft mechanism control', 'performance_claim': False,
              'process_count': 1, 'cycles_per_case': 3,
              'limits': ['Per-cycle work controls emit different token counts; timing is not equal-output end-to-end latency',
                         'Same-target proposal model is a mechanism control, not an acceptance-quality or speed result',
                         'One process with three internal cycles is not three independent benchmark processes',
                         'History case has no cost curve and uses full budget; no cost-driven adaptation is demonstrated'],
              'source_sha256': hashes(), 'model_content_sha256': audit['content_sha256'],
              'versions': {k: version(k) for k in ('mlx', 'mlx-lm')},
              'platform': platform.mac_ver()[0], 'available_before_bytes': available,
              'swap_before_bytes': read_swap_bytes(), 'native': [], 'cases': [], 'failures': []}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    model, tokenizer = load(source)
    prompts = ['Explain rain in one sentence.', 'Write a Python addition function.', '请简述水循环。']
    tokenized = [tokenizer.encode(x) for x in prompts]
    def cache_for(tokens):
        c = make_prompt_cache(model)
        model(mx.array([tokens[:-1]]), cache=c)
        mx.eval([x.state for x in c])
        return c
    for tokens in tokenized:
        cache, tail = cache_for(tokens), tokens[-1]
        output, margins = [], []
        for _ in range(13):
            logits = model(mx.array([[tail]]), cache=cache)[0, -1].astype(mx.float32)
            top = mx.sort(logits)[-2:]
            mx.eval(logits, top, [x.state for x in cache])
            tail = mx.argmax(logits).item()
            output.append(tail)
            margins.append((top[1] - top[0]).item())
        report['native'].append({'tokens': output, 'top_two_logit_margins': margins})
    for strategy, allocation in (('bucket', 'forced_control'), ('padded_control', 'forced_control'),
                                 ('shared_cap', 'forced_control'), ('bucket', 'history_no_curve')):
        rows = [VerifyRow(RequestKey(str(i), 0), cache_for(t), t[-1], (), 12)
                for i, t in enumerate(tokenized)]
        draft_caches = [cache_for(t) for t in tokenized]
        key = CostKey('same-target', 'same-target', '8bit', 3, 128, strategy, 'mechanism-control')
        session = JointSpeculativeSession(model, model, JointBudget(3, 'same-target'), key,
                                          draft_identity='same-target', calibration_identity='mechanism-control',
                                          mechanism_control=True, allocator='mlx')
        outputs = [[], [], []]
        case = {'strategy': strategy, 'allocation': allocation, 'cycles': [], 'outputs': outputs, 'matches_native': False}
        report['cases'].append(case)
        try:
            # Three rounds force actual heterogeneous lengths; output caps differ deliberately.
            for cycle in range(3):
                for i, row in enumerate(rows):
                    row.remaining = (1, 2, 4)[i] if cycle == 2 else 4
                result = session.tick(rows, draft_caches, strategy=strategy, forced_lengths=(1, 2, 4) if allocation == 'forced_control' else None)
                case['cycles'].append(result)
                for i, (row, emitted) in enumerate(zip(rows, result['outputs'])):
                    outputs[i].extend(emitted)
                    row.tail = emitted[-1]
                save()
            case['matches_native'] = all(out == ref['tokens'][:len(out)]
                                          for out, ref in zip(outputs, report['native']))
            case['cache_next_matches'] = []
            case['target_draft_next_agree'] = []
            case['accepted_history_state_controls'] = []
            case['first_native_divergences'] = []
            for i, (row, draft_cache) in enumerate(zip(rows, draft_caches)):
                target_logits = model(mx.array([[row.tail]]), cache=row.cache)[0, -1].astype(mx.float32)
                draft_logits = model(mx.array([[row.tail]]), cache=draft_cache)[0, -1].astype(mx.float32)
                reference_cache = cache_for(tokenized[i])
                # Independently rebuild the actually accepted history token by token.
                for consumed in [tokenized[i][-1], *outputs[i][:-1]]:
                    model(mx.array([[consumed]]), cache=reference_cache)
                    mx.eval([c.state for c in reference_cache])
                history_logits = model(mx.array([[row.tail]]), cache=reference_cache)[0, -1].astype(mx.float32)
                mx.eval(target_logits, draft_logits, history_logits)
                target_next = mx.argmax(target_logits).item()
                draft_next = mx.argmax(draft_logits).item()
                case['accepted_history_state_controls'].append({
                    'target_draft_logits_exact': mx.array_equal(target_logits, draft_logits).item(),
                    'target_draft_logits_max_abs': mx.max(mx.abs(target_logits - draft_logits)).item(),
                    'serial_accepted_history_logits_exact': mx.array_equal(target_logits, history_logits).item(),
                    'serial_accepted_history_logits_max_abs': mx.max(mx.abs(target_logits - history_logits)).item(),
                    'serial_accepted_history_next_token_matches': target_next == mx.argmax(history_logits).item()})
                case['target_draft_next_agree'].append(target_next == draft_next)
                difference = next((j for j, token in enumerate(outputs[i]) if token != report['native'][i]['tokens'][j]), None)
                case['first_native_divergences'].append(None if difference is None else {
                    'index': difference, 'actual_token': outputs[i][difference],
                    'native_token': report['native'][i]['tokens'][difference],
                    'native_top_two_margin': report['native'][i]['top_two_logit_margins'][difference]})
                case['cache_next_matches'].append(target_next == draft_next == report['native'][i]['tokens'][len(outputs[i])])
            if not case['matches_native'] or not all(case['cache_next_matches']):
                report['failures'].append({'strategy': strategy, 'reason': 'native token or next-state divergence'})
        except Exception as e:  # noqa: BLE001 -- preserve all actual mechanism failures
            report['failures'].append({'strategy': strategy, 'error_type': type(e).__name__, 'message': str(e)})
        save()
    report['forced_bucket_padded_outputs_equal'] = report['cases'][0]['outputs'] == report['cases'][1]['outputs']
    report.update({'source_unchanged': hashes() == report['source_sha256'],
                   'swap_after_bytes': read_swap_bytes(), 'metal_peak_bytes': mx.get_peak_memory(),
                   'rss_snapshot_bytes': psutil.Process().memory_info().rss,
                   'available_after_bytes': psutil.virtual_memory().available,
                   'completed_at_unix': time.time()})
    report['accepted'] = (len(report['cases']) == 4 and all(len(c['cycles']) == 3 for c in report['cases'])
                           and not report['failures'] and report['source_unchanged']
                           and report['swap_after_bytes'] == report['swap_before_bytes'])
    save()
    print(json.dumps({'accepted': report['accepted'], 'failures': report['failures'],
                      'metal_peak_bytes': report['metal_peak_bytes']}))
    if not report['accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
