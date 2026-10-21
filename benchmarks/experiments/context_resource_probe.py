"""Explicit real long-context resource probe; never truncates input silently."""
import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import psutil

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.models import get_profile
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='qwen3.5-9b-8bit')
    ap.add_argument('--input-tokens', type=int, default=8300)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--experimental-context-limit', type=int, default=9216)
    ap.add_argument('--output', default='benchmarks/reports/context-8300-256-qwen35-9b8.json')
    args = ap.parse_args()
    if args.input_tokens + args.max_tokens > args.experimental_context_limit:
        raise ValueError('Explicit experimental context limit does not cover the complete request')
    import mlx.core as mx

    profile = get_profile(args.profile)
    original_limit = profile['context_limit']
    report = {'source': source_manifest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'profile': args.profile, 'created_unix': time.time(), 'status': 'running',
              'scope': 'One explicitly overridden context-resource check, not full stress coverage or model advertised context validation',
              'configured_context_limit': original_limit, 'experimental_context_limit': args.experimental_context_limit,
              'requested_input_tokens': args.input_tokens, 'requested_output_tokens': args.max_tokens,
              'system_before': {'available_bytes': psutil.virtual_memory().available, 'swap_used_bytes': psutil.swap_memory().used}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    save()
    engine = InferenceEngine(cache_budget_bytes=512*1024**2, checkpoint_interval_tokens=512)
    try:
        profile['context_limit'] = args.experimental_context_limit
        mx.reset_peak_memory()
        started = time.perf_counter()
        engine._load(args.profile, False)
        report['load'] = {'seconds': time.perf_counter()-started, 'peak_metal_bytes': mx.get_peak_memory(),
                          'active_metal_bytes': mx.get_active_memory(), 'rss_bytes': psutil.Process().memory_info().rss,
                          'swap_used_bytes': psutil.swap_memory().used}
        suffix = '\nExplain local large-language-model inference in at least600words. Write continuous detailed prose and continue until the explanation is complete.'
        repeats = args.input_tokens
        request = None
        for _ in range(12):
            request = GenerationRequest(profile=args.profile,
                                        prompt='Public resource fixture: the repeated background below is irrelevant.\n'+' background'*repeats+suffix,
                                        max_tokens=args.max_tokens, prefill_step_size=256, seed=42)
            actual = len(engine._tokens(request))
            if actual == args.input_tokens:
                break
            repeats += args.input_tokens-actual
            if repeats < 0:
                raise ValueError('Requested input is shorter than the fixed wrapper/instruction')
        else:
            raise ValueError('Could not construct exact token length; input was not truncated')
        report['request'] = dataclasses.asdict(request)
        report['prompt_token_ids'] = engine._tokens(request)
        report['actual_input_tokens'] = len(report['prompt_token_ids'])
        save()
        events = []
        result = engine.generate(request, emit=events.append)
        report['result'] = dataclasses.asdict(result)
        report['stream_text_matches'] = ''.join(e.get('text', '') for e in events) == result.text
        report['actual_output_tokens'] = len(result.token_ids)
        report['completed_configured_length'] = len(result.token_ids) == args.max_tokens
        report['system_after'] = {'available_bytes': psutil.virtual_memory().available, 'swap_used_bytes': psutil.swap_memory().used}
        report['status'] = 'measured'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        engine.close()
        profile['context_limit'] = original_limit
        save()
    print(json.dumps({'status': report['status'], 'input': report.get('actual_input_tokens'),
                      'output': report.get('actual_output_tokens'), 'metrics': report.get('result', {}).get('metrics')}))


if __name__ == '__main__':
    main()
