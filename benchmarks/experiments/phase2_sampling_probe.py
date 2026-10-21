"""Real same-seed sampling controls across native/manual prefill/cache hits."""
import argparse
import dataclasses
import json
from pathlib import Path

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='qwen3.5-9b-8bit')
    ap.add_argument('--output', default='benchmarks/reports/phase2-sampling-qwen35-9b8.json')
    args = ap.parse_args()
    engine = InferenceEngine(checkpoint_interval_tokens=256)
    report = {'source': source_manifest(), 'profile': args.profile, 'cases': [],
              'scope': 'Sampling correctness/control, not competitive benchmark'}
    path = Path(args.output)
    try:
        for seed in (0, 42, 2**32-1):
            request = GenerationRequest(profile=args.profile,
                                        prompt='A box contains3red balls and2blue balls. Add4blue and remove1red. Explain the remaining total briefly.',
                                        max_tokens=48, prefill_step_size=256,
                                        temperature=.7, top_p=.8, top_k=20, presence_penalty=1.5, seed=seed)
            engine.clear_cache()
            expected = None
            for label, mode, cache in [('engine_cold', 'baseline', True), ('engine_hot', 'baseline', True),
                                       ('engine_no_cache', 'baseline', False), ('native_no_cache', 'native_mlx', False),
                                       ('native_hot', 'native_mlx', True)]:
                result = engine.generate(dataclasses.replace(request, mode=mode, use_cache=cache))
                expected = result.token_ids if expected is None else expected
                report['cases'].append({'seed': seed, 'label': label, 'matches_first': result.token_ids == expected,
                                        **dataclasses.asdict(result)})
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                print(json.dumps({'seed': seed, 'label': label, 'matches_first': result.token_ids == expected,
                                  'cache_hit': result.metrics['cache_hit_tokens']}), flush=True)
        report['all_equal'] = all(c['matches_first'] for c in report['cases'])
        report['status'] = 'measured'
    finally:
        engine.close()
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
