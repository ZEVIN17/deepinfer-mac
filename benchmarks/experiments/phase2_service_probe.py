"""Probe an already running local server with a real selected model and SSE.

Start `deepinfer-mac serve --port 8876`, then run this script. This is an
engineering check; passing transport does not imply model quality passed.
"""
import argparse
import json
from pathlib import Path

import httpx

from deepinfer_mac.provenance import source_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:8876')
    ap.add_argument('--profile', default='qwen3.5-9b-8bit')
    ap.add_argument('--official-sampling', action='store_true')
    ap.add_argument('--control', default='benchmarks/reports/quality-qwen3.5-9b-8bit.json')
    ap.add_argument('--output', default='benchmarks/reports/phase2-service-qwen3.5-9b-8bit.json')
    args = ap.parse_args()
    report = {'source': source_manifest(), 'fake_engine': False, 'checks': {}, 'base_url': args.base_url}
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        model = next(m for m in client.get('/v1/models').json()['data'] if m['id'] == args.profile)
        assert model['supported_modes'] == ['baseline']
        report['checks']['model'] = model
        prompt = '请用三句话解释统一内存为什么适合本地大语言模型推理。'
        body = {'model': model['id'], 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 64}
        if args.official_sampling:
            control = json.loads(Path(args.control).read_text())
            expected = next(c for c in control['cases'] if c['task_id'] == 'heldout-logic' and c['setting'] == 'official_non_thinking')
            body.update({'messages': expected['messages'], 'max_tokens': 128, 'temperature': .7,
                         'top_p': .8, 'top_k': 20, 'presence_penalty': 1.5, 'seed': 42})
        response = client.post('/v1/chat/completions', json=body)
        response.raise_for_status()
        first = response.json()
        report['checks']['non_stream'] = first
        direct = client.get('/api/runs/' + first['deepinfer']['run_id']).json()['result']
        report['checks']['non_stream_result'] = direct
        stream_text = ''
        lines = []
        with client.stream('POST', '/v1/chat/completions', json={**body, 'stream': True}) as stream:
            stream.raise_for_status()
            run_id = stream.headers['X-Run-ID']
            for line in stream.iter_lines():
                if line.startswith('data: '):
                    payload = line[6:]
                    lines.append(payload)
                    if payload != '[DONE]':
                        event = json.loads(payload)
                        assert 'error' not in event, event
                        stream_text += event.get('choices', [{}])[0].get('delta', {}).get('content', '')
        final = client.get('/api/runs/' + run_id).json()['result']
        assert lines[-1] == '[DONE]'
        assert stream_text == first['choices'][0]['message']['content'] == final['text']
        assert final['token_ids'] == direct['token_ids']
        control = json.loads(Path(args.control).read_text())
        task_id = 'heldout-logic' if args.official_sampling else 'dev-zh-memory'
        setting = 'official_non_thinking' if args.official_sampling else 'greedy_engine'
        native = next(c for c in control['cases'] if c['task_id'] == task_id and c['setting'] == setting)
        assert direct['token_ids'] == native['output_token_ids'][:body['max_tokens']]
        report['checks']['stream'] = {'events': lines, 'result': final, 'matches_non_stream': True, 'matches_native_control': True}
        unsupported = client.post('/v1/chat/completions', json={**body, 'mode': 'adaptive'})
        comparison = client.post('/api/compare', json={'profile': model['id'], 'prompt': prompt})
        assert unsupported.status_code == comparison.status_code == 422
        report['checks']['unsupported_modes'] = {'adaptive': unsupported.json(), 'compare': comparison.json()}
        assert client.get('/').status_code == client.get('/assets/app.js').status_code == 200
        # Real browser-style disconnect must cancel the single owner, then recover.
        with client.stream('POST', '/v1/chat/completions', json={**body, 'stream': True, 'max_tokens': 1024, 'messages': [{'role': 'user', 'content': 'Write100numbered paragraphs explaining algorithms in detail.'}]}) as stream:
            stream.raise_for_status()
            cancelled_id = stream.headers['X-Run-ID']
            for line in stream.iter_lines():
                if '"content"' in line:
                    break
        import time
        for _ in range(300):
            cancelled = client.get('/api/runs/' + cancelled_id).json()
            if cancelled['status'] in ('completed', 'cancelled', 'failed'):
                break
            time.sleep(.05)
        assert cancelled['status'] == 'cancelled', cancelled
        report['checks']['disconnect_cancellation'] = cancelled
        recovery = client.post('/v1/chat/completions', json={**body, 'max_tokens': 16})
        recovery.raise_for_status()
        report['checks']['recovered_after_cancel'] = recovery.json()
    report['engineering_passed'] = True
    report['quality_scope'] = 'Engineering transport and token/cache consistency only; see independent quality matrix for model capability.'
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'engineering_passed': True, 'report': args.output}))


if __name__ == '__main__':
    main()
