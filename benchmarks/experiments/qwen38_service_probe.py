"""Probe an already running local server with real 27B baseline and SSE.

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
    ap.add_argument('--output', default='benchmarks/reports/qwen38-service-probe.json')
    args = ap.parse_args()
    report = {'source': source_manifest(), 'fake_engine': False, 'checks': {}, 'base_url': args.base_url}
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        model = next(m for m in client.get('/v1/models').json()['data'] if m['id'] == 'qwen3.8-27b-2bit')
        assert model['supported_modes'] == ['baseline'] and model['status'] == 'quality_failed'
        report['checks']['model'] = model
        prompt = '请用三句话解释统一内存为什么适合本地大语言模型推理。'
        body = {'model': model['id'], 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 64}
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
        control = json.loads(Path('benchmarks/reports/qwen38-quality-matrix-kexjos.json').read_text())
        native = next(c for c in control['cases'] if c['label'] == 'zh' and c['setting'] == 'greedy')
        assert direct['token_ids'] == native['token_ids'][:64]
        report['checks']['stream'] = {'events': lines, 'result': final, 'matches_non_stream': True, 'matches_native_control': True}
        unsupported = client.post('/v1/chat/completions', json={**body, 'mode': 'adaptive'})
        comparison = client.post('/api/compare', json={'profile': model['id'], 'prompt': prompt})
        assert unsupported.status_code == comparison.status_code == 422
        report['checks']['unsupported_modes'] = {'adaptive': unsupported.json(), 'compare': comparison.json()}
        assert client.get('/').status_code == client.get('/assets/app.js').status_code == 200
    report['engineering_passed'] = True
    report['quality_passed'] = False
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'engineering_passed': True, 'quality_passed': False, 'report': args.output}))


if __name__ == '__main__':
    main()
