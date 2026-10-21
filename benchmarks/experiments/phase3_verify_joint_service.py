"""Independent opt-in actual ASGI and joint lifecycle probe; no listening port.

Requires coordinator GPU baton. Uses only existing audited small pair. Timings
are diagnostics, never comparative performance evidence. Raw outputs stay private.
"""
import argparse
import hashlib
import json
import threading
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/phase3-joint-service/actual.json'))
    args = parser.parse_args()
    if not args.execute:
        print('Plan: native/direct/API/SSE/concurrent/cancellation/cleanup; no model loaded')
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    import mlx.core as mx
    import psutil
    from fastapi.testclient import TestClient

    from deepinfer_mac.adapters import ModelRegistry, ModelSpec, inspect_snapshot
    from deepinfer_mac.continuous import ContinuousConfig
    from deepinfer_mac.engine import InferenceEngine, read_swap_bytes
    from deepinfer_mac.joint_service import JointServiceConfig, MLXJointBatchSession
    from deepinfer_mac.server.app import create_app
    from deepinfer_mac.state import GenerationRequest
    config_path = Path('examples/models/research-smollm2-pair.local.json')
    spec = ModelSpec.from_json(config_path)
    settings = json.loads(config_path.read_text())
    for source in (settings['source'], settings['draft']):
        audit = inspect_snapshot(Path(source))
        assert audit['weight_bytes'] < 310_000_000 and audit['config']['model_type'] == 'llama'
    registry = ModelRegistry([spec])
    cfg = ContinuousConfig(max_active=2, prefill_chunk=32, max_context_tokens=4096)
    joint = JointServiceConfig(settings['id'])
    factory = lambda: InferenceEngine(model_registry=registry, cache_budget_bytes=0)
    prompts = ['Explain why rain falls.', 'Write a Python function adding two integers.', '请解释为什么天空是蓝色的。']
    requests = [GenerationRequest(profile=settings['id'], messages=[{'role': 'user', 'content': p}], max_tokens=16, use_cache=False) for p in prompts]
    sources = [*Path('src/deepinfer_mac').rglob('*.py'), Path(__file__), config_path]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    report = {'scope': 'actual small pair functional service; no calibrated curve or speed claim', 'source_sha256': hashes(),
              'versions': {k: version(k) for k in ('mlx', 'mlx-lm')}, 'checks': {}, 'native': [], 'direct': [], 'jobs': [], 'failures': [],
              'swap_before': read_swap_bytes(), 'available_before': psutil.virtual_memory().available}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    def check(name, value):
        report['checks'][name] = bool(value)
        save()
    engine = None
    try:
        engine = factory()
        for req in requests:
            report['native'].append(engine.native_generate(req).token_ids)
            save()
        engine.close()
        engine = None
        # Direct real JointSpeculativeSession invoked by adapter; identical prompt tail/cache setup.
        for indices in ([0], [1], [2], [0, 1]):
            engine = factory()
            session = MLXJointBatchSession(engine, settings['id'], cfg, joint=joint)
            outputs = {i: [] for i in indices}
            mapping = {}
            for i in indices:
                prepared = session.prepare(requests[i])
                resource = session.resources()
                reason = session.budget.check(str(i), prepared, requests[i].max_tokens, **resource)
                if reason:
                    raise RuntimeError('direct admission rejected: '+reason)
                session.budget.reserve(str(i), prepared, requests[i].max_tokens)
                mapping[session.insert(prepared, requests[i])] = i
            while session.rows:
                _, tokens = session.tick()
                for token in tokens:
                    outputs[mapping[token.uid]].append(token.token)
            report['direct'].append({'indices': indices, 'tokens': outputs})
            session.close()
            engine.close()
            engine = None
            save()
        control = {'gate': False, 'cancel_prefill': False, 'cancel_emit': False, 'arrival': False, 'forced_eos': False}
        entered, proceed = threading.Event(), threading.Event()
        class ObservedSession(MLXJointBatchSession):
            def prepare(self, req):
                prepared = super().prepare(req)
                if control['forced_eos']:
                    prepared.eos_tokens = {report['native'][0][0]}
                if control['gate']:
                    control['gate'] = False
                    entered.set()
                    if not proceed.wait(30):
                        raise TimeoutError('probe preparation gate')
                return prepared
            def insert_request(self, prepared, req, job):
                if control['arrival']:
                    control['arrival'] = False
                    original_arrival = job.emit
                    arrived = False
                    def emit_arrival(event):
                        nonlocal arrived
                        original_arrival(event)
                        if event.get('type') == 'token' and not arrived:
                            arrived = True
                            control['arrival_job'] = worker.submit(requests[1])
                    job.emit = emit_arrival
                if control['cancel_emit']:
                    control['cancel_emit'] = False
                    original = job.emit
                    def emit(event):
                        original(event)
                        if event.get('type') == 'token':
                            job.cancel.set()
                    job.emit = emit
                if control['cancel_prefill'] and len(prepared.tokens) > 100:
                    from deepinfer_mac.ds.joint_budget import RequestKey
                    calls = 0
                    def cancel():
                        nonlocal calls
                        calls += 1
                        if calls == 2:
                            job.cancel.set()
                        return job.cancel.is_set()
                    return self.insert(prepared, req, key=RequestKey(job.id, job.slot_generation), cancel=cancel)
                return super().insert_request(prepared, req, job)
        app = create_app(engine_factory=factory, model_registry=registry, experimental_joint_verification=True,
                         joint_config=joint, continuous_config=cfg, batch_session_factory=lambda e, p, c: ObservedSession(e, p, c, joint=joint))
        with TestClient(app) as client:
            worker = app.state.worker
            def collect(job):
                if not job.done.wait(120):
                    worker.cancel_job(job.id)
                    raise TimeoutError('actual job')
                report['jobs'].append({'status': job.status, 'result': job.result, 'error': job.error,
                                       'events': list(job.events), 'id': job.id})
                save()
                return job.result
            payload = {'model': settings['id'], 'messages': requests[0].messages, 'max_tokens': 16, 'use_cache': False}
            check('health_explicit', client.get('/health').json()['execution_backend'] == 'joint_experimental')
            check('sampling_422', client.post('/v1/chat/completions', json={**payload, 'temperature': .2}).status_code == 422)
            check('profile_422', client.post('/v1/chat/completions', json={**payload, 'model': 'qwen3-4b'}).status_code == 422)
            check('compare_422', client.post('/api/compare', json={'prompt': 'x', 'profile': settings['id']}).status_code == 422)
            for i, req in enumerate(requests):
                response = client.post('/v1/chat/completions', json={**payload, 'messages': req.messages})
                check(f'api_{i}_200', response.status_code == 200)
                body = response.json()
                report.setdefault('api_responses', []).append({'status': response.status_code, 'body': body})
                save()
                if response.status_code != 200:
                    raise RuntimeError('API failed: '+json.dumps(body))
                result = collect(worker.get(body['deepinfer']['run_id']))
                check(f'api_{i}_direct_equal', result['token_ids'] == report['direct'][i]['tokens'][i])
                check(f'api_{i}_native_equal', result['token_ids'] == report['native'][i])
                check(f'api_{i}_bound', len(result['token_ids']) <= 16)
                check(f'api_{i}_fullbudget_label', result['metrics']['cost_curve_available'] is False)
            streamed = client.post('/v1/chat/completions', json={**payload, 'stream': True})
            chunks = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
            text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks)
            stream_result = collect(worker.get(streamed.headers['x-run-id']))
            check('sse_text_done', text == stream_result['text'] and streamed.text.count('data: [DONE]') == 1)
            # Gate only host admission so both actual rows enter the same first tick.
            entered.clear(); proceed.clear(); control['gate'] = True
            first = worker.submit(requests[0])
            assert entered.wait(30)
            second = worker.submit(requests[1])
            proceed.set()
            concurrent = [collect(first), collect(second)]
            for i, result in enumerate(concurrent):
                check(f'concurrent_{i}_direct_equal', result['token_ids'] == report['direct'][3]['tokens'][i])
                foreign = second.id if i == 0 else first.id
                metric_text = json.dumps(result['metrics'])
                check(f'concurrent_{i}_privacy', foreign not in metric_text and not any(k in result['metrics']['last_joint_round'] for k in ('outputs', 'outcomes', 'request_id', 'seqlens_q')))
            entered.clear(); proceed.clear(); control.update(gate=True, cancel_prefill=True)
            existing = worker.submit(requests[0])
            assert entered.wait(30)
            newcomer = worker.submit(replace(requests[1], messages=[{'role': 'user', 'content': 'water clouds rain ' * 120}]))
            proceed.set()
            result = collect(existing)
            collect(newcomer)
            check('cancel_prefill_isolated', existing.status == 'completed' and newcomer.status == 'cancelled' and result['token_ids'] == report['direct'][0]['tokens'][0])
            control['cancel_prefill'] = False
            control['cancel_emit'] = True
            cancelled = worker.submit(requests[0])
            collect(cancelled)
            emitted = [e for e in cancelled.events if e['type'] == 'token']
            check('cancel_between_tokens', cancelled.status == 'cancelled' and len(emitted) == 1 and cancelled.result is None)
            recovered = collect(worker.submit(requests[0]))
            check('recovery', recovered['token_ids'] == report['direct'][0]['tokens'][0])
            control['arrival'] = True
            arrival_first = worker.submit(requests[0])
            arrival_result = collect(arrival_first)
            arrival_second = control.get('arrival_job')
            arrival_other = collect(arrival_second) if arrival_second else None
            check('decode_new_arrival', arrival_result is not None and arrival_other is not None and arrival_first.submitted_at < arrival_second.submitted_at)
            check('decode_new_arrival_native_0', arrival_result['token_ids'] == report['native'][0])
            check('decode_new_arrival_native_1', arrival_other['token_ids'] == report['native'][1])
            one = collect(worker.submit(replace(requests[0], max_tokens=1)))
            check('actual_one_token_limit', one['token_ids'] == report['native'][0][:1] and one['finish_reason'] == 'length')
            control['forced_eos'] = True
            eos = collect(worker.submit(requests[0]))
            control['forced_eos'] = False
            # Real model output with controlled EOS membership; not natural EOS-quality evidence.
            check('actual_controlled_eos', eos['token_ids'] == report['native'][0][:1] and eos['finish_reason'] == 'stop')
        check('closed', worker.engine is None and worker.session is None and not worker.active_requests and len(worker.free_slots) == 2 and not worker.cleanup_errors)
    except Exception as exc:  # noqa: BLE001 — retain failed actual execution
        report['failures'].append({'type': type(exc).__name__, 'message': str(exc).replace(str(Path.cwd()), '$REPO')})
    finally:
        if engine is not None:
            engine.close()
        mx.synchronize(); mx.clear_cache()
        report.update(swap_after=read_swap_bytes(), metal_peak_bytes=mx.get_peak_memory(), metal_active_after=mx.get_active_memory(), metal_cache_after=mx.get_cache_memory())
        check('source_unchanged', hashes() == report['source_sha256'])
        check('no_swap_growth', report['swap_after'] == report['swap_before'])
        report['accepted'] = len(report['checks']) >= 25 and all(report['checks'].values()) and not report['failures']
        save()
    print(json.dumps({'accepted': report['accepted'], 'checks': report['checks'], 'failures': report['failures']}, indent=2))


if __name__ == '__main__':
    main()
