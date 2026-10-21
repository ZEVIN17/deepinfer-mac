"""Independent real SmolLM2 adapter verification; no comparative performance claim."""
import gc
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path.cwd()
REPORT = ROOT / 'benchmarks/reports/phase3-model-adapter-integration.json'
EVIDENCE = ROOT / 'docs/evidence/phase3-adapter-verification.json'
CONFIG = 'examples/models/smollm2-135m-8bit.local.json'
MODEL_ID = 'smollm2-135m-8bit'
MODEL_PATH = 'results/phase3-models/smollm2-135m-instruct-8bit'
paths = sorted({*ROOT.glob('src/deepinfer_mac/**/*.py'), ROOT/'tests/test_model_adapters.py', ROOT/CONFIG, ROOT/'pyproject.toml', Path(__file__).resolve()})
source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
report = {'schema_version':1, 'purpose':'Real small-model baseline compatibility, identity and lifecycle verification; not a speed or capability ranking.',
          'model':{'id':MODEL_ID,'local_path':MODEL_PATH,'family':'LlamaForCausalLM','weight_quantization':'8bit group64'},
          'source_sha256':source,'started_at_unix':time.time(),'steps':[], 'resource_policy':'Only SmolLM2-135M-8bit; normal registration/load/KV guards retained; no user service mutation.',
          'cpu_validation':{'portable_pytest':{'command':'.venv/bin/python -m pytest -q -m "not metal"','passed':270,'metal_deselected':41},
                            'ruff':{'command':'.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality','passed':True}}}
def save():
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def step(name, fn):
    started=time.time()
    try:
        details=fn()
        report['steps'].append({'name':name,'passed':True,'elapsed_seconds':time.time()-started,'details':details})
        print(name+': PASS',flush=True)
    except Exception as exc:
        report['steps'].append({'name':name,'passed':False,'elapsed_seconds':time.time()-started,'error':type(exc).__name__+': '+str(exc).replace(str(ROOT),'$REPO')})
        save()
        raise
    save()
    return details

try:
    cpu = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-m', 'not metal'], text=True, capture_output=True, timeout=90, check=False)
    assert cpu.returncode == 0, cpu.stdout + cpu.stderr
    matched = re.search(r'(\d+) passed, (\d+) deselected', cpu.stdout)
    assert matched, cpu.stdout
    ruff = subprocess.run([str(ROOT/'.venv/bin/ruff'), 'check', 'src', 'tests', 'scripts', 'benchmarks/experiments', 'benchmarks/quality'], text=True, capture_output=True, timeout=60, check=False)
    assert ruff.returncode == 0, ruff.stdout + ruff.stderr
    report['cpu_validation']['portable_pytest'].update(passed=int(matched.group(1)), metal_deselected=int(matched.group(2)))
    report['cpu_validation']['portable_pytest']['output'] = cpu.stdout.strip()
    report['cpu_validation']['ruff']['output'] = ruff.stdout.strip()
    import mlx.core as mx
    import psutil
    from fastapi.testclient import TestClient
    from mlx_lm.generate import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    from deepinfer_mac.adapters import ModelRegistry, ModelSpec, inspect_snapshot
    from deepinfer_mac.engine import InferenceEngine, read_swap_bytes
    from deepinfer_mac.server.app import create_app
    from deepinfer_mac.state import GenerationRequest
    audit = inspect_snapshot(MODEL_PATH)
    report['model'].update({k:audit[k] for k in ('weight_bytes','content_sha256','files')})
    report['versions']={'python':sys.version.split()[0],'mlx':__import__('importlib.metadata',fromlist=['version']).version('mlx'), 'mlx_lm':__import__('importlib.metadata',fromlist=['version']).version('mlx-lm')}
    available=psutil.virtual_memory().available
    required=audit['weight_bytes']+2*1024**3
    report['resources_before']={'available_bytes':available,'swap_used_bytes':read_swap_bytes(),'rss_bytes':psutil.Process().memory_info().rss,'required_weights_plus_runtime_bytes':required}
    if required > available*.85:
        raise MemoryError('Normal model admission headroom unavailable')
    registry=ModelRegistry([ModelSpec.from_json(CONFIG)])
    cli_listing=json.loads(subprocess.check_output([sys.executable,'-m','deepinfer_mac.cli','models','--model-config',CONFIG,'--json'],text=True))
    public=next(x for x in cli_listing if x['id']==MODEL_ID)
    assert public['target']==registry.list()[-1]['target']
    assert str(ROOT) not in json.dumps(public)
    step('python_cli_registration_identity',lambda:{'target':public['target'],'capabilities':public['capabilities'],'context_limit':public['context_limit']})
    engine=InferenceEngine(model_registry=registry,cache_budget_bytes=64*1024**2)
    metrics=[]
    peaks={'rss_bytes':0,'metal_active_bytes':0,'metal_peak_bytes':0}
    stop=threading.Event()
    def monitor():
        while not stop.wait(.01):
            peaks['rss_bytes']=max(peaks['rss_bytes'],psutil.Process().memory_info().rss)
    watcher=threading.Thread(target=monitor,daemon=True); watcher.start()
    english=GenerationRequest(profile=MODEL_ID,prompt='Explain why the sky looks blue in one short paragraph.',max_tokens=32,use_cache=False)
    chinese=replace(english,prompt='请用两句话解释为什么天空是蓝色的。',max_tokens=64)
    baseline={}
    native={}
    def controls():
        for name,req in [('english',english),('chinese',chinese)]:
            result=engine.generate(req)
            metrics.append({'case':name+'_baseline','metrics':result.metrics})
            tokens=engine._tokens(req)
            direct=list(stream_generate(engine.model,engine.tokenizer,tokens,max_tokens=req.max_tokens,sampler=make_sampler(temp=0),prefill_step_size=req.prefill_step_size))
            direct_ids=[int(o.token) for o in direct]
            assert result.token_ids==direct_ids,(name,result.token_ids,direct_ids)
            native[name]={'token_ids':direct_ids,'text':''.join(o.text for o in direct)}
            baseline[name]=asdict(result)
        return [{'case':name,'prompt_token_ids':engine._tokens(req),'max_tokens':req.max_tokens,'output_token_ids':baseline[name]['token_ids'],'complete_output_matches_native':True,'template_sha256':hashlib.sha256(engine.tokenizer.chat_template.encode()).hexdigest()} for name,req in [('english',english),('chinese',chinese)]]
    step('actual_baseline_vs_native_greedy_english_chinese',controls)
    def fallback():
        no_load=InferenceEngine(model_registry=registry)
        try:
            try:no_load.generate(replace(english,mode='adaptive'))
            except ValueError:pass
            else:raise AssertionError('unsupported mode accepted')
            assert no_load.model is None
            denied_before_load=True
        finally:no_load.close()
        result=engine.generate(replace(english,mode='adaptive',fallback_to_baseline=True))
        assert result.token_ids==baseline['english']['token_ids']
        assert result.metrics['requested_mode']=='adaptive' and result.metrics['effective_mode']=='baseline' and result.metrics['capability_fallback']
        metrics.append({'case':'explicit_fallback','metrics':result.metrics})
        return {'unsupported_rejected_before_array_load':denied_before_load,'requested_mode':'adaptive','effective_mode':'baseline','reason':result.metrics['capability_fallback'],'tokens_match_baseline':True}
    step('actual_capability_guard_and_explicit_fallback',fallback)
    def cache_branches():
        prefix='Facts for this local test: water freezes at zero degrees Celsius at ordinary pressure. ' * 12
        requests=[replace(english,messages=[{'role':'system','content':prefix},{'role':'user','content':text}],prompt='',use_cache=True,max_tokens=32) for text in ('Summarize the freezing fact.','Which temperature freezes water?','Summarize the freezing fact.')]
        evidence=[]
        for i,req in enumerate(requests):
            warm=engine.generate(req)
            cold=engine.generate(replace(req,use_cache=False))
            assert warm.token_ids==cold.token_ids
            metrics.extend([{'case':f'cache_branch_{i}','metrics':warm.metrics},{'case':f'cold_branch_{i}','metrics':cold.metrics}])
            evidence.append({'case':i,'hit_tokens':warm.metrics['cache_hit_tokens'],'prompt_tokens':warm.prompt_tokens,'output_token_ids':warm.token_ids,'matches_cold':True})
        assert evidence[1]['hit_tokens']>0 and evidence[2]['hit_tokens']>0
        return evidence
    step('actual_dense_cache_branch_reuse',cache_branches)
    def cancellation():
        cancel=threading.Event(); emitted=[]
        def emit(event):
            if event.get('type')=='token' and event.get('token') is not None:
                emitted.append(event['token'])
                if len(emitted)>=3:cancel.set()
        try:engine.generate(replace(english,use_cache=True,max_tokens=64),emit=emit,cancel=cancel)
        except InterruptedError:pass
        else:raise AssertionError('cancel was not observed')
        follow=engine.generate(replace(english,use_cache=True))
        assert follow.token_ids==baseline['english']['token_ids']
        metrics.append({'case':'post_cancel_recovery','metrics':follow.metrics})
        return {'cancelled_after_emitted_tokens':len(emitted),'followup_matches_clean_baseline':True,'cache_hit_tokens':follow.metrics['cache_hit_tokens']}
    step('actual_decode_cancel_and_recovery',cancellation)
    def snapshot_restore():
        caches=make_prompt_cache(engine.model)
        prompt_ids=engine._tokens(english)
        engine.model(mx.array(prompt_ids[:-1],mx.uint32)[None],cache=caches)
        mx.eval([c.state for c in caches])
        adapter=registry.adapter(MODEL_ID).cache
        snapshot=adapter.snapshot(caches)
        expected=engine.model(mx.array(prompt_ids[-1:],mx.uint32)[None],cache=caches)
        mx.eval(expected)
        engine.model(mx.array([10,11],mx.uint32)[None],cache=caches)
        mx.eval([c.state for c in caches])
        adapter.restore(caches,snapshot)
        recovered=engine.model(mx.array(prompt_ids[-1:],mx.uint32)[None],cache=caches)
        mx.eval(recovered)
        assert bool(mx.all(expected==recovered).item())
        return {'cache_types':sorted({type(c).__module__+'.'+type(c).__name__ for c in caches}),
                'capabilities':adapter.capabilities(caches),'full_next_token_logits_exact_after_mutation_and_restore':True}
    step('actual_dense_adapter_snapshot_restore',snapshot_restore)
    step('resolved_public_metadata_redacted',lambda:{'model_identity':baseline['english']['metrics']['model_identity'],'cache_capabilities':baseline['english']['metrics']['cache_capabilities'],'public_target':baseline['english']['model']['target']})
    engine.close(); engine=None; gc.collect(); mx.clear_cache()
    def cli_generation():
        cmd=[sys.executable,'-m','deepinfer_mac.cli','run','--model-config',CONFIG,'--profile',MODEL_ID,'--mode','adaptive','--fallback-to-baseline','--max-tokens','32','--no-cache','--json',english.prompt]
        reply=json.loads(subprocess.check_output(cmd,text=True,timeout=90))
        assert reply['token_ids']==baseline['english']['token_ids']
        assert reply['metrics']['model_identity']==baseline['english']['metrics']['model_identity']
        assert reply['metrics']['effective_mode']=='baseline'
        assert str(ROOT) not in json.dumps(reply)
        metrics.append({'case':'cli_explicit_fallback','metrics':reply['metrics']})
        return {'command':' '.join(cmd).replace(str(ROOT),'$REPO'),'model':reply['profile'],'token_ids':reply['token_ids'],'same_model_identity':True,'effective_mode':reply['metrics']['effective_mode']}
    step('actual_cli_generation',cli_generation)
    def api_generation():
        api_registry=ModelRegistry([ModelSpec.from_json(CONFIG)])
        app=create_app(model_registry=api_registry)
        with TestClient(app) as client:
            listing=client.get('/v1/models').json()['data']
            listed=next(x for x in listing if x['id']==MODEL_ID)
            assert listed['target']==public['target']
            assert app.state.worker.engine is None
            body={'model':MODEL_ID,'messages':[{'role':'user','content':english.prompt}],'max_tokens':32,'temperature':0,'use_cache':False}
            rejected=client.post('/v1/chat/completions',json={**body,'mode':'adaptive'})
            assert rejected.status_code==422 and app.state.worker.engine is None
            reply=client.post('/v1/chat/completions',json={**body,'mode':'adaptive','fallback_to_baseline':True})
            assert reply.status_code==200,reply.text
            data=reply.json(); run=client.get('/api/runs/'+data['deepinfer']['run_id']).json()
            assert run['result']['token_ids']==baseline['english']['token_ids']
            assert data['deepinfer']['metrics']['model_identity']==baseline['english']['metrics']['model_identity']
            metrics.append({'case':'api_fallback','metrics':data['deepinfer']['metrics']})
            streamed=client.post('/v1/chat/completions',json={**body,'stream':True})
            assert streamed.status_code==200
            events=[]
            for line in streamed.text.splitlines():
                if line.startswith('data: ') and line!='data: [DONE]':events.append(json.loads(line[6:]))
            assert 'data: [DONE]' in streamed.text
            assert not any('error' in e for e in events)
            content=''.join(e['choices'][0]['delta'].get('content','') for e in events)
            assert content==data['choices'][0]['message']['content']
            stream_run=client.get('/api/runs/'+streamed.headers['x-run-id']).json()
            assert stream_run['result']['token_ids']==baseline['english']['token_ids']
            metrics.append({'case':'api_sse','metrics':stream_run['result']['metrics']})
            assert str(ROOT) not in json.dumps(data)
            result={'transport':'FastAPI TestClient, actual MLX worker','api_model_id':data['model'],'same_python_cli_api_target':True,'same_generation_identity':True,'unsupported_422_before_engine_creation':True,'explicit_fallback':'baseline','sse_events':len(events),'sse_terminal_done':True,'sse_text_and_token_ids_match_nonstream_and_python':True}
        assert app.state.worker.engine is None and not app.state.worker.thread.is_alive()
        result['owned_worker_closed']=True
        return result
    step('actual_api_identity_guard_fallback_and_sse',api_generation)
    stop.set(); watcher.join(); gc.collect(); mx.clear_cache()
    peaks['metal_active_bytes_after_cleanup']=mx.get_active_memory()
    peaks['metal_peak_bytes']=max(m['metrics']['peak_memory_bytes'] for m in metrics)
    peaks['rss_bytes']=max(peaks['rss_bytes'], *(m['metrics']['rss_bytes'] for m in metrics))
    peaks['scope']='RSS is parent 10ms samples plus per-request parent/CLI observations; Metal is maximum per-request peak across Python, CLI and API. Not an absolute OS peak.'
    report['generation_metrics']=metrics
    report['resources_after']={'available_bytes':psutil.virtual_memory().available,'swap_used_bytes':read_swap_bytes(),'rss_bytes':psutil.Process().memory_info().rss,'metal_active_bytes':mx.get_active_memory(),'metal_cached_bytes':mx.get_cache_memory()}
    report['observed_peaks']=peaks
    before=report['resources_before']['swap_used_bytes']; after=report['resources_after']['swap_used_bytes']
    report['overall_system_swap_delta_bytes']=after-before if before is not None and after is not None else None
    changed=[name for name,sha in source.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    report['frozen_source_unchanged']=not changed
    report['source_changes_during_verification']=changed
    report['passed']=all(s['passed'] for s in report['steps']) and not changed
    report['accepted']=report['passed']
    report['acceptance_status']='verified_current_source' if report['passed'] else 'failed_current_source'
except Exception as exc:  # noqa: BLE001 — record verification failure before process exit
    report['passed']=False
    report['fatal_error']=(type(exc).__name__+': '+str(exc)).replace(str(ROOT),'$REPO')
    print(traceback.format_exc().replace(str(ROOT),'$REPO'),file=sys.stderr)
finally:
    report['ended_at_unix']=time.time()
    report['cleanup']={'user_port_8765_touched':False,'new_listening_service_started':False,'all_model_processes_owned_by_this_probe_exit_on_completion':True}
    save()
    evidence={'schema_version':1,'report':'benchmarks/reports/phase3-model-adapter-integration.json','report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),'passed':report['passed'],'accepted':report.get('accepted',False),'acceptance_status':report.get('acceptance_status','verification_failed'),'actual_model':report['model'],'source_sha256':source,'steps':report['steps'],'commands':['.venv/bin/python -m pytest -q -m "not metal"','.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality','.venv/bin/python benchmarks/experiments/phase3_verify_model_adapters.py'],'limitations':['Only SmolLM2-135M-8bit actual new-family dense baseline tested; no larger model loaded.','Not a quality benchmark, optimization validation, comparative latency measurement, or broad architecture support proof.','Existing hybrid optimized tests are portable regression coverage; no new hybrid actual execution in this probe.'],'cleanup':report['cleanup']}
    EVIDENCE.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print('verification_passed='+str(report['passed']),flush=True)

if not report['passed']:
    raise SystemExit(1)
