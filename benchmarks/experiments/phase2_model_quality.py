"""Reproducible fixed-corpus model selection; real MLX only.

PYTHONPATH=src .venv/bin/python benchmarks/experiments/phase2_model_quality.py --profile qwen3.5-9b-8bit
Smoke and public subset scores are reported separately. No cross-model latency
ratio here is a framework speedup. GPU must be exclusively owned by this process.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import dataclasses
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.models import get_profile
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


def final_answer(text, thinking=False):
    # Require an actual closing delimiter before scoring a thinking response.
    if (thinking or '<think>' in text) and '</think>' not in text:
        return None
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    for marker in ('<|im_end|>', '<|endoftext|>'):
        text = text.replace(marker, '')
    return text.strip()


def score_smoke(task, output, thinking=False):
    text = final_answer(output, thinking=thinking)
    if text is None:
        return {'passed': False, 'reason': 'thinking response did not finish within token budget'}
    kind = task['kind']
    if kind == 'zh_manual':
        return {'passed': None, 'status': 'manual_review_required', 'rubric': task['rubric']}
    if kind == 'exact':
        return {'passed': text == task['expected'], 'expected': task['expected']}
    if kind == 'json':
        try:
            return {'passed': json.loads(text) == task['expected'], 'expected': task['expected']}
        except (ValueError, TypeError) as exc:
            return {'passed': False, 'error': str(exc)}
    if kind == 'code':
        source = text.removeprefix('```python').removeprefix('```').removesuffix('```').strip()
        try:
            tree = ast.parse(source)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                return {'passed': False, 'reason': 'expected only one function'}
            if any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)) or
                   (isinstance(n, ast.Name) and n.id.startswith('__')) or
                   (isinstance(n, ast.Attribute) and n.attr not in {'add', 'append', 'fromkeys', 'keys', 'values', 'items'})
                   for n in ast.walk(tree)):
                return {'passed': None, 'reason': 'construct outside isolated harness; manual review required'}
            worker = '''import json,sys
j=json.load(sys.stdin)
env={"__builtins__":{k:__builtins__.__dict__[k] for k in ["set","list","len","dict","range","enumerate","min","max","sum","sorted","abs","zip","tuple","str","int","float","bool","isinstance","type","any","all","map","filter","round","reversed","ValueError","TypeError","KeyError","IndexError"]}}
exec(compile(j["source"],"<model-quality>","exec"),env)
f=env[j["function"]]
results=[{"actual":f(*t["args"]),"expected":t["expected"]} for t in j["tests"]]
print(json.dumps({"passed":all(x["actual"]==x["expected"] for x in results),"tests":results}))
'''
            child = subprocess.run([sys.executable, '-I', '-c', worker], text=True, capture_output=True,
                                   timeout=3, check=False, cwd='/tmp',
                                   input=json.dumps({'source': source, 'function': task['function'], 'tests': task['tests']}))
            if child.returncode == 0:
                return json.loads(child.stdout)
            missing = re.search(r"NameError: name '([^']+)' is not defined", child.stderr)
            if missing and hasattr(builtins, missing.group(1)):
                return {'passed': None, 'status': 'harness_error', 'error': child.stderr[-1500:]}
            return {'passed': False, 'error': child.stderr[-1500:]}
        except (SyntaxError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            return {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
    raise ValueError(kind)


def quality_output_path(profile, public_mode, output=None):
    default = f'results/quality-cache/quality-{profile}.json' if public_mode else f'benchmarks/reports/quality-{profile}.json'
    path = Path(output or default)
    if public_mode:
        parts = path.resolve().parts
        if not any(parts[i:i+2] == ('results', 'quality-cache') for i in range(len(parts)-1)):
            raise ValueError('Public raw reports must stay in excluded results/quality-cache; publish only sanitized summary.')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', required=True)
    ap.add_argument('--output')
    ap.add_argument('--settings', nargs='+', default=['greedy_engine', 'greedy_native', 'official_non_thinking', 'official_thinking'])
    ap.add_argument('--include-public', action='store_true')
    ap.add_argument('--public-only', action='store_true')
    ap.add_argument('--thinking-cap', type=int, default=1024)
    ap.add_argument('--sampling-family', choices=['qwen35', 'qwen3'], default='qwen35')
    ap.add_argument('--task-ids', nargs='+', help='Explicit preregistered additional-control subset; never replaces main results')
    ap.add_argument('--output-cap', type=int, help='Explicit uniform additional-control output budget')
    args = ap.parse_args()
    smoke_path = Path('benchmarks/prompts/phase2-quality-smoke.json')
    tasks = [{**t, 'benchmark': 'local_smoke'} for t in json.loads(smoke_path.read_text())['cases']]
    public = None
    if args.public_only:
        tasks = []
    if args.include_public or args.public_only:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        spec = importlib.util.spec_from_file_location('quality_runner', 'benchmarks/quality/runner.py')
        public = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(public)
        tasks += public.load_tasks()
    if args.task_ids:
        unknown = set(args.task_ids) - {task['id'] for task in tasks}
        if unknown:
            raise ValueError(f'Unknown task IDs: {sorted(unknown)}')
        tasks = [task for task in tasks if task['id'] in args.task_ids]
    if args.output_cap is not None:
        if args.output_cap <= 0:
            raise ValueError('Output cap must be positive')
        tasks = [dict(task, max_tokens=args.output_cap) for task in tasks]
    profile = get_profile(args.profile)
    report = {'source_start': source_manifest(), 'profile': args.profile, 'model': profile,
              'created_unix': time.time(), 'status': 'running', 'cases': [], 'sampling_family': args.sampling_family,
              'additional_control': {'task_ids': args.task_ids, 'uniform_output_cap': args.output_cap},
              'corpus': {'smoke_path': str(smoke_path), 'smoke_sha256': hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
                         'public_tasks_sha256': hashlib.sha256(Path('results/quality-cache/tasks.json').read_bytes()).hexdigest() if public else None},
              'frozen_source': json.loads(Path('quality-source-manifest.json').read_text()) if Path('quality-source-manifest.json').exists() else None,
              'execution_backends': {'greedy_engine': 'project', 'greedy_native': 'native MLX', 'official_engine': 'project', 'official_thinking_engine': 'project', 'official_non_thinking': 'native MLX', 'official_thinking': 'native MLX'},
              'scope': 'Fixed small smoke/public subsets; not full leaderboards or statistical model equivalence. Quality and efficiency separate; timings are diagnostic, not fair engine benchmark.'}
    public_mode = args.include_public or args.public_only
    output_path = quality_output_path(args.profile, public_mode, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    engine = InferenceEngine(checkpoint_interval_tokens=512)
    try:
        before = psutil.swap_memory().used
        start = time.perf_counter()
        mx.reset_peak_memory()
        engine._load(args.profile, False)
        report['load'] = {'seconds': time.perf_counter()-start, 'peak_metal_bytes': mx.get_peak_memory(),
                          'active_metal_bytes': mx.get_active_memory(), 'rss_bytes': psutil.Process().memory_info().rss,
                          'system_swap_delta_bytes': psutil.swap_memory().used-before}
        report['template_sha256'] = hashlib.sha256(engine.tokenizer.chat_template.encode()).hexdigest()
        report['cache_types'] = sorted({type(c).__name__ for c in make_prompt_cache(engine.model)})
        save()
        for setting in args.settings:
            for task in tasks:
                thinking = setting in ('official_thinking', 'official_thinking_engine')
                max_tokens = max(task['max_tokens'], args.thinking_cap) if thinking else task['max_tokens']
                request = GenerationRequest(profile=args.profile, messages=task['messages'], max_tokens=max_tokens,
                                            use_cache=False, prefill_step_size=256, enable_thinking=thinking)
                tokens = engine._tokens(request)
                mx.random.seed(42)
                parameters = {'temperature': 0, 'presence_penalty': 0}
                if setting.startswith('official'):
                    temp, top_p, presence = ((.6, .95, 0) if thinking else (.7, .8, 0)) if args.sampling_family == 'qwen3' else ((1.0, .95, 1.5) if thinking else (.7, .8, 1.5))
                    if thinking and (task.get('kind') == 'code' or task.get('scoring', {}).get('type') in ('code', 'python_stdin')):
                        temp, presence = .6, 0
                    parameters = {'temperature': temp, 'top_p': top_p, 'top_k': 20,
                                  'presence_penalty': presence, 'presence_context_size': 0}
                started = time.perf_counter()
                if setting in ('greedy_engine', 'greedy_native', 'official_engine', 'official_thinking_engine'):
                    if setting.startswith('official'):
                        request = dataclasses.replace(request, seed=42,
                                                      **{k: v for k, v in parameters.items() if k != 'presence_context_size'})
                    result = engine.native_generate(request) if setting == 'greedy_native' else engine.generate(request)
                    ids, text, metrics = result.token_ids, result.text, result.metrics
                    finish_reason = result.finish_reason
                else:
                    swap = psutil.swap_memory().used
                    mx.reset_peak_memory()
                    ids, first = [], None
                    gen = generate_step(mx.array(tokens, mx.uint32), engine.model, max_tokens=max_tokens,
                                        prefill_step_size=256, sampler=make_sampler(temp=temp, top_p=top_p, top_k=20),
                                        logits_processors=make_logits_processors(presence_penalty=presence, presence_context_size=0))
                    try:
                        for token, _ in gen:
                            first = first or time.perf_counter()
                            ids.append(int(token))
                            if ids[-1] in engine.tokenizer.eos_token_ids:
                                break
                    finally:
                        gen.close()
                        mx.synchronize()
                    end = time.perf_counter()
                    text = engine.tokenizer.decode(ids)
                    finish_reason = 'stop' if ids and ids[-1] in engine.tokenizer.eos_token_ids else 'length'
                    metrics = {'ttft_ms': (first-started)*1000 if first else None, 'total_ms': (end-started)*1000,
                               'decode_tps': (len(ids)-1)/(end-first) if first and end > first else 0,
                               'peak_memory_bytes': mx.get_peak_memory(), 'active_memory_bytes': mx.get_active_memory(),
                               'rss_bytes': psutil.Process().memory_info().rss, 'swap_delta_bytes': psutil.swap_memory().used-swap}
                quality = score_smoke(task, text, thinking=thinking) if task['benchmark'] == 'local_smoke' else public.score_task(task, final_answer(text, thinking=thinking) or '')
                record = {'task_id': task['id'], 'benchmark': task['benchmark'], 'split': task['split'], 'setting': setting,
                          'messages': task['messages'], 'output': text, 'prompt_token_ids': tokens, 'output_token_ids': ids,
                          'model': profile['target']['repo'], 'revision': profile['target']['revision'],
                          'quantization': profile['quantization'], 'sampling': parameters, 'seed': 42,
                          'enable_thinking': thinking, 'max_tokens': max_tokens, 'finish_reason': finish_reason,
                          'elapsed_seconds': metrics['total_ms']/1000, 'ttft_seconds': metrics['ttft_ms']/1000,
                          'metrics': metrics, 'quality': quality}
                if setting == 'greedy_native' and 'greedy_engine' in args.settings:
                    match = next(x for x in report['cases'] if x['setting'] == 'greedy_engine' and x['task_id'] == task['id'])
                    record['matches_engine_token_ids'] = ids == match['output_token_ids']
                report['cases'].append(record)
                save()
                print(json.dumps({'setting': setting, 'task': task['id'], 'quality': quality, 'tokens': len(ids), 'text_preview': text[:160]}, ensure_ascii=False), flush=True)
        report['status'] = 'measured'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        engine.close()
        report['source_end'] = source_manifest()
        save()


if __name__ == '__main__':
    main()
