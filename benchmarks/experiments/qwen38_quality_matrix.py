"""Fixed 27B quality corpus: greedy versus official non-thinking sampling.

PYTHONPATH=src .venv/bin/python benchmarks/experiments/qwen38_quality_matrix.py
This is a smoke corpus, not a claim about general model quality.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import psutil
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from deepinfer_mac.models import download_models
from deepinfer_mac.provenance import source_manifest

CASES = [
    ("zh", [{"role": "user", "content": "请用三句话解释统一内存为什么适合本地大语言模型推理。"}]),
    ("code", [{"role": "user", "content": "Write a Python function dedupe(items) that removes duplicates while preserving order. Return only the function."}]),
    ("json", [{"role": "user", "content": '仅输出有效JSON，不加解释：把名字Alice、年龄30转换为对象，键为name和age，age用数字。'}]),
    ("arithmetic", [{"role": "user", "content": "只输出计算结果，不要解释：17 × 23 = ?"}]),
    ("conversation", [{"role": "user", "content": "请记住测试代号是蓝鲸。只回答收到。"}, {"role": "assistant", "content": "收到。"}, {"role": "user", "content": "测试代号是什么？只回答代号。"}]),
    ("long", [{"role": "user", "content": "以下是公开的测试清单。\n" + "设备采用统一内存，资料均在本机处理；这些是无关说明。\n" * 100 + "\n清单末尾的测试编号是 K742。请只输出这个测试编号。"}]),
]


def check(label, text):
    cleaned = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if label == "zh":
        return {"passed": None, "rubric": "manual: exactly three coherent Chinese sentences; mentions shared memory and reduced transfer/copies; no repetition", "note": "Never auto-grade fluency from keywords."}
    if label == "code":
        candidate = cleaned.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        try:
            tree = ast.parse(candidate)
            # Whitelisted code is evaluated in an isolated, time-bounded child.
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                raise ValueError("Expected only one function")
            forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
            if any(isinstance(n, forbidden) or (isinstance(n, ast.Name) and n.id.startswith("__")) for n in ast.walk(tree)):
                return {"passed": None, "note": "inconclusive: construct outside isolated smoke harness; review manually"}
            safe_attrs = {"add", "append", "fromkeys", "keys", "values", "items"}
            if any(isinstance(n, ast.Attribute) and n.attr not in safe_attrs for n in ast.walk(tree)):
                return {"passed": None, "note": "inconclusive: attribute outside isolated smoke harness; review manually"}
            worker = """import json,sys
source=json.load(sys.stdin)
env={"__builtins__": {"set":set,"list":list,"len":len,"dict":dict,"range":range,"enumerate":enumerate}}
exec(compile(source,"<quality-smoke>","exec"),env)
fn=env["dedupe"]
cases=[([],[]),([3,1,3,2,1],[3,1,2]),(["a","b","a"],["a","b"])]
print(json.dumps({"passed":all(fn(x)==y for x,y in cases)}))
"""
            completed = subprocess.run([sys.executable, "-I", "-c", worker], input=json.dumps(candidate),
                                       text=True, capture_output=True, timeout=3, cwd="/tmp", check=False)
            if completed.returncode:
                return {"passed": False, "error": completed.stderr[-1500:]}
            return {**json.loads(completed.stdout), "rubric": "function-only, empty/integer/string lists preserve order and remove duplicates"}
        except Exception as exc:  # noqa: BLE001 — preserve every model-quality failure
            return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        if label == "json":
            passed = json.loads(cleaned) == {"name": "Alice", "age": 30}
        else:
            passed = cleaned == {"arithmetic": "391", "conversation": "蓝鲸", "long": "K742"}[label]
        return {"passed": passed}
    except Exception as exc:  # noqa: BLE001 — preserve every model-quality failure
        return {"passed": False, "error": str(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path")
    ap.add_argument("--repo", default="keXjos/Qwen3.8-27B-mlx-2Bit")
    ap.add_argument("--revision", default="39d5487ad4442961b64098e15aa11ce05e6f9976")
    ap.add_argument("--output", default="benchmarks/reports/qwen38-quality-matrix-kexjos.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    path = args.path or download_models("qwen3.8-27b-2bit", include_draft=False)["target"]
    model, tokenizer = load(path, trust_remote_code=False)
    mx.eval(model.parameters())
    report = {"source": source_manifest(), "model": {"repo": args.repo, "revision": args.revision}, "official_source": {"url": "https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "local_file": "artifacts/phase2-model-audit/qwen38-official-readme.md", "sha256": hashlib.sha256(Path("artifacts/phase2-model-audit/qwen38-official-readme.md").read_bytes()).hexdigest()}, "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(), "scope": "Six fixed tasks, one seed; smoke checks only, no general quality superiority claim", "cases": [], "created_unix": time.time()}
    output = Path(args.output)
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for setting in ["greedy", "official_non_thinking"]:
        for label, messages in CASES:
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            seed = 42
            mx.random.seed(seed)
            kwargs = {}
            parameters = {"temperature": 0, "presence_penalty": 0}
            if setting != "greedy":
                parameters = {"temperature": .7, "top_p": .8, "top_k": 20, "presence_penalty": 1.5, "presence_context_size": 0}
                kwargs = {"sampler": make_sampler(temp=.7, top_p=.8, top_k=20), "logits_processors": make_logits_processors(presence_penalty=1.5, presence_context_size=0)}
            mx.reset_peak_memory()
            swap_before = psutil.swap_memory().used
            started = time.perf_counter()
            ids = []
            first = None
            generator = generate_step(mx.array(tokens, mx.uint32), model, max_tokens=args.max_tokens, prefill_step_size=256, **kwargs)
            try:
                for token, _ in generator:
                    first = first or time.perf_counter()
                    ids.append(int(token))
                    if ids[-1] in tokenizer.eos_token_ids:
                        break
            finally:
                generator.close()
                mx.synchronize()
            end = time.perf_counter()
            text = tokenizer.decode(ids)
            record = {"label": label, "setting": setting, "messages": messages, "prompt_rendered": tokenizer.decode(tokens), "prompt_token_ids": tokens, "token_ids": ids, "text": text, "seed": seed, "parameters": parameters, "max_tokens": args.max_tokens, "finish_reason": "stop" if ids[-1] in tokenizer.eos_token_ids else "length", "quality": check(label, text), "metrics": {"ttft_ms": (first-started)*1000, "total_ms": (end-started)*1000, "decode_tps": (len(ids)-1)/(end-first), "peak_memory_bytes": mx.get_peak_memory(), "rss_bytes": psutil.Process().memory_info().rss, "system_swap_delta_bytes": psutil.swap_memory().used-swap_before}}
            report["cases"].append(record)
            save()
            print(json.dumps({"setting": setting, "label": label, "text": text, "quality": record["quality"], "total_ms": record["metrics"]["total_ms"]}, ensure_ascii=False), flush=True)
    report["status"] = "measured"
    save()


if __name__ == "__main__":
    main()
