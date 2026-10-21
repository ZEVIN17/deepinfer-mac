"""Direct real 27B probe: no available-memory preflight; retain observed metrics."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="benchmarks/reports/qwen38-27b-2bit-demo.json")
    args = ap.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_unix": time.time(),
        "source": source_manifest(),
        "profile": "qwen3.8-27b-2bit",
        "status": "running",
        "cases": [],
        "memory_gate": "estimates advisory, actual load explicitly authorized",
        "quality_scope": "small smoke corpus, not a quantization quality benchmark",
    }

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    engine = InferenceEngine(checkpoint_interval_tokens=512)

    def run(label, request):
        events = []
        r = engine.generate(request, emit=events.append)
        data = dataclasses.asdict(r)
        data["stream_matches"] = "".join(e.get("text", "") for e in events) == r.text
        report["cases"].append({"label": label, **data})
        save()
        print(
            json.dumps({"case": label, "text": r.text, "metrics": r.metrics}, ensure_ascii=False), flush=True
        )
        return r

    def req(prompt, **kw):
        return GenerationRequest(
            profile="qwen3.8-27b-2bit", prompt=prompt, max_tokens=128, prefill_step_size=256, **kw
        )

    try:
        save()
        run("zh", req("请用三句话解释统一内存为什么适合本地大语言模型推理。", use_cache=False))
        run(
            "code",
            req(
                "Write a Python function dedupe(items) that removes duplicates while preserving order. Return only the function.",
                use_cache=False,
            ),
        )
        common = "公开测试资料：本项目使用本地模型处理文本，缓存用于减少重复输入的计算。\n" * 90
        cache_request = req(common + "\n请只回答数字：2+2等于多少？")
        engine.clear_cache()
        cold = run("cache_seed", cache_request)
        hot = run("cache_hit", cache_request)
        report["cache_equal"] = cold.token_ids == hot.token_ids
        branch = dataclasses.replace(cache_request, prompt=common + "\n请只回答数字：3+3等于多少？")
        cached = run("branch_cached", branch)
        uncached = run("branch_uncached", dataclasses.replace(branch, use_cache=False))
        report["branch_equal"] = cached.token_ids == uncached.token_ids
        conversation = GenerationRequest(
            profile="qwen3.8-27b-2bit",
            messages=[
                {"role": "user", "content": "请记住测试代号是蓝鲸。只回答收到。"},
                {"role": "assistant", "content": "收到。"},
                {"role": "user", "content": "测试代号是什么？只回答代号。"},
            ],
            max_tokens=64,
            prefill_step_size=256,
        )
        run("conversation", conversation)
        report["status"] = "measured"
        report["engineering_passed"] = (
            report["cache_equal"]
            and report["branch_equal"]
            and all(x["stream_matches"] for x in report["cases"])
        )
    except BaseException as e:
        report["status"] = "failed"
        report["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        engine.close()
        save()
    print(json.dumps({"status": report["status"], "engineering_passed": report.get("engineering_passed", False), "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
