"""Forced budget isolates execution overhead from adaptive policy selection."""

import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import speculative_generate_step
from mlx_lm.models.cache import make_prompt_cache

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.speculative import adaptive_generate_step
from deepinfer_mac.state import GenerationRequest


class Fixed:
    def choose(self, remaining, memory_pressure=False):
        return min(2, remaining)

    def observe(self, *args):
        pass


def main():
    engine = InferenceEngine()
    engine._load("qwen3-4b", True)
    prompts = [
        "用一句话解释统一内存。",
        "Write a Python function dedupe(items) that preserves order.",
        "List three benefits of prefix caching.",
    ]
    report = {
        "source": source_manifest(),
        "scope": "forced k=2, 64-token cap, loaded model; isolates loop changes, not adaptive policy gain",
        "samples": [],
    }
    path = Path("benchmarks/reports/phase2-draft-sync-qwen3-4b.json")
    for repeat in range(-1, 3):
        for pi, prompt in enumerate(prompts):
            ids = engine._tokens(GenerationRequest(prompt=prompt))
            variants = ["old", "deferred", "upstream"]
            shift = (repeat + pi) % 3
            variants = variants[shift:] + variants[:shift]
            group = {"repeat": repeat, "prompt": prompt, "variants": {}}
            for variant in variants:
                caches = make_prompt_cache(engine.model)
                draft = make_prompt_cache(engine.draft_model)
                x = mx.array(ids[:-1], mx.uint32)[None]
                engine.model(x, cache=caches)
                engine.draft_model(x, cache=draft)
                mx.eval([c.state for c in caches + draft])
                mx.synchronize()
                start = time.perf_counter()
                rounds = []
                out = []
                if variant == "upstream":
                    gen = speculative_generate_step(
                        mx.array(ids[-1:], mx.uint32),
                        engine.model,
                        engine.draft_model,
                        num_draft_tokens=2,
                        max_tokens=64,
                        prompt_cache=caches + draft,
                    )
                else:
                    gen = adaptive_generate_step(
                        ids[-1:],
                        engine.model,
                        engine.draft_model,
                        caches,
                        draft,
                        Fixed(),
                        64,
                        rounds,
                        defer_draft_sync=variant == "deferred",
                    )
                try:
                    for item in gen:
                        token = int(item[0])
                        out.append(token)
                        if token in engine.tokenizer.eos_token_ids:
                            break
                finally:
                    gen.close()
                    mx.synchronize()
                group["variants"][variant] = {
                    "ms": (time.perf_counter() - start) * 1000,
                    "ids": out,
                    "rounds": rounds,
                }
            group["all_equal"] = all(
                v["ids"] == group["variants"]["upstream"]["ids"] for v in group["variants"].values()
            )
            report["samples"].append(group)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    measured = [s for s in report["samples"] if s["repeat"] >= 0]
    report["summary"] = {
        "all_equal": all(s["all_equal"] for s in measured),
        "groups": len(measured),
        "median_old_over_deferred": statistics.median(
            s["variants"]["old"]["ms"] / s["variants"]["deferred"]["ms"] for s in measured
        ),
        "median_upstream_over_deferred": statistics.median(
            s["variants"]["upstream"]["ms"] / s["variants"]["deferred"]["ms"] for s in measured
        ),
        "median_ms": {v: statistics.median(s["variants"][v]["ms"] for s in measured) for v in variants},
    }
    engine.close()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
