"""Explicit public cold/hot/branch/eviction and multi-round memory experiment."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="qwen3-4b")
    parser.add_argument("--output", default="benchmarks/reports/cache-qwen3-4b.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine()
    system = (
        "You are a concise assistant. This public context tests exact prefix caching and contains no private inputs. "
        * 90
    )
    base = [{"role": "system", "content": system}]

    def request(number=2):
        return GenerationRequest(
            profile=args.profile,
            mode="baseline",
            messages=base
            + [{"role": "user", "content": f"What is {number}+{number}? Answer with one number."}],
            max_tokens=16,
        )

    # Warm model and kernels separately; all subsequent request times retained.
    warmup = engine.generate(request())
    report = {
        "profile": args.profile,
        "repeats": 3,
        "warmup": dataclasses.asdict(warmup),
        "pairs": [],
        "multi_round": [],
        "source": source_manifest(),
    }
    for repeat in range(3):
        engine.clear_cache()
        cold = engine.generate(request())
        hot = engine.generate(request())
        branch = engine.generate(request(3))
        branch_reference = engine.generate(dataclasses.replace(request(3), use_cache=False))
        report["pairs"].append(
            {
                "repeat": repeat,
                "cold": dataclasses.asdict(cold),
                "hot": dataclasses.asdict(hot),
                "branch": dataclasses.asdict(branch),
                "branch_reference": dataclasses.asdict(branch_reference),
                "hot_equal": hot.token_ids == cold.token_ids and hot.text == cold.text,
                "branch_equal": branch.token_ids == branch_reference.token_ids
                and branch.text == branch_reference.text,
            }
        )
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Distinct branches share a long prefix. Record each resident/cache/RSS/Metal
    # sample, independently named, plus swap before and after every request.
    for round_index in range(8):
        result = engine.generate(request(round_index + 2))
        reference = engine.generate(dataclasses.replace(request(round_index + 2), use_cache=False))
        report["multi_round"].append(
            {
                "round": round_index,
                "result": dataclasses.asdict(result),
                "equal": result.token_ids == reference.token_ids and result.text == reference.text,
            }
        )
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Force byte-budget rejection of snapshots without changing the user's prompt.
    engine.clear_cache()
    engine.cache.budget_bytes = 1024
    evicted = engine.generate(request())
    replay = engine.generate(request())
    report["eviction"] = {
        "result": dataclasses.asdict(replay),
        "equal": replay.token_ids == evicted.token_ids and replay.text == evicted.text,
        "resident_bytes": engine.cache.nbytes,
    }
    report["passed"] = (
        all(p["hot_equal"] and p["branch_equal"] for p in report["pairs"])
        and all(p["equal"] for p in report["multi_round"])
        and report["eviction"]["equal"]
    )
    engine.close()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(output)}))


if __name__ == "__main__":
    main()
