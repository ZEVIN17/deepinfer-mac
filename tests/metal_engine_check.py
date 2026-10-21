"""Explicit real checkpoint correctness/cache evidence; not collected by pytest."""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
from pathlib import Path

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.state import GenerationRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="qwen3-4b")
    parser.add_argument("--output", default="benchmarks/reports/metal-engine-check.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine()
    report = {"profile": args.profile, "correctness": [], "cache": [], "cancel_recovered": False}
    prompts = [
        "Answer with only the number: 1 + 1 =",
        "用一句话解释统一内存。",
        "Write a Python function that returns the square of an integer.",
    ]
    for prompt in prompts:
        request = GenerationRequest(prompt=prompt, profile=args.profile, max_tokens=48, use_cache=False)
        results = []
        for mode, k in [
            ("native_mlx", 3),
            ("baseline", 3),
            ("fixed_speculative", 1),
            ("fixed_speculative", 3),
            ("adaptive", 3),
        ]:
            events = []
            result = engine.generate(
                dataclasses.replace(request, mode=mode, num_draft_tokens=k), emit=events.append
            )
            record = dataclasses.asdict(result)
            record["stream_text_equal"] = "".join(e.get("text", "") for e in events) == result.text
            results.append(record)
        equal = all(
            r["token_ids"] == results[0]["token_ids"]
            and r["text"] == results[0]["text"]
            and r["stream_text_equal"]
            for r in results
        )
        report["correctness"].append({"prompt": prompt, "equal": equal, "results": results})
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    engine.clear_cache()
    system = "You are a concise assistant. The following is public repeated context for a cache test. " * 90
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "What is 2+2? Answer with one number."},
    ]
    request = GenerationRequest(messages=messages, profile=args.profile, mode="baseline", max_tokens=16)
    for label, candidate in [
        ("cold", request),
        ("hot", request),
        (
            "branch",
            dataclasses.replace(
                request,
                messages=[messages[0], {"role": "user", "content": "What is 3+3? Answer with one number."}],
            ),
        ),
    ]:
        result = engine.generate(candidate)
        reference = engine.generate(dataclasses.replace(candidate, use_cache=False))
        report["cache"].append(
            {
                "label": label,
                "equal": result.token_ids == reference.token_ids,
                "result": dataclasses.asdict(result),
                "reference": dataclasses.asdict(reference),
            }
        )
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    cancel = threading.Event()
    try:
        engine.generate(
            request, emit=lambda event: cancel.set() if event.get("type") == "token" else None, cancel=cancel
        )
    except InterruptedError:
        after = engine.generate(request)
        reference = engine.generate(dataclasses.replace(request, use_cache=False))
        report["cancel_recovered"] = after.token_ids == reference.token_ids
    engine.close()
    report["passed"] = (
        all(r["equal"] for r in report["correctness"] + report["cache"]) and report["cancel_recovered"]
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "output": str(output)}))


if __name__ == "__main__":
    main()
