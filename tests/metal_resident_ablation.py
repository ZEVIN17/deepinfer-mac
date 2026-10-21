"""Isolated resident-mode blocks: no mode can unload another mode's draft."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest

PROMPTS = [
    ("zh", "用一句话解释统一内存。"),
    ("code", "Write a Python function that returns the square of an integer."),
    ("short", "Answer with only the number: 1 + 1 ="),
]
VARIANTS = [
    ("native_mlx", 0),
    ("baseline", 0),
    ("fixed_speculative", 1),
    ("fixed_speculative", 2),
    ("fixed_speculative", 4),
    ("adaptive", 0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="qwen3-4b")
    parser.add_argument("--output", default="benchmarks/reports/resident-ablation-qwen3-4b.json")
    parser.add_argument("--repeat", type=int, choices=range(3))
    parser.add_argument("--variant")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "profile": args.profile,
        "source": source_manifest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "design": "3 repetitions, rotating mode blocks, fresh engine per mode block, each prompt warmed before measurement; no prefix reuse",
        "blocks": [],
    }
    for repeat in range(3) if args.repeat is None else [args.repeat]:
        order = VARIANTS[repeat:] + VARIANTS[:repeat]
        for mode, length in order:
            variant = f"{mode}:{length}" if length else mode
            if args.variant is not None and args.variant != variant:
                continue
            engine = InferenceEngine(cache_budget_bytes=0)
            block = {
                "repeat": repeat,
                "variant": f"{mode}:{length}" if length else mode,
                "warmup": [],
                "measured": [],
            }
            report["blocks"].append(block)
            try:
                for phase in ("warmup", "measured"):
                    for prompt_id, prompt in PROMPTS:
                        request = GenerationRequest(
                            prompt=prompt,
                            profile=args.profile,
                            mode=mode,
                            max_tokens=64,
                            num_draft_tokens=max(length, 1),
                            use_cache=False,
                        )
                        try:
                            result = engine.generate(request)
                            block[phase].append({"prompt_id": prompt_id, **dataclasses.asdict(result)})
                        except Exception as exc:  # noqa: BLE001 — retain every real benchmark failure
                            block[phase].append(
                                {"prompt_id": prompt_id, "error": f"{type(exc).__name__}: {exc}"}
                            )
                        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            finally:
                engine.close()
            print(repeat, block["variant"], "complete", flush=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
