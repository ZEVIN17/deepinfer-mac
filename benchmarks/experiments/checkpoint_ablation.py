"""Real branch-cache experiment; GPU exclusive, preserved baseline default.

Compare current first-chunk snapshots with periodic snapshots.
This is not a complete upstream-server comparison.
"""

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="benchmarks/reports/phase2-checkpoints-qwen3-4b.json")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine()
    base = [
        {
            "role": "system",
            "content": "This is public benchmark context. Preserve exact input and use the requested output format. "
            * 90,
        }
    ]

    def req(i, cache=True):
        return GenerationRequest(
            messages=base + [{"role": "user", "content": f"Compute {i}+{i}. Give only the number."}],
            max_tokens=32,
            use_cache=cache,
        )

    report = {
        "source": source_manifest(),
        "created_unix": time.time(),
        "scope": "same-engine checkpoint coverage, not upstream-server advantage",
        "pairs": [],
    }
    engine.generate(req(2, False))
    for r in range(args.repeats):
        ref = engine.generate(req(3, False))
        pair = {"repeat": r, "reference": dataclasses.asdict(ref), "variants": {}}
        order = [0, 512] if r % 2 == 0 else [512, 0]
        for interval in order:
            engine.clear_cache()
            engine.checkpoint_interval_tokens = interval
            seed = engine.generate(req(2))
            branch = engine.generate(req(3))
            pair["variants"][str(interval)] = {
                "seed": dataclasses.asdict(seed),
                "branch": dataclasses.asdict(branch),
                "equal": branch.token_ids == ref.token_ids,
                "entries": len(engine.cache.entries),
            }
        report["pairs"].append(pair)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["passed"] = all(v["equal"] for p in report["pairs"] for v in p["variants"].values())
    report["summary"] = {
        str(k): {
            "median_branch_ttft_ms": statistics.median(
                p["variants"][str(k)]["branch"]["metrics"]["ttft_ms"] for p in report["pairs"]
            ),
            "median_branch_total_ms": statistics.median(
                p["variants"][str(k)]["branch"]["metrics"]["total_ms"] for p in report["pairs"]
            ),
            "hit_tokens": [
                p["variants"][str(k)]["branch"]["metrics"]["cache_hit_tokens"] for p in report["pairs"]
            ],
            "median_seed_ms": statistics.median(
                p["variants"][str(k)]["seed"]["metrics"]["total_ms"] for p in report["pairs"]
            ),
        }
        for k in (0, 512)
    }
    engine.close()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
