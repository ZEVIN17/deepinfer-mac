"""Offline cost analysis of an actual probe; no model/GPU work or speed claim.

This is off-policy replay: observations retain the historical executed budget.
Alternative selections are hypothetical; no counterfactual runtime is invented.
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from deepinfer_mac.ds.recovery_cost import LaggedMeanRecoveryPolicy
from deepinfer_mac.provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("benchmarks/reports/phase2-hybrid-ds-qwen35-9b8-probe.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/reports/phase2-hybrid-recovery-cost-analysis.json"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    report = {"input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "source": source_manifest(),
              "scope": "CPU-only descriptive analysis and off-policy historical feedback-age replay",
              "predicted_cost_is_measured_performance": False, "performance_gain_claim": False,
              "counterfactual_runtime": None, "observed_groups": [], "replay": []}
    ds_rounds = None
    for sample in data["samples"]:
        rounds = sample["result"]["rounds"]
        if sample["variant"] == "ds":
            ds_rounds = rounds
        for budget in sorted({r["budget"] for r in rounds}):
            group = [r for r in rounds if r["budget"] == budget]
            mean_cost = statistics.mean(r["cycle_ms"] for r in group)
            report["observed_groups"].append({
                "variant": sample["variant"], "budget": budget, "count": len(group),
                "accepted_counts": dict(Counter(r["accepted"] for r in group)),
                "mean_full_cycle_ms": mean_cost,
                "median_full_cycle_ms": statistics.median(r["cycle_ms"] for r in group),
                "mean_recovery_ms": statistics.mean(r["recovery_ms"] for r in group),
                "median_recovery_ms": statistics.median(r["recovery_ms"] for r in group),
                "mean_produced": statistics.mean(r["produced"] for r in group),
                "descriptive_yield_over_cost": statistics.mean(r["produced"] for r in group) / mean_cost,
                "comparison_limit": "single prompt/order and differing histories; not controlled speed evidence",
            })
    if ds_rounds is None:
        raise ValueError("No actual DS round trace found")
    policy = LaggedMeanRecoveryPolicy()
    report["original_first_startup_curve"] = next(
        r["scheduler_replay"]["curve"] for r in ds_rounds if "scheduler_replay" in r)
    for index, record in enumerate(ds_rounds):
        proposed = policy.choose(1000)
        report["replay"].append({
            "cycle": index + 1, "historically_executed_budget": record["budget"],
            "hypothetical_proposed_budget": proposed,
            "eligible_known_graphs": [k + 1 for k in policy.candidates
                                      if len(policy.available[k]) >= policy.min_samples],
            "unknown_graphs": policy.summary()["unknown_graphs"],
            "core_estimated_decision": policy.last_replay,
            "maximum_cost_feedback_cycle": max((s["cycle"] for samples in policy.available.values()
                                                for s in samples), default=None),
        })
        policy.observe_cycle(
            record["budget"], record["accepted"], record["draft_ms"], record["verify_ms"],
            record["produced"], cycle_ms=record["cycle_ms"],
            stage_ms={"snapshot_ms": record["snapshot_ms"],
                      "combined_draft_verify_ms": record["verify_ms"], "recovery_ms": record["recovery_ms"]},
            catchup_tokens=record.get("catchup_tokens", 0),
        )
    report["summary"] = {
        "replayed_actual_cycles": len(ds_rounds),
        "complete_cost_curve_available": not policy.summary()["unknown_graphs"],
        "unknown_graphs": policy.summary()["unknown_graphs"],
        "estimated_core_decisions": sum(r["core_estimated_decision"] is not None for r in report["replay"]),
        "conclusion": "Historical coverage is insufficient for the opt-in five-sample curve; "
                      "further real calibration is required, no speed prediction is supported.",
        "hypothetical_selections_are_executed_work": False,
        "performance_gain_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
