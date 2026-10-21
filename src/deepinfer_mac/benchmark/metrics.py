"""Paired statistics retain failures and mark correctness coverage explicitly."""
from __future__ import annotations

import random
import statistics


def paired_comparison(results: list[dict]) -> dict:
    successful = [r for r in results if not r.get("error")]
    sequences = [r["token_ids"] for r in successful]
    complete = len(successful) == len(results)
    aligned = complete and len(sequences) > 1 and all(s == sequences[0] for s in sequences[1:])
    output = {"token_ids_equal": aligned, "all_runs_succeeded": complete, "speedups": {}}
    by_variant = {r.get("variant", r["mode"]): r for r in successful}
    candidate = by_variant.get("adaptive", {}).get("metrics", {}).get("total_ms")
    if aligned and candidate and candidate > 0:
        output["speedups"] = {variant: r["metrics"]["total_ms"] / candidate
                              for variant, r in by_variant.items() if variant != "adaptive"}
    if sequences and not aligned:
        reference = by_variant.get("native_mlx", successful[0])
        output["mismatched_variants"] = [r.get("variant", r["mode"]) for r in successful
                                         if r["token_ids"] != reference["token_ids"]]
    return output


def _interval(values):
    if len(values) < 2:
        return None
    rng = random.Random(20261021)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(1000))
    return [means[24], means[974]]


def aggregate(samples: list[dict]) -> dict:
    groups: dict[str, list[float]] = {}
    failures: dict[str, int] = {}
    ratios: dict[str, list[float]] = {}
    all_ratios: dict[str, list[float]] = {}
    warm = [sample for sample in samples if sample.get("phase") == "warm"]
    for sample in warm:
        for result in sample["results"]:
            variant = result.get("variant", result["mode"])
            if result.get("error"):
                failures[variant] = failures.get(variant, 0) + 1
                continue
            duration = result.get("metrics", {}).get("total_ms")
            if duration is not None:
                groups.setdefault(variant, []).append(duration)
        by_variant = {r.get("variant", r["mode"]): r for r in sample["results"] if not r.get("error")}
        candidate_ms = by_variant.get("adaptive", {}).get("metrics", {}).get("total_ms")
        if candidate_ms and candidate_ms > 0:
            for variant, result in by_variant.items():
                if variant != "adaptive" and result.get("metrics", {}).get("total_ms") is not None:
                    all_ratios.setdefault(variant, []).append(result["metrics"]["total_ms"] / candidate_ms)
        for variant, speedup in sample["comparison"].get("speedups", {}).items():
            ratios.setdefault(variant, []).append(speedup)
    return {"durations_all_successful_runs": {
                mode: {"n": len(values), "median_total_ms": statistics.median(values),
                       "min_total_ms": min(values), "max_total_ms": max(values)}
                for mode, values in groups.items()},
            "failures": failures, "paired_samples": len(warm),
            "equal_token_samples": sum(s["comparison"]["token_ids_equal"] for s in warm),
            "all_paired_time_ratios_work_may_differ": {
                variant: {"n": len(values), "median": statistics.median(values),
                          "mean": statistics.mean(values), "bootstrap_mean_95_percent_interval": _interval(values)}
                for variant, values in all_ratios.items()},
            "equivalent_speedup_claim_supported": bool(warm) and all(s["comparison"]["token_ids_equal"] for s in warm),
            "adaptive_speedup_on_equal_pairs_diagnostic": {
                variant: {"n": len(values), "median": statistics.median(values),
                          "mean": statistics.mean(values), "bootstrap_mean_95_percent_interval": _interval(values)}
                for variant, values in ratios.items()},
            "interpretation": "Durations include every successful run. All paired time ratios are reported regardless of token equality and may compare different work. Equal-pair ratios are diagnostic only; no equivalent-speedup claim is supported unless the complete corpus succeeds and matches. Bootstrap intervals are descriptive for this fixed corpus, not general significance."}
