"""Summarize real raw blocks with paired session-cluster bootstrap intervals."""

import argparse
import json
import random
import statistics
from pathlib import Path

from service_inventory import audit


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(raw, candidate, draws=10000, seed=20261021):
    inventory = audit(raw)
    by_engine = {}
    for block in raw["blocks"]:
        records = by_engine.setdefault(block["engine"], {})
        for record in block["records"]:
            key = record["session_id"], record["case"]["kind"], record["repetition"]
            records.setdefault(key, record)
    result = {
        "scope": "Task correctness matched; SSE text TTFT and total time, not decode-token speedup",
        "candidate": candidate,
        "protocol_inventory": inventory,
        "resource_selection": raw.get("configuration", {}).get("resource_selection"),
        "coverage_note": "Deferred groups remain unmeasured, not zero-scored; six clusters pooled across lengths do not establish six-cluster evidence per length.",
        "engines": {},
        "paired": [],
        "failed_blocks": [
            {"engine": b["engine"], "repetition": b["repetition"], "error": b["error"]}
            for b in raw["blocks"]
            if b["error"]
        ],
    }
    for name, records in by_engine.items():
        result["engines"][name] = {
            "requests": len(records),
            "quality_pass": sum(r["quality_pass"] for r in records.values()),
            "protocol_failures": sum(r["error"] is not None for r in records.values()),
        }
    own = by_engine.get(candidate, {})
    randomizer = random.Random(seed)
    for competitor, control in by_engine.items():
        if competitor == candidate:
            continue
        kinds = sorted({key[1] for key in own} | {key[1] for key in control})
        for kind in kinds:
            all_keys = {key for key in own if key[1] == kind} | {key for key in control if key[1] == kind}
            good_keys = [
                key
                for key in sorted(all_keys)
                if key in own and key in control and own[key]["quality_pass"] and control[key]["quality_pass"]
            ]
            for metric in ("total_ms", "ttft_text_ms"):
                clusters = {}
                for key in good_keys:
                    a, b = own[key][metric], control[key][metric]
                    if a is not None and b is not None and b > 0:
                        clusters.setdefault(key[0], []).append((a, b))
                cluster_means = [
                    (statistics.mean(p[0] for p in pairs), statistics.mean(p[1] for p in pairs))
                    for pairs in clusters.values()
                ]
                row = {
                    "competitor": competitor,
                    "kind": kind,
                    "metric": metric,
                    "attempted_pairs": len(all_keys),
                    "quality_passing_pairs": len(good_keys),
                    "session_clusters": len(clusters),
                    "quality_passing_cluster_means_ms": cluster_means,
                    "latency_reduction": None,
                    "ci95": None,
                    "engineering_target_met": False,
                }
                if cluster_means:
                    row["latency_reduction"] = 1 - sum(p[0] for p in cluster_means) / sum(
                        p[1] for p in cluster_means
                    )
                if len(cluster_means) >= 2:
                    distribution = []
                    for _ in range(draws):
                        samples = randomizer.choices(cluster_means, k=len(cluster_means))
                        distribution.append(1 - sum(p[0] for p in samples) / sum(p[1] for p in samples))
                    row["ci95"] = [percentile(distribution, 0.025), percentile(distribution, 0.975)]
                    row["engineering_target_met"] = (
                        inventory["valid"]
                        and len(clusters) >= 6
                        and len(good_keys) == len(all_keys)
                        and row["latency_reduction"] >= 0.15
                        and row["ci95"][0] > 0
                    )
                result["paired"].append(row)
    result["statistics"] = {
        "bootstrap_unit": "session",
        "draws": draws,
        "seed": seed,
        "note": "Intervals are conditional on quality-passing paired tasks; failures remain above. "
        "Target flag also requires complete paired quality coverage, six session clusters, and complete configured order-position rotation. "
        "This flag does not establish DS attribution or final project completion.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--candidate", default="deepinfer-native-cache")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(summarize(json.loads(args.raw.read_text()), args.candidate), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
