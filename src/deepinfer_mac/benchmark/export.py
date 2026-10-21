from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses
        return dataclasses.asdict(value)
    return value


def export_report(report: dict, directory: str | Path) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["sample_id", "phase", "variant", "error", "output_tokens", "total_ms", "ttft_ms",
              "decode_tps", "prompt_tps", "peak_memory_gb", "draft_acceptance"]
    with (destination / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in report.get("samples", []):
            for result in sample["results"]:
                metrics = result.get("metrics", {})
                writer.writerow({"sample_id": sample["sample_id"], "phase": sample["phase"],
                                 "variant": result.get("variant", result["mode"]), "error": result.get("error"),
                                 "output_tokens": len(result.get("token_ids", [])),
                                 **{key: metrics.get(key) for key in fields[5:]}})
    with (destination / "rounds.jsonl").open("w") as stream:
        for sample in report.get("samples", []):
            for result in sample["results"]:
                for round_data in result.get("rounds", []):
                    stream.write(json.dumps({"sample_id": sample["sample_id"], "phase": sample["phase"],
                                             "run_id": result.get("run_id"),
                                             "variant": result.get("variant", result["mode"]),
                                             **round_data}, ensure_ascii=False) + "\n")
    summary = report.get("warm_summary", {})
    lines = ["# DeepInfer-Mac measured benchmark", "", (f"Profile: `{report['configuration']['profile']}`; "
             f"suite: `{report['suite']}`; repeats: {report['repeats']}."), "",
             report["configuration"]["timing"], "", "| Variant | Runs | Median request ms | Failures |",
             "| --- | ---: | ---: | ---: |"]
    for variant, values in summary.get("durations_all_successful_runs", {}).items():
        lines.append(f"| {variant} | {values['n']} | {values['median_total_ms']:.2f} | "
                     f"{summary.get('failures', {}).get(variant, 0)} |")
    lines += ["", (f"Full token equality: {summary.get('equal_token_samples', 0)} / "
              f"{summary.get('paired_samples', 0)} measured comparison groups."), "",
              "| Reference / adaptive request time (work may differ) | Pairs | Median ratio | Mean ratio 95% bootstrap interval |",
              "| --- | ---: | ---: | --- |"]
    for variant, values in summary.get("all_paired_time_ratios_work_may_differ", {}).items():
        interval = values["bootstrap_mean_95_percent_interval"]
        formatted = "insufficient repeats" if interval is None else f"{interval[0]:.3f}–{interval[1]:.3f}"
        lines.append(f"| {variant} | {values['n']} | {values['median']:.3f} | {formatted} |")
    lines += ["", ("Ratios above 1 indicate lower adaptive elapsed time, but different token sequences mean different work. "
              "Every failure and mismatch remains in summary.json and results.csv. No equivalent speedup claim is supported "
              "unless all runs succeed and all full token sequences match. Small corpus results do not establish "
              "general speed advantages or p95 latency. RSS and Metal allocation are separate metrics."), "",
              ("See summary.json for full configuration, hardware, versions, model metadata, every token sequence, "
              "failures and fallbacks; rounds.jsonl contains policy observations.")]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    return destination
