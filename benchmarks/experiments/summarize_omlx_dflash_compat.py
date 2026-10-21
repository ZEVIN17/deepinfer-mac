"""Publish a compact, path-free derivative of immutable private DFlash evidence."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(report):
    results = []
    for run in report["results"]:
        events = run["audit_events"]
        attached = next((event for event in events if event["event"] == "drafter_attached"), None)
        stats = [event["stats"] for event in events if event["event"] == "upstream_verify_stats"]
        resources = [event["resources"] for event in events if "resources" in event]
        results.append({"variant": run["engine"]["id"], "strict_passes": sum(record["quality_pass"] for record in run["records"]),
                        "strict_attempts": len(run["records"]), "all_strict_passed": run["all_strict_cases_passed"],
                        "actual_dflash_execution_proven": run["dflash_execution_proven"],
                        "official_loaded_log": run["official_drafter_loaded_log_observed"],
                        "soft_fallback": run["draft_load_soft_fallback_observed"],
                        "server_exit_code": run["server_exit_code"],
                        "cases": [{"id": record["case"]["id"], "quality_pass": record["quality_pass"], "error": record["error"],
                                   "protocol_complete": record["protocol_complete"], "text": record["text"],
                                   "system_after": record["system_after"],
                                   "actual_prompt": [{key: event[key] for key in ("prompt_tokens", "token_ids_sha256_le_u32")}
                                                     for event in record["audit_events"] if event["event"] == "request_prompt"]}
                                  for record in run["records"]],
                        "paired_baseline": run.get("paired_baseline_outputs"),
                        "draft": None if attached is None else {key: attached[key] for key in
                            ("attached_identity", "decode_enabled", "engine_property_identity", "block_size", "adaptive", "window", "owned_parameter_storage_bytes", "owned_module_classes")},
                        "draft_forward_invocations": sum(event["event"] == "draft_batched_completed" for event in events),
                        "completed_verify_cycles": sum(item["cycles"] for item in stats),
                        "accepted_draft_positions": sum(item["accepts"] for item in stats),
                        "examined_draft_positions": sum(sum(item["depth_drafted"]) for item in stats),
                        "system_swap_delta_bytes": run["system_swap_delta_bytes"],
                        "maximum_observed_metal_peak_bytes": max((r["metal_peak_bytes"] for r in resources), default=None),
                        "metal_peak_coverage": "post-cleanup snapshots for completed requests" if any(event["event"] == "requests_cleanup_completed" for event in events) else "load attachment and each request's completed verify stats" if attached else "startup/pre-request only; lower bound, no complete final-request peak",
                        "environment": next(event["environment"] for event in events if event["event"] == "audit_started"),
                        "upstream_hook_sources_sha256": next(event["hook_sources_sha256"] for event in events if event["event"] == "audit_started")})
    return {"scope": "Instrumented actual dev compatibility; no speedup, throughput or memory-peak ratio claim",
            "target_repo": report["target_repo"], "target_revision": report["target_revision"],
            "draft_repo": report["draft_repo"], "draft_revision": report["draft_revision"],
            "manifest_sha256": report["manifest_sha256"], "actual_harness_sha256": report["harness_sha256"],
            "results": results, "limitations": [
                "Authored greedy nonthinking dev attempts only; successes, failures and any paired baseline text/prompt checks are listed per run. Generated output IDs were not captured.",
                "Diagnostic I/O and wrappers may influence adaptive budget choices; do not use observed choices or times as formal performance tuning.",
                "Accepted/examined draft positions excludes generated proposals that were never examined; it is not an overall generation speedup.",
                "System-wide swap changed during the DFlash block; the change is not exclusively attributable to that process.",
                "Resource coverage is labeled per variant; the original baseline observer lacks complete final-request Metal-peak coverage.",
                "Server processes deliberately received SIGTERM after attempts; official private lifecycle logs are retained. No server remained on8084.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", default="benchmarks/reports/phase2-omlx-dflash-compat-source-v1.tar.gz")
    parser.add_argument("--source-metadata", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.raw.read_text()))
    result["immutable_private_report_sha256"] = hashlib.sha256(args.raw.read_bytes()).hexdigest()
    result["frozen_actual_harness_archive"] = args.source_archive
    if Path(args.source_archive).is_file():
        result["frozen_actual_harness_archive_sha256"] = hashlib.sha256(Path(args.source_archive).read_bytes()).hexdigest()
    if args.source_metadata:
        from service_source_freeze import source_files

        metadata = json.loads(args.source_metadata.read_text())
        result["frozen_project_source_sha256"] = metadata["sha256"]
        result["post_run_frozen_source_bytes_verified"] = source_files(Path(metadata["root"])) == metadata["files"]
        if not result["post_run_frozen_source_bytes_verified"]:
            raise ValueError("Actual frozen resource-probe source drifted")
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
