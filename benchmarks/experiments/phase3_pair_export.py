import hashlib
import json
from pathlib import Path


def digest(x):
    return hashlib.sha256(json.dumps(x, separators=(",", ":")).encode()).hexdigest()


def redact(x):
    if isinstance(x, list):
        return [redact(v) for v in x]
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "outputs":
                out["output_summaries"] = [
                    {"token_count": len(row), "token_ids_sha256": digest(row)} for row in v
                ]
            elif k == "tokens" and isinstance(v, list):
                out["output_summary"] = {"token_count": len(v), "token_ids_sha256": digest(v)}
            else:
                out[k] = redact(v)
        return out
    return x


EXPECTED_POLICIES = frozenset(
    {"cost_bucket", "fixed_full_bucket", "heterogeneous_bucket", "heterogeneous_padded", "fixed_shared_cap"}
)


def coverage(report):
    """Derive complete comparisons from raw outputs, never assumed success counts."""
    tasks = report.get("workload_manifest", {}).get("splits", {}).get("heldout", [])
    budget = report.get("workload_manifest", {}).get("output_tokens")
    native = report.get("native_serial", [])
    cases = report.get("cases", [])
    task_count = len(tasks)
    policies = [case.get("policy") for case in cases]
    complete_cases, matched, total = 0, 0, 0
    native_complete = (
        task_count > 0
        and type(budget) is int
        and budget > 0
        and len(native) == task_count
        and all(len(row.get("tokens", [])) == budget for row in native)
    )
    for case in cases:
        flags, outputs = case.get("matches_native", []), case.get("outputs", [])
        total += len(flags)
        complete = (
            native_complete
            and len(flags) == len(outputs) == task_count
            and case.get("all_output_budgets_complete") is True
            and all(len(row) == budget for row in outputs)
        )
        complete_cases += int(complete)
        for index, flag in enumerate(flags):
            matched += int(
                flag is True
                and index < len(outputs)
                and index < len(native)
                and len(outputs[index]) == budget
                and outputs[index] == native[index].get("tokens")
            )
    missing = sorted(EXPECTED_POLICIES - set(policies))
    unexpected = sorted(str(p) for p in set(policies) - EXPECTED_POLICIES)
    duplicate = len(policies) != len(set(policies))
    complete = (
        native_complete
        and not missing
        and not unexpected
        and not duplicate
        and complete_cases == len(EXPECTED_POLICIES)
        and total == task_count * len(EXPECTED_POLICIES)
    )
    return {
        "matched": matched,
        "total": total,
        "expected_comparisons": task_count * len(EXPECTED_POLICIES),
        "expected_cases": len(EXPECTED_POLICIES),
        "observed_cases": len(cases),
        "complete_cases": complete_cases,
        "native_complete": native_complete,
        "missing_policies": missing,
        "unexpected_policies": unexpected,
        "duplicate_policy": duplicate,
        "coverage_complete": complete,
        "strict_pass": complete
        and matched == total
        and report.get("accepted") is True
        and not report.get("failures"),
    }


def self_check():
    """CPU counterexamples mutate a copy of real evidence; never run inference."""
    import copy

    raw = Path("artifacts/phase3-pair-cost/heldout.json")
    original = json.loads(raw.read_text())
    variants = {"actual": original}
    for name in ("missing_case", "mismatch", "truncated_output", "empty_cases"):
        variants[name] = copy.deepcopy(original)
    variants["missing_case"]["cases"].pop()
    variants["mismatch"]["cases"][0]["matches_native"][0] = False
    variants["truncated_output"]["cases"][0]["outputs"][0].pop()
    variants["empty_cases"]["cases"] = []
    results = {name: coverage(value) for name, value in variants.items()}
    assert results["actual"]["strict_pass"]
    assert (results["missing_case"]["matched"], results["missing_case"]["total"]) == (8, 8)
    assert not results["missing_case"]["coverage_complete"]
    assert (results["mismatch"]["matched"], results["mismatch"]["total"]) == (9, 10)
    assert not results["mismatch"]["strict_pass"]
    assert not results["truncated_output"]["coverage_complete"]
    assert results["truncated_output"]["matched"] == 9
    assert results["empty_cases"]["total"] == 0 and not results["empty_cases"]["strict_pass"]
    return {
        "scope": "CPU exporter counterexamples, not additional model measurements",
        "raw_input_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": True,
        "cases": results,
    }


def main():
    for stage in ["development", "heldout"]:
        raw = Path(f"artifacts/phase3-pair-cost/{stage}.json")
        d = json.loads(raw.read_text())
        r = redact(d)
        r["public_export"] = {
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "transformation": "Replace generated token arrays by count and canonical JSON SHA256; preserve original source hashes, measured timing, costs, shapes, outcomes and acceptance. Raw prompts are not in this report.",
            "accepted_scope": "Bounded experimental calibration/mechanism only; no release or comparative speed acceptance.",
        }
        if stage == "heldout":
            r["public_export"]["derived_coverage"] = coverage(d)
        Path(f"benchmarks/reports/phase3-pair-cost-{stage}.json").write_text(json.dumps(r, indent=2) + "\n")
    d = json.loads(Path("artifacts/phase3-pair-cost/heldout.json").read_text())
    cases = {c["policy"]: c for c in d["cases"]}
    if not EXPECTED_POLICIES.issubset(cases):
        summary = {
            "scope": "Incomplete experimental input; no performance conclusion",
            "strict_heldout_native_matches": coverage(d),
            "performance_claim": False,
            "missing_metric_reason": "Required control case absent",
        }
        Path("docs/evidence/phase3-pair-cost-experiment.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2))
        return
    b = cases["heterogeneous_bucket"]
    p = cases["heterogeneous_padded"]
    a = cases["cost_bucket"]
    f = cases["fixed_full_bucket"]
    native = sum(x["complete_task_ms"] for x in d["native_serial"])
    summary = {
        "scope": "One frozen development process followed by one held-out process, 2 tasks, 32 tokens/task. Descriptive observations only; no significant speed advantage established.",
        "strict_heldout_native_matches": coverage(d),
        "swap_delta_bytes": d["swap_after_bytes"] - d["swap_before_bytes"],
        "metal_peak_bytes": d["metal_peak_bytes"],
        "native_serial_two_tasks_ms": native,
        "cost_bucket_two_tasks_ms": a["complete_task_ms"],
        "fixed_full_bucket_two_tasks_ms": f["complete_task_ms"],
        "cost_bucket_vs_fixed_latency_change_percent": (a["complete_task_ms"] / f["complete_task_ms"] - 1)
        * 100,
        "cost_bucket_vs_native_latency_ratio": a["complete_task_ms"] / native,
        "bucket_padded": {
            "bucket_verify_positions": sum(t["executed_verify_q"] for t in b["cycles"]),
            "padded_verify_positions": sum(t["executed_verify_q"] for t in p["cycles"]),
            "bucket_decode_all_model_input_positions": b["decode_model_input_positions"],
            "padded_decode_all_model_input_positions": p["decode_model_input_positions"],
            "bucket_complete_task_ms": b["complete_task_ms"],
            "padded_complete_task_ms": p["complete_task_ms"],
            "outputs_equal": d["bucket_padded_outputs_equal"],
        },
        "cost_bucket_profile_ms": {
            "prefill_setup": a["prefill_ms"],
            "full_cycle_sum": sum(t["full_cycle_ms"] for t in a["cycles"]),
            "verify_recovery_host_sum": sum(t["verify_recovery_host_ms"] for t in a["cycles"]),
            "scheduler_snapshot_allocation_sum": sum(t["scheduler_ms"] for t in a["cycles"]),
            "draft_catchup_and_remaining_host_residual": sum(
                t["full_cycle_ms"] - t["verify_recovery_host_ms"] - t["scheduler_ms"] for t in a["cycles"]
            ),
        },
        "limits": [
            "Timing partitions are nested wall intervals, not GPU kernel profiling. Residual includes draft/catch-up/statistics, not isolated kernel cost.",
            "Forced-q development costs omit ordinary JointBudget.choose and MLX rank; held-out executes and times both.",
            "No trained confidence head; sparse observed prefixes produce diagnostic Brier only.",
            "This does not supersede earlier failed serial equality reports or establish broader quality equivalence.",
        ],
        "reports": {
            s: {
                "path": f"benchmarks/reports/phase3-pair-cost-{s}.json",
                "sha256": hashlib.sha256(
                    Path(f"benchmarks/reports/phase3-pair-cost-{s}.json").read_bytes()
                ).hexdigest(),
            }
            for s in ["development", "heldout"]
        },
    }
    Path("docs/evidence/phase3-pair-cost-experiment.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        print(json.dumps(self_check(), indent=2))
    else:
        main()
