"""Short development-only proof that the real hybrid lagged curve matures.

This script never runs automatically. It uses at most three explicitly declared
development requests, preserves failures, and performs no held-out tuning or
performance claim. Default complete-cycle requirements remain min=5/window=9.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from ds_budget_ablation import DEVELOPMENT, freeze_record, frozen_execution_valid, positive_int, public_error
from verify_lagged_trace import verify

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest

# Only the already-declared development families; never inspect held-out tasks.
DEVELOPMENT_PROBES = [
    DEVELOPMENT[1][1] + " Explain time and space complexity, provide three asserts, and discuss unhashable items.",
    DEVELOPMENT[0][1] + " 再详细说明数据拷贝、带宽、模型驻留和系统内存竞争。",
    DEVELOPMENT[2][1],
]


def finalize_acceptance(report):
    """Raw functionality is useful evidence; acceptance additionally requires a stable genuine freeze."""
    summary = report["summary"]
    summary["functional_results_scope"] = "raw execution only; provenance checked separately after completion"
    summary["frozen_execution_valid"] = frozen_execution_valid(report)
    summary["accepted_frozen_probe_pass"] = bool(
        summary.get("correctness_and_maturity_pass") is True and frozen_execution_valid(report)
        and not report.get("fatal_error"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen3.5-9b-8bit")
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--max-requests", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--freeze-archive", type=Path, help="Explicit archive for legacy unbound manifests")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/reports/phase2-hybrid-lagged-dev-probe.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output path; do not replace earlier probe evidence")
    report = {"source": source_manifest(), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "freeze": freeze_record(args.freeze_manifest, required_harness="benchmarks/experiments/hybrid_lagged_probe.py",
                                      executed_harness_path=Path(__file__), archive_path=args.freeze_archive), "samples": [],
              "protocol": {"scope": "development-only maturity/correctness control; no speed claim",
                           "profile": args.profile, "max_tokens": args.max_tokens,
                           "max_requests": args.max_requests, "candidates": [0, 1, 2, 3, 4],
                           "min_samples": 5, "window": 9, "generation": "greedy, thinking=false, cache off",
                           "development_prompts": DEVELOPMENT_PROBES,
                           "stopping": "stop after a completed request has a mature real core decision, regardless of equality",
                           "policy_state": "persistent per existing engine context bucket; streams replayed independently"}}
    engine = InferenceEngine(cache_budget_bytes=0, adaptive_scheduler="ds", round_diagnostics=True,
                             experimental_hybrid_speculation=True, hybrid_cost_adapter="lagged_mean")

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def generate(request, variant, number):
        events = []
        try:
            result = engine.generate(request, emit=events.append)
            return {"variant": variant, "request_index": number, "result": dataclasses.asdict(result),
                    "policy_stream": f"{request.profile}:nonthinking:{result.prompt_tokens // 256}",
                    "events": events}
        except Exception as exc:  # noqa: BLE001 -- keep actual partial emitted output, never fake zero timing
            return {"variant": variant, "request_index": number, "error": public_error(exc),
                    "events": events, "trace_scope": "no complete result; emitted events retained, stop this probe"}

    try:
        engine._load(args.profile, True)
        from deepinfer_mac.models import get_profile

        profile = get_profile(args.profile)
        report["model_identity"] = {"target": profile["target"], "draft": profile["draft"],
                                    "target_weight_bytes": profile["weight_bytes"],
                                    "additional_draft_weight_bytes": profile["draft_weight_bytes"]}
        save()
        for number, prompt in enumerate(DEVELOPMENT_PROBES[:args.max_requests]):
            request = GenerationRequest(profile=args.profile, prompt=prompt, max_tokens=args.max_tokens,
                                        use_cache=False)
            baseline = generate(request, "baseline", number)
            report["samples"].append(baseline)
            save()
            if "error" in baseline:
                break
            candidate = generate(dataclasses.replace(request, mode="adaptive"), "ds_lagged_mean", number)
            report["samples"].append(candidate)
            if "result" in candidate:
                expected, actual = baseline["result"]["token_ids"], candidate["result"]["token_ids"]
                candidate["exact_to_baseline"] = actual == expected
                candidate["first_divergence"] = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                                                      min(len(actual), len(expected)) if len(actual) != len(expected) else None)
            save()
            if "error" in candidate:
                break
            if any("scheduler_replay" in r for r in candidate["result"]["rounds"]):
                break
        try:
            report["independent_oracle"] = verify(report)
        except Exception as exc:  # noqa: BLE001 -- failed verification remains a failed result
            report["independent_oracle"] = {"matches": False, "error": public_error(exc)}
        candidates = [s for s in report["samples"] if s["variant"] == "ds_lagged_mean"]
        failures = [s for s in report["samples"] if "error" in s]
        decisions = sum("scheduler_replay" in r for s in candidates if "result" in s for r in s["result"]["rounds"])
        report["summary"] = {"completed_candidate_requests": sum("result" in s for s in candidates),
                             "mature_core_decisions": decisions,
                             "all_outputs_exact_to_baseline": bool(candidates) and all(s.get("exact_to_baseline") is True for s in candidates),
                             "failures": len(failures),
                             "correctness_and_maturity_pass": (
                                 decisions > 0 and report["independent_oracle"].get("matches") is True
                                 and bool(candidates) and all(s.get("exact_to_baseline") is True for s in candidates)
                                 and not failures),
                             "maturity_and_oracle_pass": decisions > 0 and report["independent_oracle"].get("matches") is True and not failures,
                             "equivalent_speedup_claim_supported": False, "competitive_advantage_claim_supported": False}
        report["harness_sha256_end"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["source_end"] = source_manifest()
        report["source_unchanged"] = report["source_end"]["source_tree_sha256"] == report["source"]["source_tree_sha256"]
        report["freeze_end"] = freeze_record(args.freeze_manifest, required_harness="benchmarks/experiments/hybrid_lagged_probe.py",
                                      executed_harness_path=Path(__file__), archive_path=args.freeze_archive)
        finalize_acceptance(report)
        save()
        print(json.dumps(report["summary"], indent=2))
    except Exception as exc:
        report["fatal_error"] = public_error(exc)
        report["summary"] = {**report.get("summary", {}), "accepted_frozen_probe_pass": False,
                             "frozen_execution_valid": False, "competitive_advantage_claim_supported": False}
        save()
        raise
    finally:
        engine.close()


if __name__ == "__main__":
    main()
