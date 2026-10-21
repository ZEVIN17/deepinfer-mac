"""Opt-in real-model token/cache validation; never a competitive speed claim."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from public_service_report import public_report

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metal", action="store_true")
    parser.add_argument("--profile", default="qwen3.5-9b-8bit")
    parser.add_argument("--sampler", choices=("both", "greedy", "official"), default="both")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/reports/phase2-message-checkpoints-qwen35-9b8.json")
    )
    args = parser.parse_args()
    if not args.run_metal:
        parser.error("Explicit --run-metal and exclusive GPU ownership are required")
    manifest_bytes = Path("benchmarks/experiments/cache_sessions.v2.json").read_bytes()
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_sha = "f891ff522edc120731781f5a0e8f3b211774b4dfea696af1690ec8f4c5d81ad5"
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError("Frozen v2 manifest bytes changed; review before real-model validation")
    manifest = json.loads(manifest_bytes)
    cases = manifest["sessions"][0]["cases"][:3]
    if any(c["requested_total_context_tokens"] > 4096 for c in cases):
        raise ValueError("Validation is limited to the initial 4096-token total-context phase")
    settings = {
        "greedy": {"temperature": 0, "seed": 0},
        "official": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 42},
    }
    report = {
        "source": source_manifest(),
        "scope": "one dev session, strict token/cache controls; not performance",
        "profile": args.profile,
        "manifest_sha256": actual_manifest_sha,
        "expected_manifest_sha256": expected_manifest_sha,
        "manifest_hash_verified": True,
        "status": "running",
        "records": [],
        "checks": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(public_report(report), ensure_ascii=False, indent=2) + "\n")

    engine = InferenceEngine(message_checkpoints=False, cache_backend="native")
    try:
        for name, sampling in settings.items():
            if args.sampler != "both" and name != args.sampler:
                continue
            references = {}
            for case_index in (0, 2):
                case = cases[case_index]
                request = GenerationRequest(
                    profile=args.profile,
                    messages=case["messages"],
                    max_tokens=case["max_tokens"],
                    use_cache=False,
                    **sampling,
                )
                engine.message_checkpoints = False
                reference = engine.generate(request)
                references[case_index] = reference.token_ids
                report["records"].append(
                    {
                        "sampling": name,
                        "variant": "off_uncached",
                        "kind": case["kind"],
                        "result": dataclasses.asdict(reference),
                    }
                )
                engine.message_checkpoints = True
                unchanged = engine.generate(request)
                report["records"].append(
                    {
                        "sampling": name,
                        "variant": "on_uncached",
                        "kind": case["kind"],
                        "result": dataclasses.asdict(unchanged),
                    }
                )
                report["checks"].append(
                    {
                        "sampling": name,
                        "scope": "cache_false_path",
                        "kind": case["kind"],
                        "equal": reference.token_ids == unchanged.token_ids,
                    }
                )
                save()
            for enabled in (False, True):
                engine.message_checkpoints = enabled
                engine.clear_cache()
                for case_index, case in enumerate(cases):
                    result = engine.generate(
                        GenerationRequest(
                            profile=args.profile,
                            messages=case["messages"],
                            max_tokens=case["max_tokens"],
                            use_cache=True,
                            **sampling,
                        )
                    )
                    expected = references[2 if case_index == 2 else 0]
                    report["records"].append(
                        {
                            "sampling": name,
                            "variant": "on_cached" if enabled else "off_cached",
                            "kind": case["kind"],
                            "result": dataclasses.asdict(result),
                        }
                    )
                    report["checks"].append(
                        {
                            "sampling": name,
                            "scope": "cache_on_reference",
                            "enabled": enabled,
                            "kind": case["kind"],
                            "equal": result.token_ids == expected,
                            "prompt_tokens_match": result.prompt_tokens == case["prompt_tokens"],
                        }
                    )
                    save()
        report["all_token_checks_equal"] = all(c["equal"] for c in report["checks"])
        report["prompt_calibration_matches"] = all(
            c.get("prompt_tokens_match", True) for c in report["checks"]
        )
        passed = report["all_token_checks_equal"] and report["prompt_calibration_matches"]
        report["status"] = "completed" if passed else "validation_failed"
    finally:
        if report["status"] == "running":
            report["status"] = "incomplete"
        try:
            engine.close()
        finally:
            save()
    print(
        json.dumps({"all_token_checks_equal": report["all_token_checks_equal"], "report": str(args.output)})
    )
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
