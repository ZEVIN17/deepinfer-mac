"""Opt-in actual SmolLM2 tick/lifecycle probe; never starts a listening server.

Defaults to a CPU-only plan. --execute requires the coordinator's GPU baton.
This functional probe preserves all outputs/divergences; it is not a competitor
benchmark, a statistical speed claim, or evidence for DS batch verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path


def first_difference(expected, actual):
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else None
        right = actual[index] if index < len(actual) else None
        if left != right:
            return {"index": index, "native": left, "continuous": right}
    return None


def wait_for(job, predicate, timeout=60):
    deadline = time.perf_counter() + timeout
    while True:
        if job.done.is_set():
            return False
        if predicate(job):
            return True
        if time.perf_counter() >= deadline:
            raise TimeoutError("Requested lifecycle point was not observed")
        time.sleep(0.001)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Load only the already prepared <=200MB dense model")
    parser.add_argument("--model-config", type=Path, default=Path("examples/models/smollm2-135m-8bit.local.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase3-continuous/probe.json"))
    parser.add_argument("--policy", choices=("upstream", "interactive"), default="interactive")
    args = parser.parse_args(argv)
    plan = {"execute": args.execute, "model_config": args.model_config.name, "policy": args.policy,
            "concurrency": [1, 2, 4], "arrival_offsets_ms": [0, 5, 10, 15],
            "directions": ["short_decode_then_long_prefill", "long_prefill_then_short_arrival"],
            "cancellation": ["prefill", "decode"], "performance_claim": False,
            "model_downloads": False, "listening_service": False}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an earlier actual probe")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # No MLX imports or model registration until the explicit execution flag.
    import mlx.core as mx
    import psutil

    from deepinfer_mac.adapters import ModelRegistry, ModelSpec, inspect_snapshot
    from deepinfer_mac.continuous import ContinuousConfig
    from deepinfer_mac.engine import InferenceEngine, read_swap_bytes
    from deepinfer_mac.server.continuous_worker import ContinuousInferenceWorker
    from deepinfer_mac.state import GenerationRequest

    settings = json.loads(args.model_config.read_text())
    source = Path(settings["source"])
    if not source.is_dir():
        raise ValueError("Probe only uses existing local weights; downloading is prohibited")
    audit = inspect_snapshot(source)
    if audit["weight_bytes"] > 200_000_000 or audit["config"].get("model_type") != "llama":
        raise ValueError("This probe is bounded to the prepared <=200MB dense Llama-family model")
    registry = ModelRegistry([ModelSpec.from_json(args.model_config)])
    profile = settings["id"]
    config = ContinuousConfig(max_active=4, prefill_chunk=32, tick_token_budget=64,
                              max_context_tokens=12288, policy=args.policy)
    small = GenerationRequest(profile=profile, prompt="Explain the water cycle in detail.", max_tokens=48, use_cache=False)
    long = replace(small, prompt=("A document about water and clouds. " * 180) + "\nSummarize this document.", max_tokens=24)
    chinese = replace(small, prompt="请用中文解释为什么天空是蓝色的。", max_tokens=48)
    code = replace(small, prompt="Write a Python function that sums a list of integers.", max_tokens=48)
    requests = [small, long, chinese, code]
    root = Path.cwd().resolve()
    sources = sorted({*(path.resolve() for path in Path("src/deepinfer_mac").rglob("*.py")), Path(__file__).resolve(), args.model_config.resolve()})
    def source_hashes():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = {"schema_version": 1, "purpose": "Real continuous baseline lifecycle, token control and tick emission",
              "plan": plan, "config": asdict(config), "model": {"id": profile, "weight_bytes": audit["weight_bytes"],
              "content_sha256": audit["content_sha256"]}, "started_at_unix": time.time(),
              "source_sha256": source_hashes(),
              "native": [], "cases": [], "failures": [], "swap_before_bytes": read_swap_bytes(),
              "versions": {"python": sys.version.split()[0], "mlx": version("mlx"), "mlx_lm": version("mlx-lm"),
                           "macos": platform.mac_ver()[0], "machine": platform.machine()},
              "limits": ["One process functional probe; no stable performance conclusion",
                         "Upstream decode-first ticks; decode delivery still waits for current prefill synchronization",
                         "One native prefill slot; later short prompt does not preempt a long prefill",
                         "Cold prefix cache in continuous path; greedy baseline only, no DS varlen verification"]}
    def resources():
        return {"available_bytes": psutil.virtual_memory().available,
                "rss_snapshot_bytes": psutil.Process().memory_info().rss,
                "metal_active_bytes": mx.get_active_memory(), "metal_peak_bytes": mx.get_peak_memory(),
                "swap_used_bytes": read_swap_bytes()}
    report["resources_before_native"] = resources()
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    reference = {}
    engine = InferenceEngine(model_registry=registry, cache_budget_bytes=0)
    try:
        for index, request in enumerate(requests):
            result = engine.native_generate(request)
            reference[index] = result.token_ids
            report["native"].append({"input_index": index, "prompt_token_ids": engine._tokens(request),
                                     "output_token_ids": result.token_ids, "text": result.text,
                                     "finish_reason": result.finish_reason, "metrics": result.metrics})
            save()
    except Exception as exc:  # noqa: BLE001 — preserve native failure instead of dropping the run
        report["failures"].append({"stage": "native", "type": type(exc).__name__, "message": str(exc).replace(str(root), "$REPO")})
        save()
        return 1
    finally:
        try:
            engine.close()
        except Exception as exc:  # noqa: BLE001 — native cleanup failure is evidence too
            report["failures"].append({"stage": "native_close", "type": type(exc).__name__,
                                       "message": str(exc).replace(str(root), "$REPO")})
            save()
    if report["failures"]:
        return 1
    mx.reset_peak_memory()
    report["resources_before_continuous"] = resources()
    worker = ContinuousInferenceWorker(lambda: InferenceEngine(model_registry=registry, cache_budget_bytes=0),
                                       queue_size=8, config=config)
    def collect(name, indexed_jobs, started):
        outputs = []
        case = {"name": name, "requests": outputs, "incomplete": True}
        report["cases"].append(case)
        save()
        for index, job in indexed_jobs:
            if not job.done.wait(120):
                worker.cancel_job(job.id)
                outputs.append({"input_index": index, "request_id": job.id, "status": job.status,
                                "snapshot": job.snapshot(), "events": list(job.events), "error": "probe timeout",
                                "full_token_equality": False, "result": job.result})
                case["failure"] = "request_timeout"
                save()
                raise TimeoutError("Probe request did not finish")
            result = job.result
            difference = first_difference(reference[index], result["token_ids"]) if result else None
            outputs.append({"input_index": index, "request_id": job.id, "status": job.status,
                            "lifecycle": job.lifecycle, "slot": job.slot, "slot_generation": job.slot_generation,
                            "submitted_offset_ms": (job.submitted_at - started) * 1000,
                            "events": list(job.events), "result": result, "error": job.error,
                            "full_token_equality": result is not None and difference is None,
                            "first_difference": difference})
            save()
        wall_ms = (time.perf_counter() - started) * 1000
        def percentiles(values):
            if not values:
                return {"count": 0, "p50": None, "p95": None}
            quantiles = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else [values[0]] * 99
            return {"count": len(values), "p50": quantiles[49], "p95": quantiles[94]}
        completed = [output["result"] for output in outputs if output["result"] is not None]
        case.update({"incomplete": False, "wall_ms": wall_ms,
                                "resources_after_case": resources(),
                                "descriptive_only": True,
                                "ttft_ms": percentiles([result["metrics"]["ttft_ms"] for result in completed if result["metrics"]["ttft_ms"] is not None]),
                                "itl_ms": percentiles([value for result in completed for value in result["metrics"]["itl_ms"]]),
                                "queue_ms": percentiles([result["metrics"]["queue_ms"] for result in completed]),
                                "completed_requests": len(completed), "submitted_requests": len(outputs),
                     "observed_tokens_per_second": sum(len(result["token_ids"]) for result in completed) / (wall_ms / 1000)})
        save()
    try:
        for concurrency in (1, 2, 4):
            start = time.perf_counter()
            jobs = []
            for index in range(concurrency):
                arrival = start + index * 0.005
                while time.perf_counter() < arrival:
                    time.sleep(max(0, min(0.001, arrival - time.perf_counter())))
                jobs.append((index, worker.submit(requests[index])))
            collect(f"fixed_arrivals_concurrency_{concurrency}", jobs, start)
        # A token must actually be emitted before the long request is submitted.
        start = time.perf_counter()
        short_job = worker.submit(small)
        observed_decode = wait_for(short_job, lambda job: any(e["type"] == "token" for e in job.events))
        long_job = worker.submit(long)
        collect("short_decode_then_long_prefill", [(0, short_job), (1, long_job)], start)
        report["cases"][-1]["decode_observed_before_long_submission"] = observed_decode
        short_ticks = {event["tick"] for event in short_job.events if event["type"] == "token"}
        long_ticks = {event["tick"] for event in long_job.events if event["type"] == "prefill" and not event["end_of_prompt"]}
        report["cases"][-1]["actual_decode_prefill_shared_ticks"] = sorted(short_ticks & long_ticks)
        start = time.perf_counter()
        long_job = worker.submit(long)
        observed_prefill = wait_for(long_job, lambda job: any(e["type"] == "prefill" and not e["end_of_prompt"] for e in job.events))
        short_job = worker.submit(small)
        collect("long_prefill_then_short_arrival", [(1, long_job), (0, short_job)], start)
        report["cases"][-1]["prefill_observed_before_short_submission"] = observed_prefill
        for phase in ("prefill", "decode"):
            start = time.perf_counter()
            job = worker.submit(long if phase == "prefill" else small)
            observed = wait_for(job, lambda candidate, phase=phase: any(
                event["type"] == ("prefill" if phase == "prefill" else "token") for event in candidate.events))
            worker.cancel_job(job.id)
            if not job.done.wait(60):
                raise TimeoutError("Cancellation did not drain")
            recovery = worker.submit(small)
            collect(f"cancel_{phase}_then_recover", [(0, recovery)], start)
            report["cases"][-1]["cancelled"] = {"request_id": job.id, "observed_before_cancel": observed,
                                                 "status": job.status, "lifecycle": job.lifecycle,
                                                 "slot": job.slot, "slot_generation": job.slot_generation,
                                                 "events": list(job.events)}
            save()
    except Exception as exc:  # noqa: BLE001 — preserve all real failures and earlier cases
        report["failures"].append({"type": type(exc).__name__, "message": str(exc).replace(str(root), "$REPO")})
    finally:
        try:
            worker.close()
        except Exception as exc:  # noqa: BLE001
            report["failures"].append({"type": type(exc).__name__, "message": str(exc).replace(str(root), "$REPO")})
        report["swap_after_bytes"] = read_swap_bytes()
        report["resources_after_continuous"] = resources()
        if report["swap_before_bytes"] is not None and report["swap_after_bytes"] is not None:
            report["swap_delta_bytes"] = report["swap_after_bytes"] - report["swap_before_bytes"]
        report["finished_at_unix"] = time.time()
        report["all_completed_outputs_equal_native"] = bool(report["cases"]) and all(
            request["full_token_equality"] for case in report["cases"] for request in case["requests"])
        report["source_unchanged"] = source_hashes() == report["source_sha256"]
        direction = next((case for case in report["cases"] if case["name"] == "short_decode_then_long_prefill"), {})
        reverse_direction = next((case for case in report["cases"] if case["name"] == "long_prefill_then_short_arrival"), {})
        cancellations = [case.get("cancelled") for case in report["cases"] if case.get("cancelled")]
        report["actual_interleaving_observed"] = bool(direction.get("actual_decode_prefill_shared_ticks"))
        report["both_cancellation_phases_observed"] = len(cancellations) == 2 and all(
            item["observed_before_cancel"] and item["status"] == "cancelled" for item in cancellations)
        report["both_arrival_directions_observed"] = bool(
            direction.get("decode_observed_before_long_submission") and reverse_direction.get("prefill_observed_before_short_submission"))
        report["functional_probe_pass"] = (
            len(report["cases"]) == 7 and not report["failures"] and report["all_completed_outputs_equal_native"]
            and report["source_unchanged"] and report["actual_interleaving_observed"] and report["both_cancellation_phases_observed"]
            and report["both_arrival_directions_observed"])
        report["performance_claim"] = False
        save()
    print(json.dumps({"output": args.output.name, "functional_probe_pass": report["functional_probe_pass"],
                      "all_completed_outputs_equal_native": report["all_completed_outputs_equal_native"],
                      "failures": report["failures"]}, ensure_ascii=False, indent=2))
    return 0 if report["functional_probe_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
