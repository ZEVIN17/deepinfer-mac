"""Launch one server at a time and rotate complete real measurement blocks.

The configuration contains argv arrays, exact model/version/revision metadata,
and explicit request options. This script never downloads or changes weights.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from http_service_compare import measure
from omlx_dflash_protocol import execution_evidence
from public_service_report import public_report
from service_dev_selection import verify_formal
from service_inventory import warmup_cases
from service_resource_selection import selected_sessions
from service_resources import eligibility as resource_eligibility
from service_resources import sample as resource_sample
from service_source_freeze import check as source_check
from service_source_freeze import digest


def await_ready(process, url, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited during startup with status {process.returncode}")
        try:
            with opener.open(url.rstrip("/") + "/v1/models", timeout=2) as response:
                if response.status == 200:
                    return json.load(response)
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Server readiness timeout after {timeout}s")


def stop(process):
    # The group may still contain a worker after its original parent exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def require_free_local_port(url):
    parsed = urlparse(url)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.port is None:
        raise ValueError("Benchmark server URLs must name an explicit loopback port")
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=0.5):
            raise RuntimeError(f"Refusing to benchmark an existing listener at {url}")
    except (ConnectionRefusedError, TimeoutError):
        return


def run(configuration, manifest, output, repeats, split):
    if repeats < 1:
        raise ValueError("At least one complete repetition is required")
    if configuration.get("dev_tuning_grid") and split != "dev":
        raise ValueError("Chunk tuning candidates are dev-only; freeze independent choices before heldout")
    engines = configuration["engines"]
    if len(engines) < 2 or len({e["id"] for e in engines}) != len(engines):
        raise ValueError("Configure at least two engines with unique IDs")
    identities = {(e["weights_repo"], e["weights_revision"], e["quantization"]) for e in engines}
    if len(identities) != 1:
        raise ValueError("Strict engine comparison requires identical pinned weights/quantization")
    budgets = {e["cache_budget_bytes"] for e in engines}
    if len(budgets) != 1:
        raise ValueError("Cache budgets must match")
    for key in ("enable_thinking", "temperature", "concurrency"):
        if any(key in engine for engine in engines) and len({engine.get(key) for engine in engines}) != 1:
            raise ValueError(f"Sampling/concurrency configuration must match: {key}")
    if any(engine.get("concurrency", 1) != 1 for engine in engines):
        raise ValueError("This runner measures concurrency=1; 2/4 require a separate real concurrency protocol")
    sessions = selected_sessions(configuration, manifest, split)
    if not sessions:
        raise ValueError("Selected manifest split has no sessions")
    if (any(engine.get("dflash_execution_required") for engine in engines)
        and any(case.get("prompt_tokens", 0) >= 4096 for session in sessions for case in session["cases"])
        and not (configuration.get("omlx_dflash_resource_admission") or {}).get("verified")):
        raise ValueError("Actual4204+256 trained-DFlash resource admission is required before4K comparison; target-only4K and2K DFlash compatibility are insufficient")
    dev_selection_check = None
    if split == "heldout" and configuration.get("resource_policy"):
        verify_formal(configuration)
        dev_selection_check = {"verified": True, "binding_sha256": digest(configuration["dev_selection_report"])}
    if "total_context_cap_tokens" in configuration:
        cap = configuration["total_context_cap_tokens"]
        blocked = [
            (s["id"], c["kind"], c["requested_total_context_tokens"])
            for s in sessions
            for c in s["cases"]
            if c.get("requested_total_context_tokens", 0) > cap
        ]
        if blocked:
            raise ValueError(
                f"Requests exceed declared {cap}-token total cap; require real resource probe, never truncate: {blocked}"
            )
    warms = warmup_cases(configuration, manifest)
    report = {
        "status": "running",
        "planned_repetitions": repeats,
        "order_balance": {"engine_variants": len(engines), "complete_cycles": repeats // len(engines),
                          "fully_balanced_positions": repeats % len(engines) == 0},
        "configuration": configuration,
        "configuration_sha256": digest(configuration),
        "split": split,
        "manifest": manifest,
        "source_checks": {"initial": source_check(configuration, bind=True)},
        "scope": "Exclusive server processes; warm-model session workload; no cold-KV assumption",
        "cache_scope_note": "seed_cold_kv is a historical request-kind label, not proof of zero cache hits. Only actual engine metrics establish cold KV; unknown remains unknown.",
        "blocks": [],
        "dev_selection_check": dev_selection_check,
    }
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "raw.json"
    private_root = Path("benchmarks/results/private-service") / output.name
    private_root.mkdir(parents=True, exist_ok=True)

    def save():
        private = private_root / "raw-private.json"
        private.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        snapshot = next((e["model"] for e in engines if e["id"] == "native-mlx-lm"), None)
        public = public_report(report, snapshot=snapshot)
        public["configuration_sha256"] = digest(public["configuration"])
        if public.get("dev_selection_check"):
            public["dev_selection_check"]["binding_sha256"] = digest(public["configuration"]["dev_selection_report"])
        checkpoint.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n")

    if configuration.get("frozen_source") and not report["source_checks"]["initial"]["verified"]:
        report["status"] = "source_drift"
        report["source_checks"]["final"] = source_check(configuration, bind=True)
        save()
        return report

    try:
        for repetition in range(repeats):
            order = engines[repetition % len(engines) :] + engines[: repetition % len(engines)]
            for engine in order:
                block = {
                    "engine": engine["id"],
                    "repetition": repetition,
                    "records": [],
                    "error": None,
                    "status": "running",
                    "resource_samples": {"before_start": resource_sample(), "after_requests": []},
                }
                report["blocks"].append(block)
                log_path = private_root / f"block-{len(report['blocks']) - 1}.log"
                launched = time.perf_counter()
                process = None
                try:
                    block["source_checks"] = {"before": source_check(configuration, bind=True)}
                    if configuration.get("frozen_source") and not block["source_checks"]["before"]["verified"]:
                        raise RuntimeError("Source/environment drift before block")
                    require_free_local_port(engine["url"])
                    block["executed_command"] = engine["command"]
                    block["execution_cwd"] = engine.get("cwd")
                    with log_path.open("w") as log:
                        block["resource_samples"]["before_start"] = resource_sample()
                        process = subprocess.Popen(
                            engine["command"],
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            env=os.environ | engine.get("environment", {}),
                            cwd=engine.get("cwd"),
                        )
                        block["pid"] = process.pid
                        block["models_response"] = await_ready(process, engine["url"], 180)
                        block["resource_samples"]["after_ready"] = resource_sample(process.pid)
                        block["readiness_ms"] = (time.perf_counter() - launched) * 1000
                        options = engine.get("request_options", {})
                        block["warmups"] = []
                        for warmup_case in warms:
                            warmup = measure(engine["url"], engine["model"], warmup_case, request_options=options)
                            block["warmups"].append(warmup)
                            save()
                            if warmup["error"]:
                                raise RuntimeError(f"Warmup protocol failed: {warmup['error']}")
                        block["resource_samples"]["after_warmup"] = resource_sample(process.pid)
                        if engine.get("dflash_execution_required"):
                            block["dflash_warmup_execution"] = execution_evidence(log_path.read_text(errors="replace"), engine["dflash_execution_required"])
                            if not block["dflash_warmup_execution"]["valid"]:
                                raise RuntimeError("Official DFlash execution not proven after warmup; refuse timing a soft fallback")
                        for session in sessions:
                            for case in session["cases"]:
                                record = measure(engine["url"], engine["model"], case, request_options=options)
                                record.update(session_id=session["id"], split=split, repetition=repetition)
                                block["records"].append(record)
                                block["resource_samples"]["after_requests"].append(resource_sample(process.pid))
                                save()
                                print(
                                    engine["id"],
                                    repetition,
                                    session["id"],
                                    case["kind"],
                                    record["quality_pass"],
                                    record["total_ms"],
                                    flush=True,
                                )
                        block["status"] = "completed"
                except (OSError, RuntimeError, ValueError) as error:
                    block["error"] = f"{type(error).__name__}: {error}"
                    block["status"] = "failed"
                finally:
                    if process is not None:
                        block["resource_samples"]["before_stop"] = resource_sample(process.pid)
                        stop(process)
                    block["resource_samples"]["after_stop"] = resource_sample()
                    if configuration.get("resource_policy"):
                        block["resource_eligibility"] = resource_eligibility(block["resource_samples"], configuration["resource_policy"], len(block["records"]))
                    block["process_stopped"] = process is None or process.poll() is not None
                    if block["status"] == "running":
                        block["status"] = "interrupted"
                    block.setdefault("source_checks", {})["after"] = source_check(configuration, bind=True)
                    if configuration.get("frozen_source") and not block["source_checks"]["after"]["verified"]:
                        block["status"] = "failed"
                        block["error"] = (block.get("error") or "") + " Source/environment drift after block"
                    if log_path.exists():
                        if engine.get("dflash_execution_required"):
                            block["dflash_execution"] = execution_evidence(log_path.read_text(errors="replace"), engine["dflash_execution_required"])
                            if not block["dflash_execution"]["valid"]:
                                block["status"] = "failed"
                                block["error"] = (block.get("error") or "") + " Official DFlash execution evidence failed"
                        block["private_log"] = {"alias": f"<PRIVATE_RESULTS>/{output.name}/{log_path.name}",
                                                "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                                                "bytes": log_path.stat().st_size}
                    save()
    finally:
        report["source_checks"]["final"] = source_check(configuration, bind=True)
        report["status"] = (
            "completed"
            if len(report["blocks"]) == repeats * len(engines)
            and all(b["status"] == "completed" for b in report["blocks"])
            and (not configuration.get("frozen_source") or report["source_checks"]["final"]["verified"])
            else "completed_with_failures"
        )
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/experiments/cache_sessions.v1.json")
    )
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="CPU-only frozen source/environment check; never launch a service")
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text())
    if configuration.get("frozen_manifest") and args.manifest.resolve() != Path(configuration["frozen_manifest"]).resolve():
        raise ValueError("Execute the manifest from the frozen extraction, not a live workspace path")
    raw = args.manifest.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != configuration["manifest_sha256"]:
        raise ValueError("Manifest differs from predeclared configuration")
    if args.verify_only:
        verification = source_check(configuration, bind=True)
        print(json.dumps({"verified": verification["verified"], "errors": verification.get("errors", []),
                          "source_sha256": verification.get("source_sha256")}))
        if not verification["verified"]:
            raise SystemExit(1)
        return
    if args.output is None:
        parser.error("--output is required for actual service measurement")
    run(configuration, json.loads(raw), args.output, args.repeats, args.split)


if __name__ == "__main__":
    main()
