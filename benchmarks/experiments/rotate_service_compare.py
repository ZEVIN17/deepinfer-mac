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
from public_service_report import public_report


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
    if "project_source_at_preparation" in configuration:
        from deepinfer_mac.provenance import source_manifest

        expected = configuration["project_source_at_preparation"]["source_tree_sha256"]
        if source_manifest()["source_tree_sha256"] != expected:
            raise ValueError(
                "Runtime sources changed after preparation; freeze and prepare again before measuring"
            )
    engines = configuration["engines"]
    if len(engines) < 2 or len({e["id"] for e in engines}) != len(engines):
        raise ValueError("Configure at least two engines with unique IDs")
    identities = {(e["weights_repo"], e["weights_revision"], e["quantization"]) for e in engines}
    if len(identities) != 1:
        raise ValueError("Strict engine comparison requires identical pinned weights/quantization")
    budgets = {e["cache_budget_bytes"] for e in engines}
    if len(budgets) != 1:
        raise ValueError("Cache budgets must match")
    sessions = [s for s in manifest["sessions"] if s["split"] == split]
    if not sessions:
        raise ValueError("Selected manifest split has no sessions")
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
    warmup_case = next(s for s in manifest["sessions"] if s["split"] == "dev")["cases"][0]
    report = {
        "status": "running",
        "planned_repetitions": repeats,
        "configuration": configuration,
        "manifest": manifest,
        "scope": "Exclusive server processes; warm-model session workload; no cold-KV assumption",
        "blocks": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "raw.json"

    def save():
        private = Path("benchmarks/results") / output.name / "raw-private.json"
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        snapshot = next((e["model"] for e in engines if e["id"] == "native-mlx-lm"), None)
        checkpoint.write_text(
            json.dumps(public_report(report, snapshot=snapshot), ensure_ascii=False, indent=2) + "\n"
        )

    for repetition in range(repeats):
        order = engines[repetition % len(engines) :] + engines[: repetition % len(engines)]
        for engine in order:
            block = {
                "engine": engine["id"],
                "repetition": repetition,
                "records": [],
                "error": None,
                "status": "running",
            }
            report["blocks"].append(block)
            log_path = output / f"{repetition}-{engine['id']}.log"
            launched = time.perf_counter()
            process = None
            try:
                require_free_local_port(engine["url"])
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        engine["command"],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=os.environ | engine.get("environment", {}),
                    )
                    block["pid"] = process.pid
                    block["models_response"] = await_ready(process, engine["url"], 180)
                    block["readiness_ms"] = (time.perf_counter() - launched) * 1000
                    options = engine.get("request_options", {})
                    block["warmup"] = measure(
                        engine["url"], engine["model"], warmup_case, request_options=options
                    )
                    if block["warmup"]["error"]:
                        raise RuntimeError(f"Warmup protocol failed: {block['warmup']['error']}")
                    save()
                    for session in sessions:
                        for case in session["cases"]:
                            record = measure(engine["url"], engine["model"], case, request_options=options)
                            record.update(session_id=session["id"], split=split, repetition=repetition)
                            block["records"].append(record)
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
                    stop(process)
                block["process_stopped"] = process is None or process.poll() is not None
                if block["status"] == "running":
                    block["status"] = "interrupted"
                save()
    report["status"] = (
        "completed"
        if all(b["status"] == "completed" for b in report["blocks"])
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text())
    raw = args.manifest.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != configuration["manifest_sha256"]:
        raise ValueError("Manifest differs from predeclared configuration")
    run(configuration, json.loads(raw), args.output, args.repeats, args.split)


if __name__ == "__main__":
    main()
