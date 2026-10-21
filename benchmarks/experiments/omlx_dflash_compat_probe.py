"""Real, sequential, instrumented oMLX DFlash dev compatibility probe.

Requires explicit GPU ownership before --run. Preparation and --help do not
import MLX or load a model. The instrumented results are never latency evidence.
"""

from __future__ import annotations

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

from http_service_compare import measure
from public_service_report import public_report

TARGET_REVISION = "16daa4818c54ce5f5436f929d52542eb65bbed9d"
DRAFT_REVISION = "5fc3b3d474760f18c516db87d84c37edbfd3ede6"
MODEL_ID = "Qwen3.5-9B-8bit"
VARIANTS = {
    "baseline": {"dflash_enabled": False, "dflash_draft_model": None},
    **{f"{quant}-b{block}-{mode}": {
        "dflash_enabled": True,
        "dflash_draft_quant_enabled": quant == "q4",
        "dflash_draft_quant_weight_bits": 4,
        "dflash_draft_quant_group_size": 64,
        "dflash_block_size": block,
        "dflash_verify_mode": mode,
        "dflash_draft_window_size": None,
    } for quant in ("q4", "bf16") for block in (4, 5) for mode in ("adaptive", "dflash")},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def system_memory():
    import psutil

    vm, swap = psutil.virtual_memory(), psutil.swap_memory()
    return {"available_bytes": vm.available, "used_bytes": vm.used,
            "system_swap_used_bytes": swap.used}


def prepare(args):
    if args.target.name != TARGET_REVISION or not (args.target / "config.json").is_file():
        raise ValueError("Use the complete pinned 9B8 target snapshot")
    if args.draft.name != DRAFT_REVISION or not (args.draft / "model.safetensors").is_file():
        raise ValueError("Use the complete SHA-verified official trained DFlash snapshot")
    args.root.mkdir(parents=True, exist_ok=True)
    model_dir = args.root / "models"
    model_dir.mkdir(exist_ok=True)
    link = model_dir / MODEL_ID
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != args.target.resolve():
            raise ValueError("Preserve unrelated existing model alias")
    else:
        link.symlink_to(args.target.resolve(), target_is_directory=True)
    engines = []
    for variant in args.variants:
        settings = {"model_type_override": "llm", "enable_thinking": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "dflash_draft_model": str(args.draft.resolve()), **VARIANTS[variant]}
        base = args.root / variant
        base.mkdir(exist_ok=True)
        settings_path = base / "model_settings.json"
        value = {"version": 1, "models": {MODEL_ID: settings}}
        if settings_path.exists() and json.loads(settings_path.read_text()) != value:
            raise ValueError("Existing dev configuration differs; use a fresh --root")
        settings_path.write_text(json.dumps(value, indent=2) + "\n")
        audit = base / "runtime.jsonl"
        command = [str(args.omlx_python), str(Path(__file__).with_name("omlx_dflash_audit_server.py").resolve()),
                   "--dflash-audit", str(audit.resolve()), "serve", "--model-dir", str(model_dir.resolve()),
                   "--base-path", str(base.resolve()), "--host", "127.0.0.1", "--port", str(args.port),
                   "--hot-cache-max-size", "512MB", "--max-concurrent-requests", "1",
                   "--memory-guard", args.memory_guard, "--headless"]
        engines.append({"id": variant, "settings": settings, "command": command,
                        "memory_guard": args.memory_guard,
                        "audit_path": str(audit), "log_path": str(base / "server.log")})
    return engines


def cases(manifest, ledger_length=2048, ledger_kind="seed_cold_kv"):
    value = json.loads(manifest.read_text())
    session = next(session for session in value["sessions"] if session["split"] == "dev"
                   and session["cases"][0]["prompt_tokens"] == ledger_length)
    ledger = dict(next(case for case in session["cases"] if case["kind"] == ledger_kind))
    ledger["id"] = session["id"] + ":" + ledger_kind
    return [
        {"id": "authored-multiply", "kind": "strict_json", "messages": [{"role": "user", "content":
         'Return exactly one JSON object with key "answer" containing the integer result of 6 times 7. No code fences or explanation.'}],
         "max_tokens": 64, "expected_json": {"answer": 42}},
        {"id": "authored-squares", "kind": "strict_json", "messages": [{"role": "user", "content":
         'Return exactly one JSON array of the squares of the integers 0 through 15 in ascending input order. No code fences or explanation.'}],
         "max_tokens": 192, "expected_json": [i * i for i in range(16)]},
        ledger,
    ]


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def assert_free_port(port):
    with socket.socket() as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} already occupied; preserve existing service")


def wait_ready(process, port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before readiness: {process.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                return json.load(response)
        except (OSError, ValueError):
            time.sleep(0.25)
    raise TimeoutError("Server readiness deadline")


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    return process.returncode


def run_engine(args, engine, test_cases):
    assert_free_port(args.port)
    audit_path = Path(engine["audit_path"])
    if audit_path.exists():
        raise ValueError("Preserve previous runtime audit; use a fresh --root")
    result = {"engine": engine, "records": [], "system_before": system_memory(), "error": None}
    environment = dict(os.environ, OMLX_HOT_CACHE_ONLY="true", HF_HUB_OFFLINE="1")
    with Path(engine["log_path"]).open("w") as log:
        process = subprocess.Popen(engine["command"], stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True, cwd=Path.cwd())
        try:
            result["models_response"] = wait_ready(process, args.port)
            for case in test_cases:
                before = len(read_events(audit_path))
                raw = measure(f"http://127.0.0.1:{args.port}", MODEL_ID, case, timeout=300,
                              request_options={"stream_options": {"include_usage": True},
                                               "chat_template_kwargs": {"enable_thinking": False}})
                # Timing is intentionally excluded: instrumentation and startup
                # order invalidate performance interpretation of this probe.
                record = {key: raw.get(key) for key in
                          ("case", "request", "text", "error", "protocol_complete", "quality_pass", "cache_observation")}
                record["usage"] = [event["payload"]["usage"] for event in raw["events"]
                                   if event["payload"].get("usage")]
                record["audit_events"] = read_events(audit_path)[before:]
                record["system_after"] = system_memory()
                result["records"].append(record)
                (audit_path.parent / "partial-records.json").write_text(json.dumps(result, indent=2) + "\n")
                print(engine["id"], case["id"], "quality", raw["quality_pass"], "protocol", raw["protocol_complete"], flush=True)
                if raw["error"]:
                    break
        except (OSError, ValueError, RuntimeError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            result["server_exit_code"] = stop(process)
    result["audit_events"] = read_events(audit_path)
    result["system_after"] = system_memory()
    result["system_swap_delta_bytes"] = result["system_after"]["system_swap_used_bytes"] - result["system_before"]["system_swap_used_bytes"]
    attached = [event for event in result["audit_events"] if event["event"] == "drafter_attached"]
    drafts = [event for event in result["audit_events"] if event["event"] == "draft_batched_completed" and event["rows"] > 0]
    verifies = [event for event in result["audit_events"] if event["event"] == "upstream_verify_stats" and event["stats"]["cycles"] > 0]
    result["dflash_execution_proven"] = bool(attached and drafts and verifies and all(
        event["attached_identity"] and event["decode_enabled"] and event["engine_property_identity"] for event in attached))
    result["all_strict_cases_passed"] = len(result["records"]) == len(test_cases) and all(
        record["quality_pass"] and record["protocol_complete"] for record in result["records"])
    text = Path(engine["log_path"]).read_text(errors="replace")
    result["draft_load_soft_fallback_observed"] = "toggle ignored" in text
    result["official_drafter_loaded_log_observed"] = "DFlash drafter loaded:" in text
    result["draft_load_failure_lines"] = [line for line in text.splitlines() if "DFlash drafter load raised" in line]
    assert_free_port(args.port)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/deepinfer-omlx-dflash-probe"))
    parser.add_argument("--target", type=Path, default=Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-8bit/snapshots" / TARGET_REVISION)
    parser.add_argument("--draft", type=Path, default=Path("/tmp/deepinfer-dflash-hf/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots") / DRAFT_REVISION)
    parser.add_argument("--omlx-python", type=Path, default=Path("/tmp/deepinfer-omlx-venv/bin/python"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/experiments/cache_sessions.v2.json"))
    parser.add_argument("--ledger-length", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--ledger-kind", choices=("seed_cold_kv", "multiturn_continuation"), default="seed_cold_kv")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["baseline", "q4-b5-adaptive"])
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--memory-guard", choices=("balanced", "aggressive"), default="balanced")
    parser.add_argument("--run", action="store_true", help="Requires explicit exclusive GPU baton")
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase2-competitor-audit/dflash/compat-raw.json"),
                        help="Full diagnostic output stays excluded; publish only the compact path-free derivative")
    args = parser.parse_args()
    engines = prepare(args)
    report = {"scope": "Instrumented actual dev compatibility only; no timing or speedup claim",
              "target_repo": "mlx-community/Qwen3.5-9B-8bit", "target_revision": TARGET_REVISION,
              "draft_repo": "z-lab/Qwen3.5-9B-DFlash", "draft_revision": DRAFT_REVISION,
              "manifest_sha256": sha(args.manifest),
              "harness_sha256": {name: sha(Path(__file__).with_name(name)) for name in
                                 (Path(__file__).name, "omlx_dflash_audit_server.py", "http_service_compare.py", "public_service_report.py")},
              "status": "prepared_not_run", "engines": engines, "results": []}
    private = args.root / "report-private.json"

    def save():
        private.write_text(json.dumps(report, indent=2) + "\n")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(public_report(report, snapshot=args.target), indent=2) + "\n")

    save()
    if not args.run:
        print("CPU configuration prepared; GPU was not imported or loaded.")
        return
    test_cases = cases(args.manifest, args.ledger_length, args.ledger_kind)
    try:
        for engine in engines:
            report["results"].append(run_engine(args, engine, test_cases))
            save()
        report["status"] = "actual_dev_probe_completed"
        if report["results"] and report["results"][0]["engine"]["id"] == "baseline":
            baseline = report["results"][0]
            for result in report["results"][1:]:
                result["paired_baseline_outputs"] = [{
                    "case_id": candidate["case"]["id"], "text_equal": reference["text"] == candidate["text"],
                    "prompt_ids_equal": [e["token_ids"] for e in reference["audit_events"] if e["event"] == "request_prompt"] ==
                                        [e["token_ids"] for e in candidate["audit_events"] if e["event"] == "request_prompt"],
                } for reference, candidate in zip(baseline["records"], result["records"])]
    finally:
        save()


if __name__ == "__main__":
    main()
