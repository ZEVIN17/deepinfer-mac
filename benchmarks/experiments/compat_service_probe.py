"""Short, serial real-service compatibility audit; not a performance benchmark."""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from http_service_compare import measure
from public_service_report import public_report
from rotate_service_compare import await_ready, require_free_local_port, stop

from deepinfer_mac.provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text())
    manifest = json.loads(Path("benchmarks/experiments/cache_sessions.v1.json").read_text())
    cases = manifest["sessions"][0]["cases"][:3]
    engines = {e["id"]: e for e in configuration["engines"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audits = args.output.parent / "phase2-service-compat-audits"
    audits.mkdir(exist_ok=True)
    report = {
        "scope": "Actual 9B8bit service compatibility only; source is changing; no speed ratio claims",
        "initial_source": source_manifest(),
        "configuration": configuration,
        "engines": [],
    }

    def save():
        private = Path("benchmarks/results/phase2-service-compat-private.json")
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        snapshot = next(e["model"] for e in configuration["engines"] if e["id"] == "native-mlx-lm")
        args.output.write_text(
            json.dumps(public_report(report, snapshot=snapshot), ensure_ascii=False, indent=2) + "\n"
        )

    for name in ("omlx", "native-mlx-lm", "deepinfer-native-cache"):
        engine = engines[name]
        audit_path = audits / f"{name}-prompt-tokens.jsonl"
        runtime_path = audits / f"{name}-runtime.jsonl"
        log_path = audits / f"{name}.log"
        # Do not mix token audit sessions after a retry; retain older evidence.
        if audit_path.exists():
            audit_path = audit_path.with_name(f"{name}-prompt-tokens-{time.time_ns()}.jsonl")
        command = engine["command"] + ["--token-audit", str(audit_path)]
        if name == "omlx":
            command += ["--runtime-audit", str(runtime_path)]
        item = {
            "engine": name,
            "command": command,
            "records": [],
            "error": None,
            "source_at_start": source_manifest(),
            "token_audit_path": str(audit_path),
            "log": str(log_path),
        }
        report["engines"].append(item)
        process = None
        try:
            require_free_local_port(engine["url"])
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=os.environ | engine.get("environment", {}),
                )
                item["pid"] = process.pid
                item["models"] = await_ready(process, engine["url"], timeout=180)
                save()
                for case in cases:
                    record = measure(
                        engine["url"],
                        engine["model"],
                        case,
                        request_options=engine.get("request_options", {}),
                    )
                    item["records"].append(record)
                    save()
                    print(name, case["kind"], record["quality_pass"], record["error"], flush=True)
                    if record["error"]:
                        break
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                malformed = urllib.request.Request(
                    engine["url"] + "/v1/chat/completions",
                    data=b'{"model":',
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with opener.open(malformed, timeout=10) as response:
                        item["malformed_json"] = {"status": response.status, "body": response.read().decode()}
                except urllib.error.HTTPError as error:
                    item["malformed_json"] = {"status": error.code, "body": error.read().decode()}
                if name == "deepinfer-native-cache" and item["records"] and not item["records"][0]["error"]:
                    sampling_output = args.output.parent / "phase2-service-sampling-qwen35-9b8.json"
                    probe = subprocess.run(
                        [
                            engine["command"][0],
                            "benchmarks/experiments/phase2_service_probe.py",
                            "--base-url",
                            engine["url"],
                            "--official-sampling",
                            "--output",
                            str(sampling_output),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=240,
                    )
                    item["official_sampling_transport"] = {
                        "exit_code": probe.returncode,
                        "stdout": probe.stdout,
                        "stderr": probe.stderr,
                        "report": str(sampling_output),
                    }
                if runtime_path.exists():
                    item["runtime_audit"] = [
                        json.loads(line) for line in runtime_path.read_text().splitlines()
                    ]
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            item["error"] = f"{type(error).__name__}: {error}"
        finally:
            if process is not None:
                stop(process)
            item["stopped"] = process is None or process.poll() is not None
            if audit_path.exists():
                item["prompt_audits"] = [json.loads(line) for line in audit_path.read_text().splitlines()]
            item["source_at_end"] = source_manifest()
            save()
    first_hashes = {
        e["engine"]: e["prompt_audits"][0]["token_ids_sha256_le_u32"]
        for e in report["engines"]
        if e.get("prompt_audits")
    }
    report["first_prompt_hashes"] = first_hashes
    report["all_first_prompt_ids_equal"] = len(first_hashes) == 3 and len(set(first_hashes.values())) == 1
    report["final_source"] = source_manifest()
    save()


if __name__ == "__main__":
    main()
