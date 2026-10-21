"""Explicit real-Metal API smoke test; run manually, never substitutes for unit tests.

PYTHONPATH=src .venv/bin/python tests/metal_service_smoke.py --profile qwen3-0.6b
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="qwen3-0.6b")
    parser.add_argument("--output", default="benchmarks/results/metal-service-smoke.json")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"profile": args.profile, "checks": {}, "fake_engine": False}
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "deepinfer_mac.cli", "serve", "--port", str(port)],
                                   stdout=log, stderr=log, env={**os.environ, "DEEPINFER_MAC_TOKEN": ""})
        base = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(base_url=base, timeout=180) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError("Server exited; inspect the log")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError("Server did not become healthy")
                body = {"model": args.profile, "messages": [{"role": "user", "content": "用一句话解释缓存。"}],
                        "mode": "baseline", "max_tokens": 24}
                response = client.post("/v1/chat/completions", json=body)
                response.raise_for_status()
                result = response.json()
                assert result["usage"]["completion_tokens"] > 0
                evidence["checks"]["non_stream"] = result
                with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as stream:
                    stream.raise_for_status()
                    lines = list(stream.iter_lines())
                    assert "data: [DONE]" in lines
                    assert any('"content"' in line for line in lines)
                    evidence["checks"]["stream"] = {"events": len(lines), "done": True}
                # Disconnect an actual long stream after its first content event.
                with client.stream("POST", "/v1/chat/completions",
                                   json={**body, "messages": [{"role": "user", "content": "Write Python functions for 100 sorting algorithms with explanations."}],
                                         "max_tokens": 1024, "stream": True}) as stream:
                    stream.raise_for_status()
                    run_id = stream.headers["X-Run-ID"]
                    for line in stream.iter_lines():
                        if '"content"' in line:
                            break
                for _ in range(300):
                    disconnected = client.get(f"/api/runs/{run_id}").json()
                    if disconnected["status"] in ("cancelled", "completed", "failed"):
                        break
                    time.sleep(.05)
                assert disconnected["status"] == "cancelled", disconnected
                evidence["checks"]["disconnect_cancellation"] = disconnected
                # The owner must recover after cancellation and complete a three-mode comparison.
                compare = client.post("/api/compare", json={"profile": args.profile, "prompt": "What is 2 + 2?", "max_tokens": 8})
                compare.raise_for_status()
                identifier = compare.json()["id"]
                for _ in range(1200):
                    compared = client.get(f"/api/runs/{identifier}").json()
                    if compared["status"] in ("completed", "cancelled", "failed"):
                        break
                    time.sleep(.1)
                assert compared["status"] == "completed", compared
                evidence["checks"]["comparison_after_cancellation"] = compared
                evidence["ok"] = True
        except Exception as exc:
            evidence["ok"] = False
            evidence["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            evidence["server_exit_code"] = process.returncode
            output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
