import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import deepinfer_mac
from deepinfer_mac.adapters import ModelRegistry, ModelSpec
from deepinfer_mac.continuous import ContinuousConfig
from deepinfer_mac.engine import InferenceEngine, read_swap_bytes
from deepinfer_mac.server.app import create_app

parser = argparse.ArgumentParser(
    description="Installed-wheel real-model functional smoke; requires MLX and httpx."
)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
assert "site-packages" in deepinfer_mac.__file__, "Run with an installed wheel, not an editable checkout"
source = args.model.resolve()
checks = {}
swap = read_swap_bytes()
p = subprocess.run(
    [
        sys.executable,
        "-m",
        "deepinfer_mac.cli",
        "run",
        "--model",
        str(source),
        "--model-id",
        "smollm2-135m-8bit",
        "--max-tokens",
        "16",
        "--json",
        "Explain why rain falls.",
    ],
    check=False,
    capture_output=True,
    text=True,
    timeout=120,
)
checks["installed_cli_real_generation"] = p.returncode == 0
if p.returncode:
    raise RuntimeError(p.stderr)
cli = json.loads(p.stdout)
registry = ModelRegistry([ModelSpec(id="smollm2-135m-8bit", source=str(source))])
app = create_app(
    engine_factory=lambda: InferenceEngine(model_registry=registry),
    model_registry=registry,
    execution_backend="auto",
    continuous_config=ContinuousConfig(policy="upstream", max_active=2),
)
with TestClient(app) as client:
    checks["health_auto"] = client.get("/health").json()["execution_backend"] == "auto"
    checks["bundled_ui"] = client.get("/").status_code == 200
    checks["bundled_js"] = client.get("/assets/app.js").status_code == 200
    body = {
        "model": "smollm2-135m-8bit",
        "messages": [{"role": "user", "content": "Explain why rain falls."}],
        "max_tokens": 16,
    }
    response = client.post("/v1/chat/completions", json=body)
    checks["chat_http_200"] = response.status_code == 200
    data = response.json()
    checks["native_strategy"] = data["deepinfer"]["metrics"]["selected_execution_strategy"] == "native_batch"
    response = client.post("/v1/chat/completions", json={**body, "stream": True})
    checks["sse_done"] = response.status_code == 200 and "data: [DONE]" in response.text
    checks["sse_run_id"] = bool(response.headers.get("x-run-id"))
    checks["actual_output"] = bool(data["choices"][0]["message"]["content"])
checks["no_swap_growth"] = read_swap_bytes() <= swap
report = {
    "version": deepinfer_mac.__version__,
    "installed_wheel": True,
    "source_checkout_imported": False,
    "checks": checks,
    "output_sha256": hashlib.sha256(data["choices"][0]["message"]["content"].encode()).hexdigest(),
    "cli_token_count": len(cli.get("token_ids", [])),
    "scope": "Real 135M model; installed wheel CLI, native API, SSE and bundled UI resources. Functional smoke, not performance benchmark.",
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
assert all(checks.values())
