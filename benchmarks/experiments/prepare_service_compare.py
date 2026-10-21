"""Prepare isolated, pinned service configuration without loading any model."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, default=Path("/tmp/deepinfer-service-compare"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/experiments/cache_sessions.v1.json")
    )
    parser.add_argument("--total-context-cap", type=int, default=4096)
    parser.add_argument("--omlx-python", type=Path, default=Path("/tmp/deepinfer-omlx-venv/bin/python"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    snapshot = args.snapshot.resolve()
    revision = "16daa4818c54ce5f5436f929d52542eb65bbed9d"
    if snapshot.name != revision or not (snapshot / "config.json").is_file():
        raise ValueError("Pass the pinned, complete Qwen3.5-9B-8bit snapshot")
    args.root.mkdir(parents=True, exist_ok=True)
    model_id = "Qwen3.5-9B-8bit"
    model_dir = args.root / "models"
    model_dir.mkdir(exist_ok=True)
    link = model_dir / model_id
    if link.is_symlink() and link.resolve() != snapshot:
        raise ValueError("Existing model alias points to different weights")
    if not link.exists():
        link.symlink_to(snapshot, target_is_directory=True)
    base = args.root / "omlx-base"
    base.mkdir(exist_ok=True)
    settings = {
        "version": 1,
        "models": {
            model_id: {
                "model_type_override": "llm",
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
    }
    settings_path = base / "model_settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Existing benchmark model settings differ; use a fresh isolated root")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    python = str(workspace / ".venv/bin/python")
    experiments = workspace / "benchmarks/experiments"
    shared = {
        "weights_repo": "mlx-community/Qwen3.5-9B-8bit",
        "weights_revision": revision,
        "quantization": "8bit affine group64",
        "cache_budget_bytes": 536870912,
        "enable_thinking": False,
        "temperature": 0,
        "concurrency": 1,
    }
    engines = [
        shared
        | {
            "id": "deepinfer-native-cache",
            "model": "qwen3.5-9b-8bit",
            "version": "working-tree",
            "url": "http://127.0.0.1:8082",
            "command": [
                python,
                str(experiments / "deepinfer_service_server.py"),
                "--port",
                "8082",
                "--cache-budget-bytes",
                "536870912",
                "--cache-backend",
                "native",
                "--checkpoint-interval-tokens",
                "512",
            ],
        },
        shared
        | {
            "id": "native-mlx-lm",
            "model": str(snapshot),
            "version": "mlx-lm-0.32.2",
            "url": "http://127.0.0.1:8081",
            "request_options": {"stream_options": {"include_usage": True}},
            "command": [
                python,
                str(experiments / "native_lru_server.py"),
                "--model",
                str(snapshot),
                "--host",
                "127.0.0.1",
                "--port",
                "8081",
                "--cache-budget-bytes",
                "536870912",
                "--prompt-cache-size",
                "10000",
                "--prefill-step-size",
                "512",
                "--chat-template-args",
                '{"enable_thinking":false}',
            ],
        },
        shared
        | {
            "id": "omlx",
            "model": model_id,
            "version": "0.7.1-bd1e9b112474862ebed550a9c83b4316038a1d74",
            "url": "http://127.0.0.1:8083",
            "environment": {"OMLX_HOT_CACHE_ONLY": "true"},
            "request_options": {
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "command": [
                str(args.omlx_python),
                str(experiments / "omlx_service_server.py"),
                "serve",
                "--model-dir",
                str(model_dir),
                "--base-path",
                str(base),
                "--host",
                "127.0.0.1",
                "--port",
                "8083",
                "--hot-cache-max-size",
                "512MB",
                "--max-concurrent-requests",
                "1",
                "--headless",
            ],
        },
    ]
    manifest = args.manifest.resolve()
    result = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "status": "prepared_only_not_loaded_or_measured",
        "total_context_cap_tokens": args.total_context_cap,
        "resource_probe_required_above_cap": True,
        "engines": engines,
        "notes": [
            "oMLX hot-cache-only is its official RAM-cache mode; SSD is a separate experiment.",
            "Pure text official model_type_override=llm avoids charging vision weights in one engine.",
            "This is a cache control, not a DS mechanism or competitive superiority result.",
            "Validate template token IDs and actual byte units before formal heldout measurement.",
        ],
    }
    from deepinfer_mac.provenance import source_manifest

    result["project_source_at_preparation"] = source_manifest()
    destination = args.root / "configuration.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
