"""Prepare isolated, pinned service configuration without loading any model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from omlx_dflash_protocol import pinned_draft, resource_admission
from omlx_dflash_protocol import settings as dflash_settings
from service_dev_selection import bind as bind_dev_selection
from service_resource_selection import resource_selection
from service_resources import RESOURCE_POLICY
from service_source_freeze import digest, environment, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Explicit project/native interpreter, separate from frozen source")
    parser.add_argument("--root", type=Path, default=Path("/tmp/deepinfer-service-compare"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/experiments/cache_sessions.v1.json")
    )
    parser.add_argument("--total-context-cap", type=int, default=4096)
    parser.add_argument("--resource-approved-lengths", type=int, nargs="+")
    parser.add_argument("--own-prefill-step-size", type=int, choices=(128, 256, 512, 1024), default=512)
    parser.add_argument("--native-prefill-step-size", type=int, choices=(128, 256, 512, 1024), default=512)
    parser.add_argument("--ds-original-prefill-step-size", type=int, choices=(128, 256, 512, 1024))
    parser.add_argument("--ds-lagged-prefill-step-size", type=int, choices=(128, 256, 512, 1024))
    parser.add_argument("--dev-tuning-grid", action="store_true", help="Prepare independent chunk candidates; runner rejects heldout")
    parser.add_argument("--message-checkpoints", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "ds-original", "ds-lagged"), default=["baseline"])
    parser.add_argument("--omlx-python", type=Path, default=Path("/tmp/deepinfer-omlx-venv/bin/python"))
    parser.add_argument("--omlx-memory-guard", choices=("balanced", "aggressive"), default="balanced",
                        help="One explicit official tier shared by baseline and all DFlash candidates; guard is never disabled")
    parser.add_argument("--omlx-dflash-draft-snapshot", type=Path)
    parser.add_argument("--omlx-dflash-grid", action="store_true", help="Dev-only bounded Q4/group64 block4/5 x adaptive/full candidates")
    parser.add_argument("--omlx-control", choices=("baseline", "dflash", "both"))
    parser.add_argument("--omlx-dflash-block-size", type=int, choices=(4, 5), default=5)
    parser.add_argument("--omlx-dflash-verify-mode", choices=("adaptive", "dflash"), default="adaptive")
    parser.add_argument("--omlx-dflash-resource-report", type=Path)
    parser.add_argument("--dev-selection-report", type=Path)
    parser.add_argument("--dev-raw-report", type=Path, help="Immutable actual dev raw; summary SHA and recomputed selected settings must match")
    parser.add_argument("--dev-source-bridge-note")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.omlx_dflash_grid and not args.dev_tuning_grid:
        parser.error("--omlx-dflash-grid requires --dev-tuning-grid; heldout tuning is forbidden")
    control = args.omlx_control or ("both" if args.omlx_dflash_draft_snapshot else "baseline")
    if args.omlx_dflash_grid and control != "both":
        parser.error("The dev DFlash grid must retain the default baseline via --omlx-control both")
    if (args.omlx_dflash_grid or control != "baseline") and not args.omlx_dflash_draft_snapshot:
        parser.error("DFlash candidates require the exact SHA-verified trained draft snapshot")
    workspace = args.workspace.resolve()
    snapshot = args.snapshot.resolve()
    revision = "16daa4818c54ce5f5436f929d52542eb65bbed9d"
    if snapshot.name != revision or not (snapshot / "config.json").is_file():
        raise ValueError("Pass the pinned, complete Qwen3.5-9B-8bit snapshot")
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    frozen = freeze(workspace, args.root.resolve() / "source", manifest)
    source_root = Path(frozen["root"])
    model_id = "Qwen3.5-9B-8bit"
    model_dir = args.root / "models"
    model_dir.mkdir(exist_ok=True)
    link = model_dir / model_id
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != snapshot:
            raise ValueError("Existing model alias must be an exact symlink to the pinned snapshot; preserve unrelated directories")
    else:
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
    python = str(args.python.absolute())
    experiments = source_root / "benchmarks/experiments"
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
                "--prefill-step-size",
                str(args.own_prefill_step_size),
            ],
            "prefill_step_size": args.own_prefill_step_size,
            "message_checkpoints": args.message_checkpoints,
            "round_diagnostics": False,
            "round_diagnostics_configuration": "Adapter default False; --round-diagnostics intentionally absent. Heavy replay/profile is a separate pass.",
            "additional_draft": None,
            "request_options": {"mode": "baseline"},
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
                str(args.native_prefill_step_size),
                "--chat-template-args",
                '{"enable_thinking":false}',
            ],
            "prefill_step_size": args.native_prefill_step_size,
            "additional_draft": None,
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
            "prefill_step_size": 2048,
            "prefill_selection": "Official scheduler default with its memory guard; no audited CLI chunk override",
            "additional_draft": None,
        },
    ]
    own = engines.pop(0)
    if args.message_checkpoints:
        own["command"].append("--message-checkpoints")
    lock = json.loads((source_root / "src/deepinfer_mac/models.lock.json").read_text())
    draft_repo = lock["profiles"]["qwen3.5-9b-8bit"]["draft"]
    draft_model = next(model for model in lock["models"] if model["repo"] == draft_repo)
    draft = {"repo": draft_repo, "revision": draft_model["revision"],
                                "weight_bytes": sum(item["size_bytes"] for item in draft_model["weight_files"]),
                                "weight_files": draft_model["weight_files"],
                                "note": "Additional draft weights/cache/state cost is included in DS elapsed/resource measurements"}
    variants = []
    for variant in dict.fromkeys(args.variants):
        item = json.loads(json.dumps(own))
        if variant != "baseline":
            adapter = "original" if variant == "ds-original" else "lagged_mean"
            item.update(id=f"deepinfer-{variant}", experimental=True, hybrid_cost_adapter=adapter,
                        additional_draft=draft, request_options={"mode": "adaptive"})
            item["command"] += ["--experimental-hybrid-speculation", "--adaptive-scheduler", "ds",
                                "--hybrid-cost-adapter", adapter]
            size = getattr(args, variant.replace("-", "_") + "_prefill_step_size") or args.own_prefill_step_size
            item["prefill_step_size"] = size
            item["command"][item["command"].index("--prefill-step-size") + 1] = str(size)
        variants.append(item)
    engines = variants + engines
    runtime_files = [{"path": str(settings_path.resolve()), "sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
                      "kind": "official_omlx_model_settings"}]
    dflash_resource = None
    if args.omlx_dflash_draft_snapshot and control != "baseline":
        draft = pinned_draft(args.omlx_dflash_draft_snapshot, experiments / "omlx_dflash_9b.manifest.json")
        runtime_files.extend({"path": item["path"], "sha256": item["sha256"], "kind": "pinned_trained_draft"}
                             for item in draft["verified_files"])
        dflash_resource = resource_admission(args.omlx_dflash_resource_report, args.omlx_memory_guard)
        if args.omlx_dflash_resource_report:
            runtime_files.append({"path": str(args.omlx_dflash_resource_report.resolve()),
                                  "sha256": dflash_resource["sha256"], "kind": "actual_dflash_resource_admission"})
        original = next(engine for engine in engines if engine["id"] == "omlx")
        choices = [(block, mode) for block in (4, 5) for mode in ("adaptive", "dflash")] if args.omlx_dflash_grid else [
            (args.omlx_dflash_block_size, args.omlx_dflash_verify_mode)]
        for block, mode in choices:
            candidate = json.loads(json.dumps(original))
            candidate.update(id=f"omlx-dflash-q4-b{block}-{mode}", tuning_group="omlx",
                             additional_draft=draft, dflash_execution_required={"draft_snapshot": draft["snapshot"],
                             "model_id": model_id, "block_size": block, "verify_mode": mode},
                             resource_admission=dflash_resource, diagnostics_enabled=False)
            candidate_base = args.root / candidate["id"]
            candidate_base.mkdir(exist_ok=True)
            candidate_settings = json.loads(json.dumps(settings))
            candidate_settings["models"][model_id].update(dflash_settings(draft, block, mode))
            candidate_path = candidate_base / "model_settings.json"
            if candidate_path.exists() and json.loads(candidate_path.read_text()) != candidate_settings:
                raise ValueError("Existing DFlash candidate settings differ; use a fresh root")
            candidate_path.write_text(json.dumps(candidate_settings, indent=2) + "\n")
            runtime_files.append({"path": str(candidate_path.resolve()), "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                                  "kind": "official_omlx_dflash_settings"})
            candidate["command"][candidate["command"].index("--base-path") + 1] = str(candidate_base)
            engines.append(candidate)
        if control == "dflash":
            engines = [engine for engine in engines if engine["id"] != "omlx"]
    if args.dev_tuning_grid:
        candidates = []
        for engine in engines:
            if engine["id"].startswith("omlx"):
                candidates.append(engine)
                continue
            for size in (128, 256, 512, 1024):
                candidate = json.loads(json.dumps(engine))
                candidate.update(id=f"{engine['id']}-chunk{size}", tuning_parent=engine["id"], prefill_step_size=size)
                candidate["command"][candidate["command"].index("--prefill-step-size") + 1] = str(size)
                candidates.append(candidate)
        engines = candidates
    for engine in engines:
        engine["tuning_group"] = engine.get("tuning_group", engine.get("tuning_parent", engine["id"]))
        engine["framework"] = "omlx" if engine["id"].startswith("omlx") else "mlx-lm" if engine["id"].startswith("native-mlx-lm") else "deepinfer"
        if engine["framework"] == "omlx":
            engine["command"] += ["--memory-guard", args.omlx_memory_guard]
            engine["memory_guard"] = args.omlx_memory_guard
        engine["environment"] = engine.get("environment", {}) | {"PYTHONPATH": str(source_root / "src")}
        engine["cwd"] = str(source_root)
    interpreter_environments = {
        executable: environment(executable, experiments / "service_source_freeze.py")
        for executable in sorted({engine["command"][0] for engine in engines})
    }
    approved_selection = resource_selection(json.loads(manifest.read_bytes()), args.resource_approved_lengths,
                                            args.total_context_cap) if args.resource_approved_lengths else None
    selection_binding = None
    if args.dev_selection_report:
        if not args.dev_raw_report:
            parser.error("--dev-selection-report requires --dev-raw-report for actual evidence validation")
        summary = json.loads(args.dev_selection_report.read_text())
        raw_sha256 = hashlib.sha256(args.dev_raw_report.read_bytes()).hexdigest()
        if summary.get("raw_report_sha256") != raw_sha256:
            raise ValueError("Actual dev raw bytes differ from summary SHA256")
        selection_binding = bind_dev_selection(json.loads(args.dev_raw_report.read_text()), summary, engines,
                                              json.loads(manifest.read_text()), frozen["sha256"], interpreter_environments, args.dev_source_bridge_note,
                                              {"resource_selection": approved_selection, "total_context_cap_tokens": args.total_context_cap,
                                               "resource_policy": RESOURCE_POLICY})
        selection_binding.update(path=str(args.dev_selection_report.resolve()),
                                 sha256=hashlib.sha256(args.dev_selection_report.read_bytes()).hexdigest(), raw_sha256=raw_sha256)
        selection_binding["raw_path"] = str(args.dev_raw_report.resolve())
        runtime_files.append({"path": str(args.dev_selection_report.resolve()),
                              "sha256": hashlib.sha256(args.dev_selection_report.read_bytes()).hexdigest(),
                              "kind": "dev_selected_candidates"})
        runtime_files.append({"path": str(args.dev_raw_report.resolve()), "sha256": raw_sha256, "kind": "actual_dev_raw"})
    result = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "manifest_canonical_sha256": digest(json.loads(manifest.read_bytes())),
        "frozen_source": frozen,
        "frozen_manifest": str(source_root / frozen["manifest_relative_path"]),
        "model_alias": {"path": str(link.absolute()),
                        "snapshot": str(snapshot), "weights_revision": revision},
        "status": "prepared_only_not_loaded_or_measured",
        "total_context_cap_tokens": args.total_context_cap,
        "resource_probe_required_above_cap": True,
        "dev_tuning_grid": args.dev_tuning_grid,
        "omlx_dflash_grid": args.omlx_dflash_grid,
        "omlx_dflash_resource_admission": dflash_resource,
        "resource_policy": RESOURCE_POLICY,
        "runtime_files": runtime_files,
        "dev_selection_report": selection_binding,
        "tuning_protocol": {
            "split": "dev only; never choose parameters using heldout outputs",
            "candidates": [128, 256, 512, 1024],
            "selection": "Per engine/mode independently: reject protocol/strict-quality regression first; minimize aggregate paired dev task total_ms, TTFT as tie-break; retain all results.",
            "omlx": "Keep official 2048 default and memory guard; no supported CLI chunk override was verified.",
            "heldout": "After actual dev measurements, regenerate non-grid configuration with independent chosen sizes and freeze sources before heldout.",
        },
        "engines": engines,
        "notes": [
            "oMLX hot-cache-only is its official RAM-cache mode; SSD is a separate experiment.",
            "Pure text official model_type_override=llm avoids charging vision weights in one engine.",
            "This is a cache control, not a DS mechanism or competitive superiority result.",
            "Validate template token IDs and actual byte units before formal heldout measurement.",
            "DS variants use the same target but additionally load the explicitly pinned draft; total memory is not equal.",
            "Own/native chunk sizes are independent dev choices; oMLX keeps its audited default/guard, not a forced handicap.",
        ],
    }
    if args.resource_approved_lengths:
        result["resource_selection"] = approved_selection
    result["project_source_at_preparation"] = {"source_tree_sha256": frozen["sha256"], "files": frozen["files"],
                                                "scope": "complete frozen runtime, harness, helper, manifest and lock extraction"}
    names = ("prepare_service_compare.py", "rotate_service_compare.py", "http_service_compare.py",
             "service_resource_selection.py", "summarize_service_compare.py", "deepinfer_service_server.py",
             "native_lru_server.py", "omlx_service_server.py")
    result["harness_sources_at_preparation"] = {
        name: hashlib.sha256((experiments / name).read_bytes()).hexdigest() for name in names
    }
    result["interpreter_environments"] = interpreter_environments
    destination = args.root / "configuration.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
