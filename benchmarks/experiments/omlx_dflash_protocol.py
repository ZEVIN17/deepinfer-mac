"""Pinned official DFlash candidates and fail-closed external log evidence.

Performance runs use the ordinary oMLX adapter, with no diagnostic monkeypatches.
These checks happen outside request timing and cannot establish speedup alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

DRAFT_REPO = "z-lab/Qwen3.5-9B-DFlash"
DRAFT_REVISION = "5fc3b3d474760f18c516db87d84c37edbfd3ede6"
TARGET_LAYERS = [1, 5, 9, 13, 17, 21, 25, 29]


def resource_admission(path, memory_guard="balanced"):
    if path is None:
        return {"verified": False, "reason": "Actual4204-input+256-output DFlash resource probe is pending;2K compatibility or target-only4K is insufficient"}
    path = Path(path)
    report = json.loads(path.read_text())
    required_settings = {"dflash_enabled": True, "dflash_draft_quant_enabled": True,
                         "dflash_draft_quant_weight_bits": 4, "dflash_draft_quant_group_size": 64,
                         "dflash_block_size": 5, "dflash_verify_mode": "dflash", "dflash_draft_window_size": None}
    def actual_input(record):
        import struct

        return (record.get("case", {}).get("prompt_tokens") == 4204
                and record.get("request", {}).get("max_tokens", 0) >= 256
                and any(event.get("event") == "request_prompt" and event.get("prompt_tokens") == 4204
                        and len(event.get("token_ids", [])) == 4204
                        and event.get("token_ids_sha256_le_u32") == hashlib.sha256(struct.pack("<4204I", *event["token_ids"])).hexdigest()
                        for event in record.get("audit_events", [])))

    passed = (report.get("status") == "actual_dev_probe_completed"
              and report.get("target_revision") == "16daa4818c54ce5f5436f929d52542eb65bbed9d"
              and report.get("draft_revision") == DRAFT_REVISION
              and any(result.get("engine", {}).get("id") == "q4-b5-dflash"
                      and result.get("engine", {}).get("memory_guard") == memory_guard
                      and result.get("error") is None
                      and result.get("draft_load_soft_fallback_observed") is False
                      and result.get("official_drafter_loaded_log_observed") is True
                      and all(key in result.get("engine", {}).get("settings", {}) and result["engine"]["settings"][key] == value for key, value in required_settings.items())
                      and result.get("dflash_execution_proven") is True
                      and result.get("all_strict_cases_passed") is True
                      and result.get("system_swap_delta_bytes", 1) <= 0
                      and any(actual_input(record)
                              and record.get("case", {}).get("max_tokens", 0) >= 256
                              and record.get("quality_pass") is True for record in result.get("records", []))
                      and all(record.get("system_after", {}).get("available_bytes", 0) >= 1024**3 for record in result.get("records", []))
                      for result in report.get("results", [])))
    return {"verified": passed, "path": str(path.resolve()), "memory_guard": memory_guard,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reason": "Require actual4204+256 strict Q4/group64/block5/full trained-draft execution with no system swap increase. This admits shared load/window/max-proposal geometry, not all mode-specific transients; every dev candidate's resource samples remain independently checked"}


def pinned_draft(snapshot, manifest_path):
    snapshot = Path(snapshot).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["repo"] != DRAFT_REPO or manifest["revision"] != DRAFT_REVISION or snapshot.name != DRAFT_REVISION:
        raise ValueError("Use the exact audited trained Qwen3.5-9B DFlash checkpoint")
    actual = []
    for expected in manifest["files"]:
        path = snapshot / expected["path"]
        if not path.is_file() or path.stat().st_size != expected["size_bytes"]:
            raise ValueError("Incomplete or differently sized pinned DFlash file")
        with path.open("rb") as handle:
            sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if sha256 != expected["sha256"]:
            raise ValueError("Pinned DFlash checkpoint SHA256 mismatch")
        actual.append({"path": str(path), "sha256": sha256, "size_bytes": path.stat().st_size})
    config = json.loads((snapshot / "config.json").read_text())
    if (config.get("hidden_size"), config.get("num_target_layers"), config.get("vocab_size")) != (4096, 32, 248320):
        raise ValueError("Pinned DFlash target geometry differs")
    if config.get("dflash_config", {}).get("target_layer_ids") != TARGET_LAYERS:
        raise ValueError("Pinned DFlash capture layers differ")
    return {"repo": DRAFT_REPO, "revision": DRAFT_REVISION, "license": manifest["license"],
            "snapshot": str(snapshot), "verified_files": actual,
            "weight_bytes": sum(item["size_bytes"] for item in actual if item["path"].endswith(".safetensors")),
            "note": "Trained block-diffusion draft; original BF16 bytes remain intact. Official load-time4bit/group64 quantization and all draft context/state costs belong to measured task/resource totals."}


def settings(draft, block, mode):
    if block not in (4, 5) or mode not in ("adaptive", "dflash"):
        raise ValueError("Bounded quantized MLX dev candidates are block4/5 and adaptive/full")
    return {"dflash_enabled": True, "dflash_draft_model": draft["snapshot"],
            "dflash_draft_quant_enabled": True, "dflash_draft_quant_weight_bits": 4,
            "dflash_draft_quant_group_size": 64, "dflash_draft_window_size": None,
            "dflash_block_size": block, "dflash_verify_mode": mode}


def execution_evidence(text, expected):
    """Parse actual official logs; config/HTTP200 alone never passes this gate."""
    errors = []
    loaded = re.findall(r"DFlash drafter loaded: path=(.*?) kind=(\S+) block=(\d+) adaptive=(True|False) target_layers=(\[[^\n]+?\])", text)
    attached = re.findall(r"DFlash drafter attached to engine: (.*?) \(kind=(\S+), block=(\d+), layers=(\[[^\n]+?\])\)", text)
    quantized = re.findall(r"DFlash drafter quantized: (\d+) linear layers at (\d+) bits \(group (\d+)\)", text)
    verified = re.findall(r"MTP\[([^\]]+)\] finish=(\S+) tokens=(\d+) cycles=(\d+) tok/cycle=\S+ accept=(\d+)/(\d+)", text)
    if "toggle ignored" in text or "DFlash drafter load raised" in text:
        errors.append("Official DFlash load soft-fallback observed")
    if len(loaded) != 1:
        errors.append("Require exactly one official successful DFlash load")
    else:
        path, kind, block, adaptive, layers = loaded[0]
        if (Path(path).resolve() != Path(expected["draft_snapshot"]).resolve() or kind != "dflash"
                or int(block) != expected["block_size"] or (adaptive == "True") != (expected["verify_mode"] == "adaptive")
                or json.loads(layers) != TARGET_LAYERS):
            errors.append("Loaded DFlash pin/block/mode/capture layers differ from frozen candidate")
    if len(attached) != 1:
        errors.append("Require exactly one official DFlash engine attachment")
    else:
        model, kind, block, layers = attached[0]
        if model != expected["model_id"] or kind != "dflash" or int(block) != expected["block_size"] or json.loads(layers) != TARGET_LAYERS:
            errors.append("Attached engine model/block/layers differ from frozen candidate")
    if quantized != [("43", "4", "64")]:
        errors.append("Official43-linear4bit/group64 quantization not observed exactly once")
    stats = [{"uid": uid, "finish_reason": finish, "emitted_tokens": int(tokens),
              "verify_cycles": int(cycles), "accepted_drafts": int(accepted),
              "verified_draft_positions": int(drafted)}
             for uid, finish, tokens, cycles, accepted, drafted in verified]
    if not any(item["verify_cycles"] > 0 and item["verified_draft_positions"] > 0 for item in stats):
        errors.append("No completed actual draft/verify statistics; settings alone are insufficient")
    if any(item["accepted_drafts"] > item["verified_draft_positions"] for item in stats):
        errors.append("Impossible upstream accepted/drafted counters")
    return {"required": True, "valid": not errors, "errors": errors,
            "successful_load_count": len(loaded), "attachment_count": len(attached),
            "official_quantization_records": [list(map(int, row)) for row in quantized],
            "completed_sequence_stats": stats,
            "scope": "Official external log observations across completed warmup/scored requests, outside request timing. Adaptive may legitimately park individual cycles or requests. No diagnostic hooks were enabled."}
