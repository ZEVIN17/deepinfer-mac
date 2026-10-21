"""Portable checks for the resource gate; no Metal import or allocation."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import psutil

ROOT = Path(__file__).resolve().parent.parent
PROBE = runpy.run_path(str(ROOT / "scripts/probe_qwen27.py"))


def test_header_audit_covers_all_weights_and_removes_vision():
    manifest = json.loads((ROOT / "docs/qwen27.manifest.json").read_text())
    assert manifest["text_tensor_bytes"] + manifest["vision_tensor_bytes"] == manifest["all_tensor_bytes"]
    assert sum(s["tensor_count"] for s in manifest["tensor_header_audit"]) == 2180
    assert (
        sum(s["language_model_bytes"] for s in manifest["tensor_header_audit"])
        == manifest["text_tensor_bytes"]
    )
    estimate = PROBE["budget"](manifest, 4480, 256, 512 * 1024**2, 2 * 1024**3, 1024**3)
    assert estimate["required_available_bytes"] > manifest["text_tensor_bytes"]
    assert estimate["weight_basis"] == "verified_safetensors_headers_text_only"
    assert manifest["mtp_weight_keys"] == []


def test_unobservable_swap_is_unknown_instead_of_zero(monkeypatch):
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(total=24 * 1024**3, available=8 * 1024**3)
    )

    def unavailable():
        raise OSError("restricted sysctl")

    monkeypatch.setattr(psutil, "swap_memory", unavailable)
    state = PROBE["memory_snapshot"]()
    assert state["swap_available"] is False
    assert state["swap_used_bytes"] is None
    assert "restricted sysctl" in state["swap_error"]
