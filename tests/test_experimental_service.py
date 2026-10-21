"""Experimental service capability views must never promote public metadata."""

import importlib.util
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from deepinfer_mac.models import list_models
from deepinfer_mac.server.app import create_app


class Engine:
    options: ClassVar[list] = []

    def __init__(self, **kwargs):
        type(self).options.append(kwargs)

    def generate(self, request, emit=None, cancel=None):
        if emit:
            emit({"type": "token", "text": "ok"})
        return {"text": "ok", "token_ids": [10], "prompt_tokens": 2,
                "finish_reason": "stop", "metrics": {"mode": request.mode, "prefill_step_size": request.prefill_step_size}}


BODY = {"model": "qwen3.5-9b-8bit", "messages": [{"role": "user", "content": "fixture"}]}


def test_default_remains_baseline_only_and_experimental_view_is_local():
    before = list_models()
    with TestClient(create_app(Engine)) as client:
        assert client.post("/v1/chat/completions", json={**BODY, "mode": "adaptive"}).status_code == 422
    with TestClient(create_app(Engine, experimental_hybrid_speculation=True)) as client:
        model = next(m for m in client.get("/v1/models").json()["data"] if m["id"] == BODY["model"])
        assert model["experimental"] is True
        assert model["verified_supported_modes"] == ["baseline"]
        assert model["experimental_modes"] == ["fixed_speculative", "adaptive"]
        assert model["capabilities"]["speculative"] is True
        assert model["capabilities"]["speculative_verified"] is False
        assert model["capabilities"]["speculative_status"] == "experimental"
        assert client.get("/health").json()["experimental_hybrid_speculation"] is True
        for mode in ("adaptive", "fixed_speculative"):
            assert client.post("/v1/chat/completions", json={**BODY, "mode": mode}).status_code == 200
        stream = client.post("/v1/chat/completions", json={**BODY, "mode": "adaptive", "stream": True})
        assert stream.status_code == 200 and stream.text.endswith("data: [DONE]\n\n")
        # Existing speculative sampling constraints are unchanged.
        for settings in ({"temperature": .7}, {"presence_penalty": 1}, {"top_p": .8}, {"top_k": 20}):
            assert client.post("/v1/chat/completions", json={**BODY, "mode": "adaptive", **settings}).status_code == 422
        assert client.post("/v1/chat/completions", json={**BODY, "model": "qwen3-14b", "mode": "adaptive"}).status_code == 422
    assert list_models() == before
    with TestClient(create_app(Engine)) as client:
        model = next(m for m in client.get("/v1/models").json()["data"] if m["id"] == BODY["model"])
        assert model["supported_modes"] == ["baseline"] and "experimental_modes" not in model


def test_app_default_factory_requires_and_forwards_its_own_engine_flag(monkeypatch):
    Engine.options = []
    monkeypatch.setattr("deepinfer_mac.engine.InferenceEngine", Engine)
    with TestClient(create_app(experimental_hybrid_speculation=True)) as client:
        assert client.post("/v1/chat/completions", json={**BODY, "mode": "adaptive"}).status_code == 200
    assert Engine.options == [{"experimental_hybrid_speculation": True}]


def test_metadata_without_a_real_draft_is_not_promoted(monkeypatch):
    from deepinfer_mac.models import get_profile

    monkeypatch.setitem(get_profile(BODY["model"]), "draft", None)
    with TestClient(create_app(Engine, experimental_hybrid_speculation=True)) as client:
        assert client.post("/v1/chat/completions", json={**BODY, "mode": "adaptive"}).status_code == 422


def service_launcher():
    path = Path(__file__).parents[1] / "benchmarks/experiments/deepinfer_service_server.py"
    spec = importlib.util.spec_from_file_location("experimental_service_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("adapter", ["original", "lagged_mean"])
def test_benchmark_launcher_forwards_all_explicit_experiment_flags(monkeypatch, adapter):
    Engine.options = []
    monkeypatch.setattr("deepinfer_mac.engine.InferenceEngine", Engine)
    launcher = service_launcher()
    options = launcher.parser().parse_args([
        "--experimental-hybrid-speculation", "--hybrid-cost-adapter", adapter,
        "--adaptive-scheduler", "ds", "--cache-budget-bytes", "1234", "--prefill-step-size", "128",
    ])
    with TestClient(launcher.build_app(options)) as client:
        response = client.post("/v1/chat/completions", json={**BODY, "mode": "adaptive"})
        assert response.status_code == 200
        assert response.json()["deepinfer"]["metrics"]["prefill_step_size"] == 128
    assert Engine.options[0]["experimental_hybrid_speculation"] is True
    assert Engine.options[0]["hybrid_cost_adapter"] == adapter
    assert Engine.options[0]["adaptive_scheduler"] == "ds"
    assert Engine.options[0]["cache_budget_bytes"] == 1234
    assert Engine.options[0]["round_diagnostics"] is False


def test_benchmark_round_diagnostics_requires_explicit_profile_flag():
    options = service_launcher().parser()
    assert options.parse_args([]).round_diagnostics is False
    assert options.parse_args(["--round-diagnostics"]).round_diagnostics is True
