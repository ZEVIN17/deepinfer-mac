"""Portable contract/guard checks; tiny fabricated files are not model evidence."""
import json
import struct
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from deepinfer_mac.adapters import CacheAdapter, ModelAdapter, ModelRegistry, ModelSpec, inspect_snapshot
from deepinfer_mac.cli import main
from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.server.app import create_app
from deepinfer_mac.state import GenerationRequest


@pytest.fixture
def local_model(tmp_path):
    config = {"model_type": "llama", "vocab_size": 8, "num_hidden_layers": 2,
              "num_attention_heads": 2, "hidden_size": 8, "max_position_embeddings": 2048}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "fixture"}))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}))
    header = json.dumps({"model.embed_tokens.weight": {"dtype": "F32", "shape": [8, 1], "data_offsets": [0, 32]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(32))
    return tmp_path


def test_registry_identity_and_content_change_guard(local_model):
    registry = ModelRegistry([ModelSpec("new-family", str(local_model))])
    metadata = registry.get("new-family")
    assert metadata["context_limit"] == 2048
    assert metadata["vocab_size"] == 8
    assert registry.list()[-1]["target"] == {"local": True, "content_sha256": metadata["target_content_sha256"]}
    engine = InferenceEngine(model_registry=registry)
    engine.paths = {"target": str(local_model)}
    engine._audit_custom_snapshots(metadata, False)
    request = GenerationRequest(prompt="x", profile="new-family")
    engine.tokenizer = SimpleNamespace(chat_template="fixture")
    original = engine._identity(request, False)
    engine._check_snapshot_unchanged()
    weights = local_model / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"\x01")
    with pytest.raises(ValueError, match="files changed"):
        engine._check_snapshot_unchanged()
    with pytest.raises(ValueError, match="changed after registration"):
        engine._audit_custom_snapshots(metadata, False)
    fresh = ModelRegistry([ModelSpec("new-family", str(local_model))])
    fresh_engine = InferenceEngine(model_registry=fresh)
    fresh_engine.paths = {"target": str(local_model)}
    fresh_engine._audit_custom_snapshots(fresh.get("new-family"), False)
    fresh_engine.tokenizer = engine.tokenizer
    assert original != fresh_engine._identity(request, False)


def test_real_file_budget_guard_precedes_any_model_arrays(local_model, monkeypatch):
    registry = ModelRegistry([ModelSpec("local", str(local_model))])
    engine = InferenceEngine(model_registry=registry, memory_limit_bytes=1024**3)
    engine.paths = {"target": str(local_model)}
    with pytest.raises(MemoryError, match="Resolved weights"):
        engine._audit_custom_snapshots(registry.get("local"), False)
    engine.memory_limit_bytes = None
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=1024))
    with pytest.raises(MemoryError, match="insufficient available"):
        engine._audit_custom_snapshots(registry.get("local"), False)


@pytest.mark.parametrize("mutation", ["missing_shard", "unsafe_shard", "bad_vocab", "bad_tensor_vocab", "remote_code", "bad_geometry"])
def test_incompatible_files_fail_before_weight_load(local_model, mutation):
    config = json.loads((local_model / "config.json").read_text())
    if mutation in ("missing_shard", "unsafe_shard"):
        name = "absent.safetensors" if mutation == "missing_shard" else "../escape.safetensors"
        (local_model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": name}}))
    elif mutation == "bad_vocab":
        (local_model / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 9}}}))
    elif mutation == "bad_tensor_vocab":
        config["vocab_size"] = 7
    elif mutation == "remote_code":
        config["model_file"] = "remote.py"
    else:
        config.pop("hidden_size")
    (local_model / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        ModelRegistry([ModelSpec("invalid", str(local_model))])


def test_exact_revision_and_profile_preservation(local_model):
    with pytest.raises(ValueError, match="exact"):
        ModelSpec("unpinned", "org/repo", "main")
    with pytest.raises(ValueError, match="omit revision"):
        ModelSpec("local", str(local_model), "a" * 40)
    registry = ModelRegistry([ModelSpec("remote", "org/repo", "a" * 40)])
    assert registry.get("remote")["target"]["revision"] == "a" * 40
    for name in ("qwen3-4b", "qwen3-14b", "qwen3.5-9b-8bit"):
        from deepinfer_mac.models import get_profile
        assert registry.get(name) == get_profile(name)
    with pytest.raises(ValueError, match="reserved"):
        registry.register(ModelSpec("qwen3-4b", str(local_model)))


def test_same_python_cli_api_model_identity_and_explicit_fallback(local_model, capsys):
    settings = local_model / "registration.json"
    settings.write_text(json.dumps({"id": "new-family", "source": str(local_model)}))
    # Keep registration config outside model content so writing it does not
    # change the model we subsequently compare across frontends.
    settings.rename(local_model.parent / (local_model.name + "-registration.json"))
    settings = local_model.parent / (local_model.name + "-registration.json")
    registry = ModelRegistry([ModelSpec.from_json(settings)])
    assert main(["models", "--model-config", str(settings), "--json"]) == 0
    cli = next(m for m in json.loads(capsys.readouterr().out) if m["id"] == "new-family")
    requests = []

    class Engine:
        def generate(self, request, emit=None, cancel=None):
            requests.append(request)
            adapter = registry.adapter(request.profile)
            mode, reason = adapter.select_mode(request.mode, fallback=request.fallback_to_baseline)
            return {"text": "fixture", "token_ids": [1], "prompt_tokens": 1, "finish_reason": "length",
                    "metrics": {"effective_mode": mode, "capability_fallback": reason}}

    with TestClient(create_app(Engine, model_registry=registry)) as client:
        api = next(m for m in client.get("/v1/models").json()["data"] if m["id"] == "new-family")
        assert api["target"] == cli["target"] == registry.list()[-1]["target"]
        assert str(local_model) not in json.dumps(api)
        body = {"model": "new-family", "messages": [{"role": "user", "content": "hi"}], "mode": "adaptive"}
        assert client.post("/v1/chat/completions", json=body).status_code == 422
        assert not requests
        reply = client.post("/v1/chat/completions", json={**body, "fallback_to_baseline": True})
        assert reply.status_code == 200
        assert reply.json()["deepinfer"]["metrics"]["effective_mode"] == "baseline"
        assert reply.json()["deepinfer"]["metrics"]["capability_fallback"]
    assert requests[0].profile == "new-family"


def test_capability_and_template_guards_are_architecture_independent():
    adapter = ModelAdapter({"id": "arbitrary", "supported_modes": ["baseline"]})
    assert adapter.select_mode("adaptive", fallback=True)[0] == "baseline"
    with pytest.raises(ValueError, match="use baseline"):
        adapter.select_mode("adaptive")
    config = {"model_type": "arbitrary", "hidden_size": 16, "num_attention_heads": 4, "num_hidden_layers": 3}
    assert adapter.memory_estimate(config, 5) == 5 * 3 * 4 * 4 * 4
    plain = SimpleNamespace(encode=lambda text: [len(text)])
    assert adapter.tokenize(plain, GenerationRequest(prompt="hello")) == [5]
    with pytest.raises(ValueError, match="no chat template"):
        adapter.tokenize(plain, GenerationRequest(messages=[{"role": "system", "content": "x"}]))
    unknown = [SimpleNamespace(is_trimmable=lambda: False)]
    assert CacheAdapter().capabilities(unknown) == {"trim": False, "batch": False, "snapshot_replay": False, "varlen_verify": False}
    with pytest.raises(ValueError, match="not audited"):
        CacheAdapter().snapshot(unknown)


def test_confidence_and_draft_version_change_cache_identity():
    tokenizer = SimpleNamespace(chat_template="template")
    request = GenerationRequest(prompt="hello")
    metadata = {"target": {"repo": "target", "revision": "a"}, "draft": {"repo": "draft", "revision": "b"}}
    first = ModelAdapter(metadata).identity(tokenizer, request, True)
    for extra in ({"draft": {"repo": "draft", "revision": "c"}}, {"confidence_revision": "c"},
                  {"calibration_revision": "c"}, {"quantization": "4bit"}):
        assert ModelAdapter(metadata | extra).identity(tokenizer, request, True) != first
    assert ModelAdapter(metadata).identity(tokenizer, request, False) != first


def test_header_audit_is_metadata_only(local_model):
    audit = inspect_snapshot(local_model)
    assert audit["weight_bytes"] == (local_model / "model.safetensors").stat().st_size
    assert len(audit["content_sha256"]) == 64


def test_indexed_snapshot_rejects_unindexed_weights_loaded_by_mlx(local_model):
    index = local_model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}))
    before = inspect_snapshot(local_model)
    assert before["weight_bytes"] == (local_model / "model.safetensors").stat().st_size
    # This is valid weight content, not an error hidden by a corrupt header.
    extra = local_model / "model-extra.safetensors"
    extra.write_bytes((local_model / "model.safetensors").read_bytes())
    with pytest.raises(ValueError, match="exactly match MLX-LM"):
        inspect_snapshot(local_model)
    with pytest.raises(ValueError, match="exactly match MLX-LM"):
        ModelRegistry([ModelSpec("stale-index", str(local_model))])


def test_index_cannot_select_weights_ignored_by_actual_loader(local_model):
    (local_model / "model.safetensors").rename(local_model / "weights.safetensors")
    (local_model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
    with pytest.raises(ValueError, match="exactly match MLX-LM"):
        inspect_snapshot(local_model)


def test_snapshot_without_index_accounts_for_every_loader_weight(local_model):
    before = inspect_snapshot(local_model)
    extra = local_model / "model-extra.safetensors"
    extra.write_bytes((local_model / "model.safetensors").read_bytes())
    after = inspect_snapshot(local_model)
    assert after["weight_bytes"] == before["weight_bytes"] + extra.stat().st_size
    assert "model-extra.safetensors" in after["files"]
    assert after["content_sha256"] != before["content_sha256"]
