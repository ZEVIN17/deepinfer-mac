"""Portable service contract controls; no model/performance acceptance."""
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from test_continuous_worker import Engine, Session, config, wait

from deepinfer_mac.joint_service import JointServiceConfig
from deepinfer_mac.server.app import create_app
from deepinfer_mac.server.joint_worker import JointInferenceWorker
from deepinfer_mac.state import GenerationRequest


class MessageSession(Session):
    def prepare(self, request):
        return super().prepare(replace(request, prompt=request.prompt or request.messages[-1]["content"]))


def test_explicit_optin_and_config_validation():
    with pytest.raises(ValueError, match="explicit"):
        create_app(execution_backend="joint_experimental")
    with pytest.raises(ValueError, match="requires"):
        create_app(joint_config=JointServiceConfig("qwen3-4b"))
    with pytest.raises(ValueError, match="cannot combine"):
        create_app(execution_backend="continuous", experimental_joint_verification=True,
                   joint_config=JointServiceConfig("qwen3-4b"))
    for depth in (True, 0, 17):
        with pytest.raises(ValueError):
            JointServiceConfig("pair", depth=depth)


@pytest.mark.parametrize("change", [{"profile": "other"}, {"temperature": 0.1}, {"mode": "adaptive"}, {"top_k": 2}])
def test_joint_rejects_unsupported_before_worker(change):
    joint = JointServiceConfig("qwen3-4b")
    with pytest.raises(ValueError):
        joint.validate_request(GenerationRequest(prompt="x", **change))


def test_joint_lifecycle_labels_and_cache_contract():
    worker = JointInferenceWorker(Engine, joint_config=JointServiceConfig("qwen3-4b"),
                                  config=config(), session_factory=Session)
    try:
        job = wait(worker.submit(GenerationRequest(prompt="x", max_tokens=2)))
        assert job.status == "completed"
        assert job.result["mode"] == "joint_experimental"
        metrics = job.result["metrics"]
        assert metrics["requested_mode"] == "baseline"
        assert metrics["effective_mode"] == "joint_experimental"
        assert metrics["use_cache_effective"] is False
        assert metrics["strict_serial_equivalence_validated"] is False
        assert job.slot_generation == 1
    finally:
        worker.close()


def test_api_experimental_metadata_rejection_and_sse():
    app = create_app(engine_factory=Engine, experimental_joint_verification=True,
                     joint_config=JointServiceConfig("qwen3-4b"), continuous_config=config(),
                     batch_session_factory=MessageSession)
    with TestClient(app) as client:
        models = client.get("/v1/models").json()["data"]
        model = next(m for m in models if m["id"] == "qwen3-4b")
        assert model["experimental_joint"]["verified"] is False
        assert client.get("/health").json()["execution_backend"] == "joint_experimental"
        payload = {"model": "qwen3-4b", "messages": [{"role": "user", "content": "x"}]}
        assert client.post("/v1/chat/completions", json={**payload, "temperature": 0.1}).status_code == 422
        result = client.post("/v1/chat/completions", json={**payload, "max_tokens": 2})
        assert result.status_code == 200
        assert result.json()["deepinfer"]["metrics"]["effective_mode"] == "joint_experimental"
        streamed = client.post("/v1/chat/completions", json={**payload, "stream": True, "max_tokens": 2})
        assert '[DONE]' in streamed.text
        assert '"content": "x"' in streamed.text
        assert client.post("/api/compare", json={"profile": "qwen3-4b", "prompt": "x"}).status_code == 422


def test_source_bound_cost_measurements(tmp_path, monkeypatch):
    import json

    from deepinfer_mac import joint_service
    from deepinfer_mac.joint_service import read_cost_curves, runtime_source_hashes

    # Parser-only fixture: CPU CI has no MLX distribution metadata.
    monkeypatch.setattr(joint_service, "version", lambda name: "parser-fixture-version")
    key = {"model_identity": "target", "draft_identity": "draft", "quantization": "q",
           "batch_size": 1, "context_bucket": 10, "execution_strategy": "bucket"}
    # Explicit synthetic parser fixture, not a model benchmark or stored curve.
    payload = {"scope": "complete_cycle", "partition": "development", "depth": 1,
               "source_sha256": runtime_source_hashes(), "mlx_lm_version": "parser-fixture-version",
               "curves": [{"key": key, "measurements": [{"total_q": 1, "full_cycle_ms": 1},
                                                          {"total_q": 2, "full_cycle_ms": 2}]}]}
    path = tmp_path / "parser-control.json"
    path.write_text(json.dumps(payload))
    assert read_cost_curves(path, 1)[0][1] == [(1, 1), (2, 2)]
    payload["mlx_lm_version"] = "other-version"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="current-source"):
        read_cost_curves(path, 1)
    payload["mlx_lm_version"] = "parser-fixture-version"
    payload["source_sha256"] = {}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="current-source"):
        read_cost_curves(path, 1)


class MultiTokenSession(Session):
    """Verified-output protocol fixture; never executes a model."""
    def tick(self):
        from types import SimpleNamespace
        out = []
        for uid, item in list(self.pending.items()):
            for token, reason in [(65, None), (66, "length")]:
                out.append(SimpleNamespace(uid=uid, token=token, finish_reason=reason,
                                           all_tokens=None, prompt_cache=None))
            del self.pending[uid]
        return [], out


def test_verified_multitoken_round_completes_once_and_releases():
    worker = JointInferenceWorker(Engine, joint_config=JointServiceConfig("qwen3-4b"),
                                  config=config(), session_factory=MultiTokenSession)
    try:
        job = wait(worker.submit(GenerationRequest(prompt="x", max_tokens=2)))
        assert job.result["token_ids"] == [65, 66]
        events, _ = job.read_events(0)
        assert sum(e["type"] == "complete" for e in events) == 1
        assert [e["token_id"] for e in events if "token_id" in e] == [65, 66]
        assert len(worker.free_slots) == worker.config.max_active
    finally:
        worker.close()


def test_cancel_between_verified_token_emissions_hides_rest_and_releases():
    import threading

    from test_continuous_worker import until
    gate, proceed = threading.Event(), threading.Event()
    class Gated(MultiTokenSession):
        def tick(self):
            gate.set()
            assert proceed.wait(2)
            return super().tick()
    worker = JointInferenceWorker(Engine, joint_config=JointServiceConfig("qwen3-4b"),
                                  config=config(), session_factory=Gated)
    try:
        job = worker.submit(GenerationRequest(prompt="x", max_tokens=2))
        assert gate.wait(2)
        emit = job.emit
        def cancel_after_first(event):
            emit(event)
            if event.get("token_id") == 65:
                worker.cancel_job(job.id)
        job.emit = cancel_after_first
        proceed.set()
        wait(job)
        assert job.status == "cancelled" and job.result is None
        events, _ = job.read_events(0)
        assert [e["token_id"] for e in events if "token_id" in e] == [65]
        until(lambda: not worker.active_requests)
        assert len(worker.free_slots) == worker.config.max_active
    finally:
        proceed.set()
        worker.close()


def test_cancel_newcomer_prefill_preserves_existing_request():
    import threading

    from deepinfer_mac.server.continuous_worker import RequestInsertionCancelled
    entered, proceed = threading.Event(), threading.Event()
    class CancelInsert(MultiTokenSession):
        def insert_request(self, prepared, request, job):
            if request.prompt == "cancel-newcomer":
                job.cancel.set()
                raise RequestInsertionCancelled("local prefill discarded")
            return self.insert(prepared, request)

        def prepare(self, request):
            if request.prompt == "first":
                entered.set()
                assert proceed.wait(2)
            return super().prepare(request)
    worker = JointInferenceWorker(Engine, joint_config=JointServiceConfig("qwen3-4b"),
                                  config=config(), session_factory=CancelInsert)
    try:
        first = worker.submit(GenerationRequest(prompt="first", max_tokens=2))
        assert entered.wait(2)
        newcomer = worker.submit(GenerationRequest(prompt="cancel-newcomer", max_tokens=2))
        proceed.set()
        assert wait(first).status == "completed"
        assert wait(newcomer).status == "cancelled"
        assert first.result["token_ids"] == [65, 66]
        assert len(worker.free_slots) == worker.config.max_active
    finally:
        proceed.set()
        worker.close()


def test_prepare_accounts_both_models_and_rejects_draft_context():
    from types import SimpleNamespace

    from test_continuous_worker import Detokenizer

    from deepinfer_mac.joint_service import MLXJointBatchSession
    class ResetDetokenizer(Detokenizer):
        def reset(self):
            self.segment = ""
    calls = []
    class Adapter:
        def memory_estimate(self, cfg, context):
            calls.append((cfg["tag"], context))
            return cfg["kv"]
    session = object.__new__(MLXJointBatchSession)
    session.profile = "qwen3-4b"
    session.joint = JointServiceConfig(session.profile)
    session.config = config(prefill_chunk=8)
    session.metadata = {"context_limit": 2048}
    session.configs = [{"tag": "target", "kv": 10, "vocab_size": 100, "max_position_embeddings": 2048},
                       {"tag": "draft", "kv": 20, "vocab_size": 100, "max_position_embeddings": 16}]
    session.adapter = Adapter()
    session.engine = SimpleNamespace(_check_snapshot_unchanged=lambda: None, _tokens=lambda request: [1, 2],
                                     _identity=lambda request, spec: "pair",
                                     tokenizer=SimpleNamespace(detokenizer=ResetDetokenizer(), eos_token_ids=[0]),
                                     model_registry=SimpleNamespace(public_metadata=lambda x: x))
    prepared = session.prepare(GenerationRequest(prompt="x", max_tokens=2))
    assert calls == [("target", 520), ("draft", 520)]
    assert prepared.reserve_bytes == 8 * 30 + 2 * 8 * 100 * 4
    assert prepared.kv_bytes == 4 * 30
    with pytest.raises(ValueError, match="context limit"):
        session.prepare(GenerationRequest(prompt="x", max_tokens=16))


def test_public_result_does_not_disclose_foreign_round_tokens():
    import json
    import threading
    entered, proceed = threading.Event(), threading.Event()
    class PrivateRound(MultiTokenSession):
        def prepare(self, request):
            if request.prompt == "first":
                entered.set()
                assert proceed.wait(2)
            return super().prepare(request)

        def tick(self):
            self.last_report = {"outputs": [[100001], [100002]], "outcomes": [[True], [False]],
                                "seqlens_q": [1, 2], "requested_q": 3, "full_cycle_ms": 1}
            return super().tick()
    worker = JointInferenceWorker(Engine, joint_config=JointServiceConfig("qwen3-4b"),
                                  config=config(), session_factory=PrivateRound)
    try:
        first = worker.submit(GenerationRequest(prompt="first", max_tokens=2))
        assert entered.wait(2)
        second = worker.submit(GenerationRequest(prompt="second", max_tokens=2))
        proceed.set()
        for job in (first, second):
            result = wait(job).result
            events, _ = job.read_events(0)
            text = json.dumps({"result": result, "events": events})
            assert "100001" not in text and "100002" not in text
            assert result["metrics"]["last_joint_round"] == {"requested_q": 3, "full_cycle_ms": 1}
    finally:
        proceed.set()
        worker.close()
