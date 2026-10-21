"""Control-flow tests use no MLX and make no performance claims."""
import threading

import pytest
from fastapi.testclient import TestClient

from deepinfer_mac.models import list_models
from deepinfer_mac.server.app import create_app
from deepinfer_mac.server.worker import InferenceWorker, Job, QueueFull
from deepinfer_mac.state import GenerationRequest


class FakeEngine:
    def generate(self, request, emit=None, cancel=None):
        if request.prompt == "fail":
            raise ValueError("intentional model failure")
        if emit:
            emit({"type": "token", "text": "hello"})
        return {"mode": request.mode, "text": "hello", "token_ids": [10, 2], "prompt_tokens": 4,
                "finish_reason": "stop", "metrics": {"total_ms": 1.0}}


def wait_done(job):
    assert job.done.wait(3)
    return job


def test_api_subset_stream_and_error():
    with TestClient(create_app(FakeEngine)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        assert {m["id"] for m in client.get("/v1/models").json()["data"]} == {m["id"] for m in list_models()}
        body = {"messages": [{"role": "user", "content": "hi"}]}
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello"
        assert response.json()["usage"]["total_tokens"] == 6
        stream = client.post("/v1/chat/completions", json={**body, "stream": True})
        assert '"content": "hello"' in stream.text
        assert stream.text.endswith("data: [DONE]\n\n")
        assert '"finish_reason": "stop"' in stream.text
        for extra in ({"temperature": 0.7}, {"stop": "!"}, {"tools": []}, {"mode": "dspark"}):
            assert client.post("/v1/chat/completions", json={**body, **extra}).status_code == 422
        assert client.post("/v1/chat/completions", json={**body, "model": "unknown"}).status_code == 404
        comparison = client.post("/api/compare", json={"prompt": "hi"})
        assert comparison.status_code == 202
        identifier = comparison.json()["id"]
        events = client.get(f"/api/runs/{identifier}/events")
        assert '"type": "complete"' in events.text
        result = client.get(f"/api/runs/{identifier}").json()
        assert result["result"]["comparison"]["token_ids_equal"] is True
        assert result["result"]["comparison"]["speedups"] == {}
        assert "not a benchmark" in result["result"]["timing_semantics"]


def test_authorization():
    with TestClient(create_app(FakeEngine, token="secret")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_worker_single_owner_queue_cancellation_recovery():
    started, release = threading.Event(), threading.Event()
    owners = []

    class BlockingEngine(FakeEngine):
        def __init__(self):
            owners.append(threading.get_ident())

        def generate(self, request, emit=None, cancel=None):
            owners.append(threading.get_ident())
            if request.prompt == "block":
                started.set()
                while not release.wait(0.01):
                    if cancel.is_set():
                        raise InterruptedError
            return super().generate(request, emit, cancel)

        def close(self):
            owners.append(threading.get_ident())

    worker = InferenceWorker(BlockingEngine, queue_size=1, history_size=2)
    try:
        active = worker.submit(GenerationRequest(prompt="block"))
        assert started.wait(2)
        queued = worker.submit(GenerationRequest(prompt="queued"))
        with pytest.raises(QueueFull):
            worker.submit(GenerationRequest(prompt="overflow"))
        worker.cancel_job(queued.id)
        worker.cancel_job(active.id)
        assert wait_done(active).status == "cancelled"
        assert wait_done(queued).status == "cancelled"
        assert sum(e["type"] == "cancelled" for e in queued.events) == 1
        failed = worker.submit(GenerationRequest(prompt="fail"))
        assert wait_done(failed).status == "failed"
        recovered = worker.submit(GenerationRequest(prompt="recover"))
        assert wait_done(recovered).status == "completed"
        assert len(worker.jobs) <= 2
    finally:
        release.set()
        worker.close()
    assert len(set(owners)) == 1
    assert owners[0] != threading.get_ident()
    with pytest.raises(QueueFull):
        worker.submit(GenerationRequest(prompt="after shutdown"))


def test_bounded_event_history_and_replay_gap():
    job = Job("id", object(), False)
    for i in range(10000):
        job.emit({"type": "token", "text": str(i)})
    batch, gap = job.read_events(0)
    assert gap and len(batch) == 2048
    assert batch[-1]["sequence"] == 9999
    assert job.read_events(10000) == ([], False)
    assert job.snapshot()["event_count"] == 10000


def test_shutdown_cancels_active_and_pending():
    started = threading.Event()

    class SlowEngine(FakeEngine):
        def generate(self, request, emit=None, cancel=None):
            started.set()
            while not cancel.wait(0.01):
                pass
            raise InterruptedError

    worker = InferenceWorker(SlowEngine, queue_size=2)
    active = worker.submit(GenerationRequest(prompt="first"))
    assert started.wait(2)
    pending = worker.submit(GenerationRequest(prompt="second"))
    worker.close()
    assert wait_done(active).status == "cancelled"
    assert wait_done(pending).status == "cancelled"
    assert not worker.thread.is_alive()


def test_candidate_modes_rejected_before_submission():
    app = create_app(FakeEngine)
    with TestClient(app) as client:
        registry = {model["id"]: model for model in client.get("/v1/models").json()["data"]}
        candidate = registry["qwen3.8-27b"]
        assert candidate["status"] == "candidate"
        assert candidate["supported_modes"] == ["baseline"]
        assert candidate["capabilities"]["speculative"] is False
        body = {"model": candidate["id"], "messages": [{"role": "user", "content": "hi"}]}
        for mode in ("fixed_speculative", "adaptive"):
            response = client.post("/v1/chat/completions", json={**body, "mode": mode})
            assert response.status_code == 422
        response = client.post("/api/compare", json={"profile": candidate["id"], "prompt": "hi"})
        assert response.status_code == 422
        assert app.state.worker.engine is None
        assert app.state.worker.queue.empty() and not app.state.worker.jobs
        response = client.post("/v1/chat/completions", json={**body, "mode": "baseline"})
        assert response.status_code == 200
