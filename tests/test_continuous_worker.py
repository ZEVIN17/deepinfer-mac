"""Portable state/control fixtures, never model or performance evidence."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from deepinfer_mac.continuous import AdmissionBudget, ContinuousConfig, Prepared, batch_eligible
from deepinfer_mac.server.app import create_app
from deepinfer_mac.server.continuous_worker import ContinuousInferenceWorker
from deepinfer_mac.server.worker import Job, QueueFull
from deepinfer_mac.state import GenerationRequest


class Detokenizer:
    def __init__(self):
        self.segment = ""

    def add_token(self, token):
        self.segment += chr(token)

    @property
    def last_segment(self):
        value, self.segment = self.segment, ""
        return value

    def finalize(self):
        pass


class Engine:
    def __init__(self):
        self.calls = []
        self.closed = False

    def generate(self, request, emit=None, cancel=None):
        self.calls.append(("exclusive", threading.get_ident(), request.temperature, request.mode))
        if request.prompt == "exclusive-block":
            while not cancel.wait(0.005):
                pass
            raise InterruptedError
        emit({"type": "token", "text": "exclusive"})
        return {"text": "exclusive", "token_ids": [1], "prompt_tokens": 1,
                "finish_reason": "length", "metrics": {}}

    def close(self):
        self.closed = True
        self.calls.append(("engine-close", threading.get_ident()))


class Session:
    """Independent control fixture: one prefill row/chunk and decode first."""
    def __init__(self, engine, profile, config, hook=None, capacity=10**9):
        self.engine, self.profile, self.config, self.hook = engine, profile, config, hook
        self.pending = {}
        self.uid = 0
        self.budget = AdmissionBudget(config, 10, capacity)
        self.load_started = self.loaded_at = time.perf_counter()
        self.model_warm = False
        self.closed = False
        self.tick_number = 0
        engine.calls.append(("open", threading.get_ident(), profile))

    def prepare(self, request):
        if request.prompt == "prepare-fail":
            raise ValueError("preparation failure")
        reserve = 10**10 if request.prompt == "too-large" else 20
        return Prepared([ord(c) for c in request.prompt], reserve, self.profile,
                        Detokenizer(), {"id": self.profile}, {0})

    def resources(self):
        return {"actual_bytes": 10}

    def insert(self, prepared, request):
        self.uid += 1
        self.pending[self.uid] = {"prepared": prepared, "left": len(prepared.tokens),
                                  "outputs": 0, "max": request.max_tokens,
                                  "fail": request.prompt == "tick-fail"}
        if request.prompt == "insert-fail":
            raise RuntimeError("partial insertion")
        self.engine.calls.append(("insert", threading.get_ident(), self.uid, self.profile))
        return self.uid

    def tick(self):
        self.tick_number += 1
        self.engine.calls.append(("tick", threading.get_ident(), self.tick_number, self.profile))
        if self.hook:
            self.hook(self)
        if any(item["fail"] for item in self.pending.values()):
            raise RuntimeError("tick failure")
        prefills, generated = [], []
        for uid, item in list(self.pending.items()):
            if item["left"] == 0:
                item["outputs"] += 1
                token = item["prepared"].tokens[0]
                finished = "length" if item["outputs"] >= min(3, item["max"]) else None
                generated.append(SimpleNamespace(uid=uid, token=token, finish_reason=finished,
                                                  prompt_cache=[object()] if finished else None,
                                                  all_tokens=item["prepared"].tokens + [token] * item["outputs"] if finished else None))
                if finished:
                    del self.pending[uid]
        for uid, item in self.pending.items():
            if item["left"]:
                item["left"] = max(0, item["left"] - 2)
                total = len(item["prepared"].tokens)
                prefills.append(SimpleNamespace(uid=uid, progress=(total - item["left"], total),
                                                end_of_prompt=item["left"] == 0))
                break
        return prefills, generated

    def remove(self, uid):
        self.engine.calls.append(("remove", threading.get_ident(), uid))
        del self.pending[uid]

    def close(self):
        self.engine.calls.append(("session-close", threading.get_ident(), len(self.pending)))
        self.pending.clear()
        self.closed = True


def config(**kwargs):
    return ContinuousConfig(temporary_reserve_bytes=0, **kwargs)


def wait(job):
    assert job.done.wait(3), job.snapshot()
    return job


def until(predicate):
    deadline = time.perf_counter() + 3
    while not predicate() and time.perf_counter() < deadline:
        time.sleep(0.002)
    assert predicate()


def worker_with_gate(**options):
    entered, release = threading.Event(), threading.Event()
    sessions = []
    def factory(engine, profile, cfg):
        def hook(session):
            if not entered.is_set():
                entered.set()
                assert release.wait(3)
        session = Session(engine, profile, cfg, hook=hook)
        sessions.append(session)
        return session
    worker = ContinuousInferenceWorker(Engine, config=config(**options), session_factory=factory, queue_size=8)
    return worker, entered, release, sessions


def test_lifecycle_rejects_invalid_reentry():
    job = Job("stable", object(), False)
    for state in ("prefill", "decode", "draining", "finished"):
        job.transition(state)
    with pytest.raises(RuntimeError, match="Invalid request transition"):
        job.transition("prefill")


def test_actual_tick_contract_interleaves_output_with_long_prefill_and_no_crosstalk():
    worker, entered, release, sessions = worker_with_gate()
    try:
        short = worker.submit(GenerationRequest(prompt="A", max_tokens=3))
        assert entered.wait(2)
        long = worker.submit(GenerationRequest(prompt="BBBBBBBBBBBB", max_tokens=3))
        release.set()
        assert wait(short).result["text"] == "AAA"
        assert wait(long).result["text"] == "BBB"
        short_ticks = {event["tick"] for event in short.events if event["type"] == "token"}
        long_prefill_ticks = {event["tick"] for event in long.events if event["type"] == "prefill" and not event["end_of_prompt"]}
        assert short_ticks & long_prefill_ticks
        assert len(sessions) == 1
        assert [event["stage"] for event in short.events if event["type"] == "lifecycle"] == ["prefill", "decode", "draining", "finished"]
        assert short.result["metrics"]["ttft_ms"] >= short.result["metrics"]["queue_ms"] >= 0
        assert short.result["metrics"]["total_ms"] >= short.result["metrics"]["ttft_ms"]
        assert short.result["metrics"]["completed_cache_tokens"] == 4
        assert short.result["metrics"]["completed_cache_supplied"]
        assert short.result["token_ids"] == [65] * 3
    finally:
        release.set()
        engine = worker.engine
        worker.close()
    assert engine.closed
    assert {entry[1] for entry in engine.calls} == {worker.thread.ident}


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_cancel_active_at_prefill_or_decode_and_slot_generation_reuse(phase):
    entered, release = threading.Event(), threading.Event()
    sessions = []
    def factory(engine, profile, cfg):
        def hook(session):
            state = next(iter(session.pending.values()), None)
            matched = phase == "prefill" or state and state["left"] == 0
            if matched and not entered.is_set():
                entered.set()
                assert release.wait(3)
        session = Session(engine, profile, cfg, hook=hook)
        sessions.append(session)
        return session
    worker = ContinuousInferenceWorker(Engine, config=config(max_active=1), session_factory=factory)
    try:
        cancelled = worker.submit(GenerationRequest(prompt="long prompt", max_tokens=3))
        assert entered.wait(2)
        worker.cancel_job(cancelled.id)
        release.set()
        assert wait(cancelled).lifecycle == "cancelled"
        count = cancelled.event_count
        recovered = worker.submit(GenerationRequest(prompt="R", max_tokens=1))
        assert wait(recovered).result["text"] == "R"
        assert recovered.slot == cancelled.slot == 0
        assert recovered.slot_generation > cancelled.slot_generation
        assert cancelled.event_count == count
        assert any(call[0] == "remove" for call in worker.engine.calls)
        assert all(not session.budget.reservations for session in sessions)
    finally:
        release.set()
        worker.close()


def test_bounded_pending_cancel_frees_capacity_and_shutdown_drains():
    worker, entered, release, sessions = worker_with_gate(max_active=1)
    worker.queue_size = 1
    active = worker.submit(GenerationRequest(prompt="A"))
    assert entered.wait(2)
    pending = worker.submit(GenerationRequest(prompt="B"))
    with pytest.raises(QueueFull):
        worker.submit(GenerationRequest(prompt="C"))
    worker.cancel_job(pending.id)
    assert wait(pending).lifecycle == "cancelled"
    pending2 = worker.submit(GenerationRequest(prompt="D"))
    closer = threading.Thread(target=worker.close)
    closer.start()
    until(lambda: worker.stopping.is_set())
    release.set()
    closer.join(3)
    assert not closer.is_alive()
    assert wait(active).status == wait(pending2).status == "cancelled"
    assert all(session.closed and not session.pending and not session.budget.reservations for session in sessions)
    with pytest.raises(QueueFull):
        worker.submit(GenerationRequest(prompt="closed"))


@pytest.mark.parametrize("prompt,code", [("too-large", 503), ("prepare-fail", 422), ("tick-fail", 500), ("insert-fail", 500)])
def test_resource_and_backend_failures_release_capacity_and_recover(prompt, code):
    worker = ContinuousInferenceWorker(Engine, config=config(), session_factory=Session)
    try:
        bad = worker.submit(GenerationRequest(prompt=prompt, max_tokens=2))
        assert wait(bad).status == "failed"
        assert bad.error_status == code
        good = worker.submit(GenerationRequest(prompt="G", max_tokens=2))
        assert wait(good).result["text"] == "GG"
        assert not worker.active_requests
    finally:
        worker.close()


def test_budget_counts_all_uid_context_and_reserves_without_rss_doublecount():
    cfg = config(max_active=2, max_context_tokens=10, memory_budget_bytes=200)
    budget = AdmissionBudget(cfg, 100, 1000)
    prepared = Prepared([1, 2], 30, "id", None, {})
    assert budget.check("a", prepared, 2, actual_bytes=100) is None
    budget.reserve("a", prepared, 2)
    # Model+all reserves=160, allocator+new=170; alternate lower bound, not sum.
    assert budget.check("b", prepared, 2, actual_bytes=140) is None
    assert budget.check("b", prepared, 2, actual_bytes=190) == "memory_capacity"
    budget.reserve("b", prepared, 2)
    assert budget.check("c", prepared, 1) == "active_capacity"
    budget.release("a")
    assert budget.check("c", prepared, 7) == "context_capacity"
    budget.release("b")
    assert budget.check("c", prepared, 2, available_bytes=20) == "available_memory"
    assert budget.check("c", prepared, 2, available_bytes=20, actual_cache_bytes=30) is None


def test_heterogeneous_context_padding_is_reserved_before_short_row_is_inserted():
    budget = AdmissionBudget(config(max_active=4, memory_budget_bytes=650), 100, 1000)
    long = Prepared([1] * 10, 200, "long", None, {}, kv_bytes=100)
    short = Prepared([1], 20, "short", None, {}, kv_bytes=10)
    budget.reserve("long", long, 1)
    assert budget.total_reserved(short) == 400  # 220 independent + 180 padded/extraction
    budget.reserve("short1", short, 1)
    assert budget.check("short2", short, 1) == "memory_capacity"  # padded reservation 600 + model 100


def test_incompatible_models_and_non_greedy_requests_are_exclusive_and_fair():
    worker, entered, release, _sessions = worker_with_gate(max_active=2, group_quota=1, aging_seconds=0.001)
    try:
        first = worker.submit(GenerationRequest(prompt="A", profile="first", max_tokens=3))
        assert entered.wait(2)
        foreign = worker.submit(GenerationRequest(prompt="B", profile="second", max_tokens=1, priority="background"))
        sampled = worker.submit(GenerationRequest(prompt="S", profile="first", temperature=0.5, max_tokens=1))
        spec = worker.submit(GenerationRequest(prompt="T", profile="first", mode="adaptive", max_tokens=1))
        later = worker.submit(GenerationRequest(prompt="C", profile="first", max_tokens=1))
        time.sleep(0.003)  # deterministic aging threshold, not benchmark timing
        release.set()
        for job in (first, foreign, sampled, spec, later):
            assert wait(job).status == "completed"
        assert foreign.result["profile"] == "second"
        assert sampled.result["metrics"]["execution_backend"] == "exclusive_serial"
        assert spec.result["metrics"]["execution_backend"] == "exclusive_serial"
        opens = [entry[2] for entry in worker.engine.calls if entry[0] == "open"]
        assert opens == ["first", "second", "first"]
        assert [job.result["text"] for job in (first, foreign, later)] == ["AAA", "B", "C"]
        assert [entry[2:] for entry in worker.engine.calls if entry[0] == "exclusive"] == [(0.5, "baseline"), (0.0, "adaptive")]
    finally:
        release.set()
        worker.close()


def test_sampling_guard_and_switches_are_independent():
    assert batch_eligible(GenerationRequest(prompt="a"))
    for options in ({"temperature": 0.5}, {"top_p": 0.8}, {"presence_penalty": 0.2}, {"mode": "adaptive"}):
        assert not batch_eligible(GenerationRequest(prompt="a", **options))
    assert not batch_eligible(GenerationRequest(prompt="a"), compare=True)
    assert config(max_active=1, policy="upstream").policy == "upstream"
    for kwargs in ({"max_active": 0}, {"tick_token_budget": 4}, {"policy": "ds"}, {"target_tick_ms": float("nan")}):
        with pytest.raises(ValueError):
            config(**kwargs)


def test_future_kv_reservation_defers_second_uid_then_recovers_without_rejection():
    entered, release = threading.Event(), threading.Event()
    sessions = []
    def factory(engine, profile, cfg):
        def hook(session):
            if not entered.is_set():
                entered.set()
                assert release.wait(3)
        session = Session(engine, profile, cfg, hook=hook, capacity=40)
        sessions.append(session)
        return session
    worker = ContinuousInferenceWorker(Engine, config=config(max_active=2), session_factory=factory)
    try:
        first = worker.submit(GenerationRequest(prompt="A", max_tokens=2))
        assert entered.wait(2)
        second = worker.submit(GenerationRequest(prompt="B", max_tokens=2))
        release.set()
        assert wait(first).status == wait(second).status == "completed"
        first_complete = next(event["result"]["metrics"]["total_ms"] for event in first.events if event["type"] == "complete")
        second_admitted = next(event["queue_ms"] for event in second.events if event.get("stage") == "prefill" and event["type"] == "status")
        assert second_admitted >= 0 and first_complete >= 0
        assert all(not session.budget.reservations for session in sessions)
        assert first.slot == second.slot and second.slot_generation > first.slot_generation
    finally:
        release.set()
        worker.close()


def test_all_upstream_uids_count_toward_external_active_cap():
    observed = []
    entered, release = threading.Event(), threading.Event()
    class CountedSession(Session):
        def insert(self, prepared, request):
            uid = super().insert(prepared, request)
            observed.append(len(self.pending))
            return uid
    def factory(engine, profile, cfg):
        def hook(session):
            if not entered.is_set():
                entered.set()
                assert release.wait(3)
        return CountedSession(engine, profile, cfg, hook=hook)
    worker = ContinuousInferenceWorker(Engine, config=config(max_active=2), session_factory=factory, queue_size=8)
    try:
        jobs = [worker.submit(GenerationRequest(prompt="AAAAAAAA", max_tokens=3))]
        assert entered.wait(2)
        jobs.extend(worker.submit(GenerationRequest(prompt="BBBBBBBB", max_tokens=3)) for _ in range(8))
        with pytest.raises(QueueFull):
            worker.submit(GenerationRequest(prompt="overflow"))
        release.set()
        assert all(wait(job).status == "completed" for job in jobs)
        assert max(observed) == 2
    finally:
        release.set()
        worker.close()


def test_api_optin_and_resource_status_and_stream_contract():
    with TestClient(create_app(Engine, execution_backend="continuous", continuous_config=config(), batch_session_factory=Session)) as client:
        health = client.get("/health").json()
        assert health["execution_backend"] == "continuous" and health["continuous_policy"] == "interactive"
        body = {"messages": [{"role": "user", "content": "A"}], "max_tokens": 1, "priority": "background"}
        # Fixture tokenizer deliberately uses prompt; API supplies messages.
        original = Session.prepare
        def prepare_api(session, request):
            request.prompt = request.messages[0]["content"]
            return original(session, request)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Session, "prepare", prepare_api)
            response = client.post("/v1/chat/completions", json=body)
            assert response.status_code == 200
            assert response.json()["deepinfer"]["metrics"]["execution_backend"] == "continuous"
            stream = client.post("/v1/chat/completions", json={**body, "stream": True})
            assert '"content": "A"' in stream.text and stream.text.endswith("data: [DONE]\n\n")
            bad = client.post("/v1/chat/completions", json={**body, "messages": [{"role": "user", "content": "too-large"}]})
            assert bad.status_code == 503


def test_engine_constructor_failure_does_not_lose_dequeued_request():
    attempts = []
    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("constructor failure")
        return Engine()
    worker = ContinuousInferenceWorker(factory, config=config(), session_factory=Session)
    try:
        assert wait(worker.submit(GenerationRequest(prompt="a"))).status == "failed"
        assert wait(worker.submit(GenerationRequest(prompt="b"))).status == "completed"
    finally:
        worker.close()


def test_shutdown_remove_and_close_failures_still_terminate_every_job():
    entered, release = threading.Event(), threading.Event()
    sessions = []
    class BrokenSession(Session):
        def remove(self, uid):
            raise RuntimeError("injected remove failure")

        def close(self):
            super().close()
            raise RuntimeError("injected close failure")

    def factory(engine, profile, cfg):
        def hook(session):
            if not entered.is_set():
                entered.set()
                assert release.wait(3)
        session = BrokenSession(engine, profile, cfg, hook=hook)
        sessions.append(session)
        return session
    worker = ContinuousInferenceWorker(Engine, config=config(max_active=1), session_factory=factory)
    active = worker.submit(GenerationRequest(prompt="AAA"))
    assert entered.wait(2)
    pending = worker.submit(GenerationRequest(prompt="P"))
    errors = []
    def close():
        try:
            worker.close()
        except RuntimeError as exc:
            errors.append(str(exc))
    closer = threading.Thread(target=close)
    closer.start()
    until(lambda: worker.stopping.is_set())
    release.set()
    closer.join(3)
    assert not closer.is_alive() and not worker.thread.is_alive()
    assert wait(active).status == wait(pending).status == "cancelled"
    assert not worker.active_requests and worker.free_slots == [0]
    assert all(not session.budget.reservations for session in sessions)
    assert errors and "remove failure" in errors[0] and "close failure" in errors[0]


def test_tick_failure_plus_close_failure_recreates_engine_and_recovers_queue():
    sessions = []
    def factory(engine, profile, cfg):
        session = Session(engine, profile, cfg)
        if not sessions:
            def bad_close():
                session.closed = True
                raise RuntimeError("bad close")
            session.close = bad_close
        sessions.append(session)
        return session
    worker = ContinuousInferenceWorker(Engine, config=config(), session_factory=factory)
    bad = worker.submit(GenerationRequest(prompt="tick-fail"))
    assert wait(bad).status == "failed"
    good = worker.submit(GenerationRequest(prompt="G", max_tokens=1))
    assert wait(good).status == "completed"
    assert sessions[0].engine is not sessions[1].engine
    assert sessions[0].engine.closed
    with pytest.raises(RuntimeError, match="bad close"):
        worker.close()


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
def test_disconnect_at_first_role_chunk_cancels_before_event_iterator_starts(spec_version):
    entered, release = threading.Event(), threading.Event()
    def factory(engine, profile, cfg):
        def hook(session):
            entered.set()
            assert release.wait(3)
        session = Session(engine, profile, cfg, hook=hook)
        original = session.prepare
        def prepare(request):
            request.prompt = "A long prompt to remain in prefill"
            return original(request)
        session.prepare = prepare
        return session
    app = create_app(Engine, execution_backend="continuous", continuous_config=config(), batch_session_factory=factory)

    async def exercise():
        async with app.router.lifespan_context(app):
            received = asyncio.Queue()
            payload = json.dumps({"messages": [{"role": "user", "content": "A"}], "stream": True}).encode()
            await received.put({"type": "http.request", "body": payload, "more_body": False})
            async def receive():
                return await received.get()
            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    if spec_version == "2.4":
                        raise OSError("client closed socket")
                    await received.put({"type": "http.disconnect"})
                    await asyncio.sleep(0)
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version},
                     "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
                     "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "",
                     "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 1)}
            try:
                try:
                    await asyncio.wait_for(app(scope, receive, send), 3)
                except ClientDisconnect:
                    assert spec_version == "2.4"
                job = next(iter(app.state.worker.jobs.values()))
                assert job.cancel.is_set()
            finally:
                release.set()
            assert await asyncio.to_thread(job.done.wait, 3)
            assert job.status == "cancelled"
    asyncio.run(exercise())
