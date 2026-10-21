"""Local API and bundled web client; model calls stay on a single worker thread."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import partial
from importlib.resources import files

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .schemas import ChatRequest, CompareRequest
from .worker import InferenceWorker, QueueFull


class RunStreamingResponse(StreamingResponse):
    """Always release an unfinished request, including disconnect during send."""
    def __init__(self, content, *, worker, job, **kwargs):
        super().__init__(content, **kwargs)
        self.worker, self.job = worker, job

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self.job.done.is_set():
                self.worker.cancel_job(self.job.id)


def default_engine_factory(experimental_hybrid_speculation=False, model_registry=None, memory_limit_bytes=None):
    from ..engine import InferenceEngine
    options = {"experimental_hybrid_speculation": experimental_hybrid_speculation}
    if model_registry is not None:
        options["model_registry"] = model_registry
    if memory_limit_bytes is not None:
        options["memory_limit_bytes"] = memory_limit_bytes
    return InferenceEngine(**options)


def create_app(
    engine_factory=None, queue_size=4, token: str | None = None,
    experimental_hybrid_speculation=False, model_registry=None,
    execution_backend="serial", continuous_config=None, batch_session_factory=None,
    experimental_joint_verification=False, joint_config=None,
):
    from ..adapters import ModelRegistry
    registry_source = model_registry or ModelRegistry()
    if execution_backend == "joint_experimental":
        raise ValueError("Joint backend requires explicit experimental_joint_verification opt-in")
    if experimental_joint_verification:
        if execution_backend != "serial" or joint_config is None:
            raise ValueError("Joint opt-in requires joint_config and cannot combine with another backend")
        if experimental_hybrid_speculation:
            raise ValueError("Joint service does not support experimental hybrid speculation")
        execution_backend = "joint_experimental"
    elif joint_config is not None:
        raise ValueError("joint_config requires experimental_joint_verification=True")
    if execution_backend not in ("serial", "continuous", "auto", "joint_experimental"):
        raise ValueError("execution_backend must be serial, continuous or auto")
    if execution_backend in ("continuous", "auto", "joint_experimental"):
        from ..continuous import ContinuousConfig
        continuous_config = continuous_config or ContinuousConfig(policy="upstream" if execution_backend == "auto" else "interactive")
    access_token = token or os.getenv("DEEPINFER_MAC_TOKEN")
    bearer = HTTPBearer(auto_error=False)

    async def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):  # noqa: B008 — FastAPI dependency injection
        import secrets
        if access_token and (not credentials or not secrets.compare_digest(credentials.credentials, access_token)):
            raise HTTPException(401, "A valid bearer token is required", headers={"WWW-Authenticate": "Bearer"})

    @asynccontextmanager
    async def lifespan(app):
        factory = engine_factory or partial(
            default_engine_factory, experimental_hybrid_speculation=experimental_hybrid_speculation, model_registry=model_registry,
            memory_limit_bytes=continuous_config.memory_budget_bytes if continuous_config else None,
        )
        if execution_backend in ("continuous", "auto", "joint_experimental"):
            from .continuous_worker import ContinuousInferenceWorker
            if execution_backend == "joint_experimental":
                from .joint_worker import JointInferenceWorker
                app.state.worker = JointInferenceWorker(factory, queue_size=queue_size, config=continuous_config,
                                                        joint_config=joint_config, session_factory=batch_session_factory)
            elif execution_backend == "auto":
                from .runtime_worker import RuntimeInferenceWorker
                app.state.worker = RuntimeInferenceWorker(factory, queue_size=queue_size, config=continuous_config,
                                                         session_factory=batch_session_factory)
            else:
                app.state.worker = ContinuousInferenceWorker(factory, queue_size=queue_size,
                                                            config=continuous_config, session_factory=batch_session_factory)
        else:
            app.state.worker = InferenceWorker(factory, queue_size=queue_size)
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.worker.close)

    app = FastAPI(title="DeepInfer-Mac", version=__version__, lifespan=lifespan)
    security = [Depends(authorize)]

    def submit(request, compared=False):
        try:
            return app.state.worker.submit(request, compare=compared)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except QueueFull as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "2"}) from exc

    def find(identifier):
        job = app.state.worker.get(identifier)
        if job is None:
            raise HTTPException(404, "Run not found (completed runs are kept in bounded memory)")
        return job

    async def event_iterator(job, request):
        cursor = 0
        try:
            while True:
                if await request.is_disconnected():
                    return
                batch, gap = job.read_events(cursor)
                if gap:
                    yield {"type": "history_gap", "message": "Earlier events expired; fetch the run result."}
                for event in batch:
                    cursor = event["sequence"] + 1
                    yield event
                if job.done.is_set() and cursor >= job.event_count:
                    return
                await asyncio.sleep(0.025)
        finally:
            if not job.done.is_set():
                app.state.worker.cancel_job(job.id)

    def registered_models():
        registry = registry_source.list()
        if experimental_joint_verification:
            registry = deepcopy(registry)
            for model in registry:
                if model["id"] == joint_config.profile:
                    model["experimental_joint"] = {"enabled": True, "verified": False,
                        "effective_mode": "joint_experimental", "greedy_only": True,
                        "trained_confidence": False, "depth": joint_config.depth,
                        "strategy": joint_config.strategy, "draft_budget": joint_config.draft_budget}
        if not experimental_hybrid_speculation:
            return registry
        # This explicit experimental service advertises its own local view.
        # Public metadata and the default service remain baseline-only for
        # hybrid candidates. An engine flag is independently required as well.
        registry = deepcopy(registry)
        for model in registry:
            if not (model.get("experimental_speculation") and model.get("hybrid_cache") and model.get("draft")):
                continue
            verified = list(model["supported_modes"])
            modes = ["fixed_speculative", "adaptive"]
            model.update(
                verified_supported_modes=verified,
                supported_modes=list(dict.fromkeys([*verified, *modes])),
                experimental=True, experimental_modes=modes,
            )
            model["capabilities"].update(
                speculative=True, speculative_verified="adaptive" in verified,
                speculative_status="experimental",
            )
        return registry

    def validate_model(name, mode=None, compared=False, fallback=False):
        model = next((item for item in registered_models() if item["id"] == name), None)
        if model is None:
            raise HTTPException(404, f"Unknown model profile: {name}")
        supported = model["supported_modes"]
        if mode is not None and mode not in supported and not fallback:
            raise HTTPException(422, f"Profile {name} supports only {supported}; {mode} is not verified")
        if compared and not {"baseline", "fixed_speculative", "adaptive"}.issubset(supported):
            raise HTTPException(422, f"Profile {name} supports baseline/cache only; three-mode comparison is unavailable")
        return model

    @app.get("/health")
    async def health():
        worker = app.state.worker
        return {"status": "ok", "model_loaded": worker.engine is not None,
                "active_run": worker.active.id if worker.active else None, "queued": worker.queue.qsize(),
                "execution_backend": execution_backend,
                "active_requests": len(getattr(worker, "active_requests", {})) if execution_backend in ("continuous", "auto", "joint_experimental") else int(worker.active is not None),
                "continuous_policy": continuous_config.policy if continuous_config else None,
                "preparing_run": worker.preparing.id if execution_backend in ("continuous", "auto", "joint_experimental") and worker.preparing else None,
                "cleanup_error_count": len(getattr(worker, "cleanup_errors", [])),
                "experimental_hybrid_speculation": experimental_hybrid_speculation}

    @app.get("/v1/models", dependencies=security)
    async def models():
        return {"object": "list", "data": [{**model, "object": "model", "owned_by": "community"}
                                           for model in registered_models()]}

    @app.post("/v1/chat/completions", dependencies=security)
    async def chat(body: ChatRequest, request: Request):
        from ..state import GenerationRequest
        model = validate_model(body.model, mode=body.mode, fallback=body.fallback_to_baseline)
        if body.top_k and model.get("vocab_size") and body.top_k >= model["vocab_size"]:
            raise HTTPException(422, "top_k must be smaller than the model vocabulary")
        generation = GenerationRequest(messages=[message.model_dump() for message in body.messages],
                                       max_tokens=body.max_tokens, mode=body.mode, profile=body.model,
                                       temperature=body.temperature, top_p=body.top_p, top_k=body.top_k,
                                       presence_penalty=body.presence_penalty, seed=body.seed,
                                       enable_thinking=body.enable_thinking, use_cache=body.use_cache,
                                       priority=body.priority,
                                       fallback_to_baseline=body.fallback_to_baseline)
        job = submit(generation)
        created = int(time.time())
        if not body.stream:
            async for _ in event_iterator(job, request):
                pass
            if job.cancel.is_set() or job.status == "cancelled":
                raise HTTPException(409, "Generation cancelled")
            if job.error:
                raise HTTPException(job.error_status, job.error)
            result = job.result
            return {"id": f"chatcmpl-{job.id}", "object": "chat.completion", "created": created,
                    "model": body.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                                                        "finish_reason": result["finish_reason"]}],
                    "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": len(result["token_ids"]),
                              "total_tokens": result["prompt_tokens"] + len(result["token_ids"])},
                    "deepinfer": {"run_id": job.id, "metrics": result["metrics"]}}
        async def stream():
            base = {"id": f"chatcmpl-{job.id}", "object": "chat.completion.chunk", "created": created, "model": body.model}
            try:
                yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
                async for event in event_iterator(job, request):
                    if event["type"] == "token" and event.get("text"):
                        yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"content": event["text"]}, "finish_reason": None}]}, ensure_ascii=False) + "\n\n"
                    elif event["type"] == "error":
                        yield "data: " + json.dumps({"error": {"message": event["message"], "type": "inference_error"}}) + "\n\n"
                if job.status == "completed":
                    yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": job.result["finish_reason"]}]}) + "\n\n"
                if job.status == "cancelled":
                    yield "data: " + json.dumps({"error": {"message": "Generation cancelled", "type": "cancelled"}}) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # Disconnect can occur at the role-only first chunk, before
                # event_iterator was ever entered. Its finally cannot cover it.
                if not job.done.is_set():
                    app.state.worker.cancel_job(job.id)
        return RunStreamingResponse(stream(), worker=app.state.worker, job=job, media_type="text/event-stream", headers={"X-Run-ID": job.id, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/compare", status_code=202, dependencies=security)
    async def compare(body: CompareRequest):
        from ..state import GenerationRequest
        validate_model(body.profile, compared=True)
        job = submit(GenerationRequest(prompt=body.prompt, max_tokens=body.max_tokens, profile=body.profile), compared=True)
        return {"id": job.id, "status": job.status, "events_url": f"/api/runs/{job.id}/events"}

    @app.get("/api/runs/{identifier}", dependencies=security)
    async def run(identifier: str):
        return find(identifier).snapshot()

    @app.delete("/api/runs/{identifier}", dependencies=security)
    async def cancel(identifier: str):
        find(identifier)
        return app.state.worker.cancel_job(identifier).snapshot()

    @app.get("/api/runs/{identifier}/events", dependencies=security)
    async def events(identifier: str, request: Request):
        job = find(identifier)
        async def stream():
            async for event in event_iterator(job, request):
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        return RunStreamingResponse(stream(), worker=app.state.worker, job=job, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    web = str(files("deepinfer_mac").joinpath("web"))
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(f"{web}/index.html")

    return app
