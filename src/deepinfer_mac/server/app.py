"""Local API and bundled web client; model calls stay on a single worker thread."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from .schemas import ChatRequest, CompareRequest
from .worker import InferenceWorker, QueueFull


def default_engine_factory():
    from ..engine import InferenceEngine
    return InferenceEngine()


def create_app(engine_factory=None, queue_size=4, token: str | None = None):
    access_token = token or os.getenv("DEEPINFER_MAC_TOKEN")
    bearer = HTTPBearer(auto_error=False)

    async def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):  # noqa: B008 — FastAPI dependency injection
        import secrets
        if access_token and (not credentials or not secrets.compare_digest(credentials.credentials, access_token)):
            raise HTTPException(401, "A valid bearer token is required", headers={"WWW-Authenticate": "Bearer"})

    @asynccontextmanager
    async def lifespan(app):
        app.state.worker = InferenceWorker(engine_factory or default_engine_factory, queue_size=queue_size)
        yield
        await asyncio.to_thread(app.state.worker.close)

    app = FastAPI(title="DeepInfer-Mac", version="0.1.0a1", lifespan=lifespan)
    security = [Depends(authorize)]

    def submit(request, compared=False):
        try:
            return app.state.worker.submit(request, compare=compared)
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
        from ..models import list_models
        return list_models()

    def validate_model(name, mode=None, compared=False):
        model = next((item for item in registered_models() if item["id"] == name), None)
        if model is None:
            raise HTTPException(404, f"Unknown model profile: {name}")
        supported = model["supported_modes"]
        if mode is not None and mode not in supported:
            raise HTTPException(422, f"Profile {name} supports only {supported}; {mode} is not verified")
        if compared and not {"baseline", "fixed_speculative", "adaptive"}.issubset(supported):
            raise HTTPException(422, f"Profile {name} supports baseline/cache only; three-mode comparison is unavailable")

    @app.get("/health")
    async def health():
        worker = app.state.worker
        return {"status": "ok", "model_loaded": worker.engine is not None,
                "active_run": worker.active.id if worker.active else None, "queued": worker.queue.qsize()}

    @app.get("/v1/models", dependencies=security)
    async def models():
        return {"object": "list", "data": [{**model, "object": "model", "owned_by": "community"}
                                           for model in registered_models()]}

    @app.post("/v1/chat/completions", dependencies=security)
    async def chat(body: ChatRequest, request: Request):
        from ..state import GenerationRequest
        validate_model(body.model, mode=body.mode)
        generation = GenerationRequest(messages=[message.model_dump() for message in body.messages],
                                       max_tokens=body.max_tokens, mode=body.mode, profile=body.model)
        job = submit(generation)
        created = int(time.time())
        if not body.stream:
            async for _ in event_iterator(job, request):
                pass
            if job.status == "cancelled":
                raise HTTPException(409, "Generation cancelled")
            if job.error:
                raise HTTPException(500, job.error)
            result = job.result
            return {"id": f"chatcmpl-{job.id}", "object": "chat.completion", "created": created,
                    "model": body.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                                                        "finish_reason": result["finish_reason"]}],
                    "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": len(result["token_ids"]),
                              "total_tokens": result["prompt_tokens"] + len(result["token_ids"])},
                    "deepinfer": {"run_id": job.id, "metrics": result["metrics"]}}
        async def stream():
            base = {"id": f"chatcmpl-{job.id}", "object": "chat.completion.chunk", "created": created, "model": body.model}
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
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Run-ID": job.id, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    web = str(files("deepinfer_mac").joinpath("web"))
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(f"{web}/index.html")

    return app
