"""One model owner, bounded queue/history, and cooperative cancellation."""
from __future__ import annotations

import collections
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from ..benchmark.export import jsonable
from ..benchmark.runner import compare_once


class QueueFull(Exception):
    pass


@dataclass
class Job:
    id: str
    request: object
    compare: bool
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    result: dict | None = None
    error: str | None = None
    events: collections.deque = field(default_factory=lambda: collections.deque(maxlen=2048))
    event_count: int = 0
    cancel: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)
    done: threading.Event = field(default_factory=threading.Event)

    def emit(self, event):
        with self.condition:
            if self.cancel.is_set() and event.get("type") not in ("cancelled", "error"):
                return
            self.events.append({**event, "sequence": self.event_count})
            self.event_count += 1
            self.condition.notify_all()

    def read_events(self, cursor):
        with self.condition:
            first = self.events[0]["sequence"] if self.events else self.event_count
            gap = cursor < first
            return [event for event in self.events if event["sequence"] >= cursor], gap

    def snapshot(self):
        return {"id": self.id, "status": self.status, "created_at": self.created_at,
                "result": self.result, "error": self.error, "event_count": self.event_count,
                "retained_events": len(self.events)}


class InferenceWorker:
    def __init__(self, engine_factory: Callable, queue_size=4, history_size=64):
        if queue_size < 1 or history_size < 1:
            raise ValueError("queue_size and history_size must be positive")
        self.engine_factory = engine_factory
        self.engine = None
        self.queue = queue.Queue(maxsize=queue_size)
        self.jobs: collections.OrderedDict[str, Job] = collections.OrderedDict()
        self.history_size = history_size
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.active: Job | None = None
        self.thread = threading.Thread(target=self._run, name="deepinfer-metal-worker", daemon=True)
        self.thread.start()

    def _trim_history(self):
        while len(self.jobs) > self.history_size:
            removable = next((key for key, old in self.jobs.items() if old.done.is_set()), None)
            if removable is None:
                break
            del self.jobs[removable]

    def submit(self, request, compare=False):
        job = Job(uuid.uuid4().hex, request, compare)
        with self.lock:
            if self.stopping.is_set():
                raise QueueFull("Server is shutting down")
            self.jobs[job.id] = job
            job.emit({"type": "status", "stage": "queued", "run_id": job.id})
            try:
                self.queue.put_nowait(job)
            except queue.Full:
                del self.jobs[job.id]
                raise QueueFull("Inference queue is full; retry after the active request finishes") from None
            self._trim_history()
        return job

    def get(self, identifier):
        with self.lock:
            return self.jobs.get(identifier)

    def cancel_job(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job and not job.done.is_set():
                job.cancel.set()
                if job.status == "queued":
                    job.status = "cancelled"
                    job.emit({"type": "cancelled"})
                    job.done.set()
            return job

    def _run(self):
        try:
            while not self.stopping.is_set():
                try:
                    job = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self.lock:
                    if job.done.is_set():
                        self.queue.task_done()
                        continue
                    self.active = job
                    job.status = "running"
                try:
                    if job.cancel.is_set() or self.stopping.is_set():
                        raise InterruptedError("Request cancelled")
                    job.emit({"type": "status", "stage": "loading"})
                    if self.engine is None:
                        self.engine = self.engine_factory()
                    start = time.perf_counter()
                    if job.compare:
                        job.result = compare_once(self.engine, job.request, emit=job.emit, cancel=job.cancel)
                        job.result["timing_semantics"] = "Interactive load/cache state may differ across modes; not a benchmark"
                        job.result["comparison"]["speedups"] = {}
                    else:
                        job.result = jsonable(self.engine.generate(job.request, emit=job.emit, cancel=job.cancel))
                    if job.cancel.is_set():
                        raise InterruptedError("Request cancelled")
                    job.status = "completed"
                    job.emit({"type": "complete", "result": job.result,
                              "worker_ms": (time.perf_counter() - start) * 1000})
                except InterruptedError:
                    job.status = "cancelled"
                    job.result = None
                    job.emit({"type": "cancelled"})
                except Exception as exc:  # noqa: BLE001 — isolate arbitrary model/backend failures
                    job.status = "failed"
                    job.error = str(exc)
                    job.emit({"type": "error", "message": job.error})
                finally:
                    job.done.set()
                    with job.condition:
                        job.condition.notify_all()
                    self.active = None
                    self.queue.task_done()
                    with self.lock:
                        self._trim_history()
        finally:
            # Model resources are released on their owning thread.
            if self.engine is not None and hasattr(self.engine, "close"):
                self.engine.close()
            self.engine = None

    def close(self):
        self.stopping.set()
        with self.lock:
            for job in self.jobs.values():
                if not job.done.is_set():
                    job.cancel.set()
                    if job is not self.active:
                        job.status = "cancelled"
                        job.emit({"type": "cancelled"})
                        job.done.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Inference worker did not stop within 30 seconds")
