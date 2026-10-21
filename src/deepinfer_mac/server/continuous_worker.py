"""Bounded request lifecycle and experimental, single-owner batch tick loop."""
from __future__ import annotations

import collections
import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..benchmark.export import jsonable
from ..benchmark.runner import compare_once
from ..continuous import ContinuousConfig, MLXBatchSession, batch_eligible
from .worker import InferenceWorker, Job, QueueFull


class PendingQueue:
    """Owned under worker.lock; same inspection surface as serial queue."""
    def __init__(self):
        self.items = []

    def qsize(self):
        return len(self.items)

    def empty(self):
        return not self.items


class RequestInsertionCancelled(InterruptedError):
    """Backend certifies cancellation left existing batch state unchanged."""


class BatchInsertionError(RuntimeError):
    """Insertion may have mutated upstream state; discard the whole batch."""


@dataclass
class ActiveRequest:
    job: Job
    prepared: object
    uid: int
    started_at: float
    token_ids: list = field(default_factory=list)
    text_parts: list = field(default_factory=list)
    emitted_at: list = field(default_factory=list)
    ticks: int = 0


class ContinuousInferenceWorker(InferenceWorker):
    """Opt-in same-model greedy batching; unsupported jobs run exclusively.

    No device work occurs in submit/cancel. Backend creation, insertion, ticks,
    removal and close all occur on one owning thread. Exclusive work never
    coexists with batch UIDs, including sampled requests with global RNG state.
    """
    def __init__(self, engine_factory, queue_size=4, history_size=64, config=None, session_factory=None):
        if queue_size < 1 or history_size < 1:
            raise ValueError("queue_size and history_size must be positive")
        self.config = config or ContinuousConfig()
        self.session_factory = session_factory or MLXBatchSession
        self.engine_factory, self.engine = engine_factory, None
        self.queue, self.queue_size = PendingQueue(), queue_size
        self.jobs = collections.OrderedDict()
        self.history_size = history_size
        self.lock = threading.RLock()
        self.wakeup = threading.Condition(self.lock)
        self.stopping = threading.Event()
        self.active = None  # backwards compatible health: oldest active job
        self.active_requests = {}
        self.session = None
        self.preparing = None
        self.slot_generations = [0] * self.config.max_active
        self.free_slots = list(range(self.config.max_active))
        self.group_admissions = 0
        self.tick_index = 0
        self.cleanup_errors = collections.deque(maxlen=32)
        self.thread = threading.Thread(target=self._run, name="deepinfer-continuous-worker", daemon=True)
        self.thread.start()

    backend_name = "continuous"

    def decorate_result(self, result):
        return result

    def decorate_serial_result(self, result):
        return result

    def create_session(self, job):
        return self.session_factory(self.engine, job.request.profile, self.config)

    def submit(self, request, compare=False):
        job = Job(uuid.uuid4().hex, request, compare)
        with self.wakeup:
            if self.stopping.is_set():
                raise QueueFull("Server is shutting down")
            if self.queue.qsize() + (self.preparing is not None) >= self.queue_size:
                raise QueueFull("Continuous inference queue is full; retry after capacity is released")
            self.jobs[job.id] = job
            self.queue.items.append(job)
            job.emit({"type": "status", "stage": "queued", "run_id": job.id,
                      "execution_backend": self.backend_name, "eligible_for_batch": batch_eligible(request, compare)})
            self._trim_history()
            self.wakeup.notify_all()
        return job

    def cancel_job(self, identifier):
        with self.wakeup:
            job = self.jobs.get(identifier)
            if job and not job.done.is_set():
                job.cancel.set()
                if job in self.queue.items:
                    self.queue.items.remove(job)
                    self._finish_cancel(job)
                self.wakeup.notify_all()
            return job

    def _finish_cancel(self, job):
        if job.done.is_set():
            return
        job.result = None
        job.status = "cancelled"
        job.transition("cancelled")
        job.emit({"type": "cancelled", "run_id": job.id})
        self._done(job)

    def _fail(self, job, exc):
        if job.done.is_set():
            return
        if job.cancel.is_set():
            self._finish_cancel(job)
            return
        job.result = None
        job.status, job.error = "failed", str(exc)
        job.error_status = 503 if isinstance(exc, MemoryError) else 422 if isinstance(exc, ValueError) else 500
        job.transition("failed")
        job.emit({"type": "error", "message": job.error, "status": job.error_status, "run_id": job.id})
        self._done(job)

    def _done(self, job):
        job.done.set()
        with job.condition:
            job.condition.notify_all()
        with self.lock:
            self._trim_history()

    def _rank(self, job):
        if self.config.policy == "upstream":
            return (0, job.submitted_at)
        aged = time.perf_counter() - job.submitted_at >= self.config.aging_seconds
        return (0 if aged else 1 if getattr(job.request, "priority", "interactive") == "interactive" else 2,
                job.submitted_at)

    def _choose(self):
        return min(self.queue.items, key=self._rank) if self.queue.items else None

    def _compatible(self, job):
        return batch_eligible(job.request, job.compare) and job.request.profile == self.session.profile

    def _admit(self, job):
        """Prepare/admit only on owner thread; caller removed job from queue."""
        try:
            if job.cancel.is_set() or self.stopping.is_set():
                self._finish_cancel(job)
                return True
            prepared = self.session.prepare(job.request)
            reason = self.session.budget.check(job.id, prepared, job.request.max_tokens, **self.session.resources())
            if reason:
                if self.active_requests:
                    with self.lock:
                        self.queue.items.append(job)
                    return False
                raise MemoryError(f"Continuous admission rejected: {reason}; input is never truncated")
            if job.cancel.is_set() or self.stopping.is_set():
                self._finish_cancel(job)
                return True
            slot = self.free_slots.pop(0)
            self.slot_generations[slot] += 1
            job.slot, job.slot_generation = slot, self.slot_generations[slot]
            try:
                uid = (self.session.insert_request(prepared, job.request, job)
                       if hasattr(self.session, "insert_request") else self.session.insert(prepared, job.request))
            except RequestInsertionCancelled:
                self.free_slots.append(slot)
                self._finish_cancel(job)
                return True
            except Exception as exc:
                self.free_slots.append(slot)
                self._fail(job, exc)
                raise BatchInsertionError("Batch insertion failed; batch state discarded") from exc
            self.session.budget.reserve(job.id, prepared, job.request.max_tokens)
            active = ActiveRequest(job, prepared, uid, job.scheduled_at or time.perf_counter())
            self.active_requests[uid] = active
            job.status = "running"
            job.transition("prefill")
            job.emit({"type": "status", "stage": "prefill", "run_id": job.id, "backend_uid": uid,
                      "slot": slot, "slot_generation": job.slot_generation,
                      "queue_ms": (active.started_at - job.submitted_at) * 1000,
                      "prompt_tokens": len(prepared.tokens), "reserve_bytes": prepared.reserve_bytes,
                      "cache_reuse": "cold", "model_identity": prepared.identity})
            self.group_admissions += 1
            return True
        except BatchInsertionError:
            raise
        except Exception as exc:  # noqa: BLE001 — isolate per-request preparation errors
            self._fail(job, exc)
            return True
        finally:
            with self.lock:
                self.preparing = None

    def _release(self, active):
        self.active_requests.pop(active.uid, None)
        self.session.budget.release(active.job.id)
        self.free_slots.append(active.job.slot)
        self.free_slots.sort()

    def _cancel_active(self):
        for active in list(self.active_requests.values()):
            if active.job.cancel.is_set() or self.stopping.is_set():
                try:
                    self.session.remove(active.uid)
                except Exception as exc:
                    self.cleanup_errors.append(f"remove: {exc}")
                    raise
                self._release(active)
                self._finish_cancel(active.job)

    def _complete(self, active, response):
        job = active.job
        job.transition("draining")
        detokenizer = active.prepared.detokenizer
        detokenizer.finalize()
        tail = detokenizer.last_segment
        if tail:
            active.text_parts.append(tail)
            job.emit({"type": "token", "text": tail, "final_segment": True, "run_id": job.id,
                      "slot_generation": job.slot_generation, "tick": self.tick_index})
        end = time.perf_counter()
        emitted = active.emitted_at
        intervals = [(b - a) * 1000 for a, b in itertools.pairwise(emitted)]
        # Only upstream explicitly supplied completion boundaries are recorded;
        # no guessed prompt+generated-minus-one cache key or snapshot reuse.
        consumed = list(response.all_tokens) if response.all_tokens is not None else None
        job.result = {
            "run_id": job.id, "mode": "baseline", "profile": job.request.profile,
            "text": "".join(active.text_parts), "token_ids": active.token_ids,
            "prompt_tokens": len(active.prepared.tokens), "finish_reason": response.finish_reason,
            "model": active.prepared.metadata, "rounds": [],
            "metrics": {
                "execution_backend": self.backend_name, "continuous_policy": self.config.policy,
                "experimental": True, "cache_hit_tokens": 0, "cache_reuse": "cold",
                "use_cache_requested": active.job.request.use_cache, "use_cache_effective": False,
                "model_identity": active.prepared.identity,
                "queue_ms": (active.started_at - job.submitted_at) * 1000,
                "ttft_ms": (emitted[0] - job.submitted_at) * 1000 if emitted else None,
                "total_ms": (end - job.submitted_at) * 1000,
                "service_ms": (end - active.started_at) * 1000,
                "itl_ms": intervals, "generation_tokens": len(active.token_ids),
                "prompt_tokens": len(active.prepared.tokens), "ticks": active.ticks,
                "group_load_ms": (self.session.loaded_at - self.session.load_started) * 1000,
                "model_warm": self.session.model_warm,
                "tick_token_budget": self.config.tick_token_budget,
                "prefill_batch_size": 1, "max_active": self.config.max_active,
                "reserved_request_bytes": active.prepared.reserve_bytes,
                "completed_cache_tokens": len(consumed) if consumed is not None else None,
                "completed_cache_supplied": response.prompt_cache is not None,
                "timing_semantics": "Monotonic submit-to-emit; includes queue/load and whole-tick prefill synchronization",
            },
        }
        if job.cancel.is_set() or self.stopping.is_set():
            self._release(active)
            self._finish_cancel(job)
            return
        job.result = self.decorate_result(job.result)
        job.status = "completed"
        job.transition("finished")
        self._release(active)
        job.emit({"type": "complete", "result": job.result, "run_id": job.id})
        self._done(job)

    def _tick(self):
        self.tick_index += 1
        prompt_responses, generation_responses = self.session.tick()
        # Tick returns only after upstream's prefill synchronization. Actual
        # emission times below include that delay, even though decode ran first.
        for active in self.active_requests.values():
            active.ticks += 1
        for response in prompt_responses:
            active = self.active_requests.get(response.uid)
            if active is None:
                raise RuntimeError("Upstream prefill response has no live UID")
            if active.job.cancel.is_set():
                continue
            if response.end_of_prompt:
                active.job.transition("decode")
            active.job.emit({"type": "prefill", "run_id": active.job.id, "progress": list(response.progress),
                             "end_of_prompt": response.end_of_prompt, "tick": self.tick_index,
                             "slot_generation": active.job.slot_generation})
        for response in generation_responses:
            active = self.active_requests.get(response.uid)
            if active is None:
                raise RuntimeError("Upstream decode response has no live UID")
            job = active.job
            if job.cancel.is_set():
                # A finished response was already removed by upstream. For any
                # other response the normal cancellation pass removes its UID.
                if response.finish_reason is not None:
                    self._release(active)
                    self._finish_cancel(job)
                continue
            job.transition("decode")
            # Preserve the engine's full token-history convention, including
            # the terminal EOS. EOS is never passed to the detokenizer.
            active.token_ids.append(response.token)
            text = ""
            if response.token not in active.prepared.eos_tokens:
                active.prepared.detokenizer.add_token(response.token)
                text = active.prepared.detokenizer.last_segment
                active.text_parts.append(text)
            now = time.perf_counter()
            active.emitted_at.append(now)
            job.emit({"type": "token", "text": text, "token_id": response.token, "run_id": job.id,
                      "is_eos": response.token in active.prepared.eos_tokens,
                      "token_index": len(active.token_ids) - 1, "tick": self.tick_index,
                      "slot_generation": job.slot_generation,
                      "elapsed_ms": (now - job.submitted_at) * 1000})
            if response.finish_reason is not None:
                self._complete(active, response)

    def _exclusive(self, job):
        """Sampled/speculative/comparison jobs retain the serial engine guard."""
        self.active = job
        try:
            if job.cancel.is_set() or self.stopping.is_set():
                self._finish_cancel(job)
                return
            job.status = "running"
            job.transition("prefill")
            job.emit({"type": "status", "stage": "exclusive_serial", "run_id": job.id,
                      "reason": "sampling/speculation/comparison requires exclusive serial execution"})
            start = time.perf_counter()
            if job.compare:
                result = compare_once(self.engine, job.request, emit=job.emit, cancel=job.cancel)
                result["timing_semantics"] = "Interactive load/cache state may differ across modes; not a benchmark"
                result["comparison"]["speedups"] = {}
            else:
                result = jsonable(self.engine.generate(job.request, emit=job.emit, cancel=job.cancel))
            if job.cancel.is_set() or self.stopping.is_set():
                self._finish_cancel(job)
                return
            job.transition("draining")
            result.setdefault("metrics", {}).update(
                execution_backend="exclusive_serial", queue_ms=(start - job.submitted_at) * 1000,
                total_with_queue_ms=(time.perf_counter() - job.submitted_at) * 1000,
            )
            result = self.decorate_serial_result(result)
            job.result, job.status = result, "completed"
            job.transition("finished")
            job.emit({"type": "complete", "result": result, "run_id": job.id})
            self._done(job)
        except InterruptedError:
            self._finish_cancel(job)
        except Exception as exc:  # noqa: BLE001
            self._fail(job, exc)
        finally:
            self.active = None

    def _close_session(self):
        if self.session is not None:
            session = self.session
            self.session = None
            try:
                session.close()
            except Exception as exc:  # noqa: BLE001 — report cleanup failure and rebuild owner safely
                self.cleanup_errors.append(f"session.close: {exc}")
                try:
                    if self.engine is not None and hasattr(self.engine, "close"):
                        self.engine.close()
                except Exception as engine_exc:  # noqa: BLE001
                    self.cleanup_errors.append(f"engine.close: {engine_exc}")
                    self.stopping.set()
                self.engine = None
        self.group_admissions = 0

    def _run(self):
        try:
            while not self.stopping.is_set():
                try:
                    self._cancel_active()
                    with self.wakeup:
                        if not self.active_requests and not self.queue.items:
                            self._close_session()
                            self.active = None
                            self.wakeup.wait(timeout=0.1)
                            continue
                        next_job = self._choose()
                        if not self.active_requests:
                            self._close_session()
                            if self.stopping.is_set():
                                break
                            if next_job is None:
                                continue
                            self.queue.items.remove(next_job)
                            self.preparing = next_job
                            next_job.scheduled_at = time.perf_counter()
                        else:
                            next_job = None
                    if self.engine is None:
                        try:
                            self.engine = self.engine_factory()
                        except Exception as exc:  # noqa: BLE001
                            self.preparing = None
                            self._fail(next_job, exc)
                            continue
                    if next_job is not None:
                        if next_job.cancel.is_set():
                            self.preparing = None
                            self._finish_cancel(next_job)
                            continue
                        if not batch_eligible(next_job.request, next_job.compare):
                            self.preparing = None
                            self._exclusive(next_job)
                            continue
                        try:
                            self.session = self.create_session(next_job)
                        except Exception as exc:  # noqa: BLE001
                            self.preparing = None
                            self._fail(next_job, exc)
                            continue
                        if self.session is None:
                            self.preparing = None
                            self._exclusive(next_job)
                            continue
                        self._admit(next_job)
                    if self.session is None:
                        continue
                    with self.lock:
                        # An incompatible waiter stops new admission once quota
                        # is reached, then finite active requests drain. Aging
                        # lifts background/other-profile work ahead of new chat.
                        foreign = any(not self._compatible(job) for job in self.queue.items)
                        draining = foreign and self.group_admissions >= self.config.group_quota
                        candidates = sorted((job for job in self.queue.items if self._compatible(job)), key=self._rank)
                    for job in candidates:
                        if draining or foreign and self.group_admissions >= self.config.group_quota or len(self.active_requests) >= self.config.max_active:
                            break
                        with self.lock:
                            if job not in self.queue.items:
                                continue
                            self.queue.items.remove(job)
                            self.preparing = job
                            job.scheduled_at = time.perf_counter()
                        if not self._admit(job):
                            break
                    self.active = next((active.job for active in self.active_requests.values()), None)
                    self._cancel_active()
                    if self.active_requests:
                        self._tick()
                except Exception as exc:  # noqa: BLE001 — batch failure fails all affected UIDs, then recovers
                    for active in list(self.active_requests.values()):
                        self._release(active)
                        self._fail(active.job, exc)
                    self._close_session()
                    self.active = None
        finally:
            # Every request reaches a terminal state even if device removal or
            # shutdown itself fails. No waiting SSE client is left stranded.
            for active in list(self.active_requests.values()):
                try:
                    self.session.remove(active.uid)
                except Exception as exc:  # noqa: BLE001
                    self.cleanup_errors.append(f"shutdown.remove: {exc}")
                finally:
                    self._release(active)
                    active.job.cancel.set()
                    self._finish_cancel(active.job)
            with self.lock:
                for job in list(self.queue.items):
                    self.queue.items.remove(job)
                    job.cancel.set()
                    self._finish_cancel(job)
                if self.preparing is not None and not self.preparing.done.is_set():
                    self.preparing.cancel.set()
                    self._finish_cancel(self.preparing)
                    self.preparing = None
            self._close_session()
            if self.engine is not None and hasattr(self.engine, "close"):
                try:
                    self.engine.close()
                except Exception as exc:  # noqa: BLE001
                    self.cleanup_errors.append(f"shutdown.engine.close: {exc}")
            self.engine = None

    def close(self):
        self.stopping.set()
        with self.wakeup:
            for job in list(self.jobs.values()):
                if not job.done.is_set():
                    job.cancel.set()
                    if job in self.queue.items:
                        self.queue.items.remove(job)
                        self._finish_cancel(job)
            self.wakeup.notify_all()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Continuous inference worker did not stop within 30 seconds")
        if self.cleanup_errors:
            raise RuntimeError("Continuous worker cleanup errors: " + "; ".join(self.cleanup_errors))
