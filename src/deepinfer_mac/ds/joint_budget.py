"""Explicit confidence and dynamic-UID budget contracts (no device import at import time).

Rank/tie semantics: DeepInfer@9bd9cf2 schedule_mtp_varlen.py, lines 26–81.
MTPVarlen remains the separately oracle-tested CPU cost selector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol

from .mtp_varlen import MTPState, MTPVarlen, f32


@dataclass(frozen=True)
class RequestKey:
    uid: str
    generation: int

    def __post_init__(self):
        if not isinstance(self.uid, str) or not self.uid or type(self.generation) is not int or self.generation < 0:
            raise ValueError("Nonempty UID and nonnegative generation required")


@dataclass(frozen=True)
class Confidence:
    values: tuple[float, ...]
    kind: Literal["conditional", "cumulative"]
    model_identity: str
    calibration_identity: str
    source: Literal["empirical", "trained_head", "mechanism_control"]

    def __post_init__(self):
        if self.kind not in ("conditional", "cumulative"):
            raise ValueError("Unknown probability semantics")
        if self.source not in ("empirical", "trained_head", "mechanism_control"):
            raise ValueError("Unknown confidence source")
        if any(not isinstance(v, str) or not v for v in (self.model_identity, self.calibration_identity)):
            raise ValueError("Model and calibration identity required")
        if any(isinstance(x, bool) or not isinstance(x, Real)
               or not math.isfinite(x) or not 0 <= x <= 1 for x in self.values):
            raise ValueError("Probabilities must be finite in [0,1]")
        object.__setattr__(self, "values", tuple(self.values))
        if self.kind == "cumulative" and any(a < b for a, b in zip(self.values, self.values[1:])):
            raise ValueError("Cumulative survival must be nonincreasing")

    def cumulative(self):
        if self.kind == "cumulative":
            return tuple(map(f32, self.values))
        result, product = [], 1.0
        for value in self.values:
            product = f32(product * f32(value))
            result.append(product)
        return tuple(result)


class ConfidenceProvider(Protocol):
    def predict(self) -> Confidence: ...


class EmpiricalConfidence:
    """Conditional observations: first rejection observed; subsequent suffix censored."""
    def __init__(self, depth, model_identity, calibration_identity):
        if type(depth) is not int or depth < 0:
            raise ValueError("Negative depth")
        self.counts = [[0, 0] for _ in range(depth)]
        self.model_identity = model_identity
        self.calibration_identity = calibration_identity
        self.brier_sums = [0.0] * depth

    def observe(self, outcomes):
        if len(outcomes) != len(self.counts):
            raise ValueError("Observation depth mismatch")
        ended = False
        updates = []
        for value in outcomes:
            if value is not None and type(value) is not bool:
                raise ValueError("Outcomes must be bool or censored None")
            if ended and value is not None:
                raise ValueError("Suffix after rejection/censor must remain censored")
            updates.append(value)
            ended |= value is not True
        for position, (count, value) in enumerate(zip(self.counts, updates)):
            if value is not None:
                predicted = (count[0] + 1) / (count[1] + 2)
                self.brier_sums[position] += (predicted - int(value)) ** 2
                count[0] += int(value)
                count[1] += 1

    def predict(self):
        return Confidence(tuple((a + 1) / (n + 2) for a, n in self.counts), "conditional",
                          self.model_identity, self.calibration_identity, "empirical")


    def diagnostics(self):
        return {"kind": "conditional", "model_identity": self.model_identity,
                "calibration_identity": self.calibration_identity,
                "coverage": [n for _, n in self.counts],
                "accepted": [a for a, _ in self.counts],
                "prequential_brier": [s / n if n else None
                                       for s, (_, n) in zip(self.brier_sums, self.counts)],
                "scope": "online diagnostics; not held-out calibration acceptance"}


def remap_keys(previous, current):
    if len(set(previous)) != len(previous) or len(set(current)) != len(current):
        raise ValueError("Duplicate request/generation")
    lookup = {key: i for i, key in enumerate(previous)}
    return tuple(lookup.get(key, -1) for key in current)


def allocate_cpu(confidences, remap, total_q, depth):
    rows = [c.cumulative() for c in confidences]
    _validate(rows, remap, total_q, depth)
    scores = [rows[i] if i >= 0 else (0.0,) * depth for i in remap]
    ranked = sorted(((p, r, j) for r, row in enumerate(scores) for j, p in enumerate(row)),
                    key=lambda x: (-x[0], x[1], x[2]))
    lengths = [1] * len(remap)
    for _, row, _ in ranked[:total_q - len(remap)]:
        lengths[row] += 1
    return tuple(lengths)


def _validate(rows, remap, total_q, depth):
    if type(depth) is not int or depth < 0 or any(len(row) != depth for row in rows):
        raise ValueError("Uniform nonnegative confidence depth required")
    if any(type(i) is not int or i < -1 or i >= len(rows) for i in remap):
        raise ValueError("Invalid remap")
    if type(total_q) is not int or not len(remap) <= total_q <= len(remap) * (depth + 1):
        raise ValueError("Budget must include one golden token per request")


def allocate_mlx(confidences, remap, total_q, depth):
    """Actual MLX all-pairs rank; lazy import and explicit host-validated contract.

    No custom kernel, claimed GPU oracle, or asynchronous host-free API. Scores
    are converted once; output is an MLX int32 array for downstream consumers.
    """
    rows = [c.cumulative() for c in confidences]
    _validate(rows, remap, total_q, depth)
    import mlx.core as mx
    if not remap or not depth:
        return mx.ones((len(remap),), dtype=mx.int32)
    # An extra zero row is an explicit fresh sentinel, including empty history.
    device_rows = [mx.cumprod(mx.array(c.values, dtype=mx.float32))
                   if c.kind == "conditional" else mx.array(c.values, dtype=mx.float32)
                   for c in confidences]
    values = mx.stack([*device_rows, mx.zeros((depth,), dtype=mx.float32)])
    indices = mx.array([i if i >= 0 else len(rows) for i in remap], dtype=mx.int32)
    scores = values[indices].reshape(-1)
    flat = mx.arange(scores.size)
    rank = mx.sum((scores[None, :] > scores[:, None]) |
                  ((scores[None, :] == scores[:, None]) & (flat[None, :] < flat[:, None])), axis=1)
    return 1 + mx.sum((rank < total_q - len(remap)).reshape(len(remap), depth), axis=1)


@dataclass(frozen=True)
class CostKey:
    model_identity: str
    draft_identity: str
    quantization: str
    batch_size: int
    context_bucket: int
    execution_strategy: str
    calibration_identity: str = "uncalibrated"

    def __post_init__(self):
        names = (self.model_identity, self.draft_identity, self.quantization,
                 self.execution_strategy, self.calibration_identity)
        if any(not isinstance(v, str) or not v for v in names):
            raise ValueError("Complete string cost identity required")
        if type(self.batch_size) is not int or self.batch_size < 0:
            raise ValueError("Batch size must be a nonnegative integer")
        if type(self.context_bucket) is not int or self.context_bucket < 0:
            raise ValueError("Context bucket must be a nonnegative integer")


class JointBudget:
    """Synchronous two-generation history: N-2 total choice, N-1 row allocation.

    History is committed only for successful rounds; cancellation/slot reuse is
    keyed by (uid,generation). Curves are whole-cycle measurements per CostKey,
    never multiplied single-request costs. No curve means full available work.
    """
    def __init__(self, depth, model_identity):
        if type(depth) is not int or depth < 0:
            raise ValueError("Negative depth")
        self.depth, self.model_identity = depth, model_identity
        self.previous, self.latest = {}, {}
        self.schedulers = {}
        self.calibration_identity = None

    def set_curve(self, key, curve):
        if key.model_identity != self.model_identity or key.batch_size < 1:
            raise ValueError("Cost identity/batch mismatch")
        self.schedulers[key] = MTPVarlen(curve)

    def commit(self, observations):
        candidate = dict(observations)
        if len(candidate) != len(observations):
            raise ValueError("Duplicate request identity")
        for confidence in candidate.values():
            if len(confidence.values) != self.depth or confidence.model_identity != self.model_identity:
                raise ValueError("Confidence model/depth mismatch")
        ids = {c.calibration_identity for c in candidate.values()}
        if len(ids) > 1 or (ids and self.calibration_identity not in (None, next(iter(ids)))):
            raise ValueError("Calibration identity changed; reset history explicitly")
        if ids:
            self.calibration_identity = next(iter(ids))
        self.previous, self.latest = self.latest, candidate

    def choose(self, keys, cost_key):
        remap_keys((), keys)
        if (cost_key.model_identity != self.model_identity or cost_key.batch_size != len(keys)
                or self.calibration_identity not in (None, cost_key.calibration_identity)):
            raise ValueError("Cost key does not match active batch")
        states = []
        for key in keys:
            new, old = self.latest.get(key), self.previous.get(key)
            states.append(MTPState(list(range(self.depth)),
                                   list(new.cumulative()) if new else [0.0] * self.depth,
                                   list(old.cumulative()) if old else [0.0] * self.depth))
        lengths = self.schedulers.get(cost_key, MTPVarlen()).schedule(states)
        return tuple(lengths)
