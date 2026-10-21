"""Mac reimplementation of DeepInfer's MTPVarlen budget algorithm (Apache-2.0).

Reference: deepseek-ai/DeepInfer@774ce244b87cc0eb8d4b94ab9f5de2ab22049f01,
csrc/engine/mtp_varlen.hpp. The unmodified header is an independent test oracle.
This module preserves float32 accumulation and adjacent-history isolation; it
contains neither the upstream Confidence Head nor a model execution backend.
"""

from __future__ import annotations

import math
import statistics
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import pairwise


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class SamplerRuntimeEstimator:
    def __init__(self):
        self.samples = deque(maxlen=9)

    def update(self, value_ms):
        self.samples.append(float(value_ms))

    def get(self):
        return statistics.median(self.samples) if len(self.samples) >= 5 else None


class ForwardRuntimeEstimator:
    """Same 5000-sample, equal-graph median semantics; portable sorted medians.

    Python recomputes medians rather than using C++ balanced multisets. This is
    a deliberate implementation difference, with identical window/thresholds.
    """

    def __init__(self):
        self.window = deque()
        self.by_graph = defaultdict(list)

    def update(self, graph_size, normalized_residual):
        if len(self.window) == 5000:
            graph, value = self.window.popleft()
            self.by_graph[graph].remove(value)
            if not self.by_graph[graph]:
                del self.by_graph[graph]
        value = float(normalized_residual)
        self.window.append((graph_size, value))
        self.by_graph[graph_size].append(value)

    def get(self):
        eligible = [statistics.median(v) for v in self.by_graph.values() if len(v) >= 10]
        return statistics.median(eligible) if eligible else 0.0


@dataclass
class MTPState:
    draft_tokens: list[int] = field(default_factory=list)
    accept_rates: list[float] = field(default_factory=list)
    prev_accept_rates: list[float] = field(default_factory=list)
    has_draft_info: bool = False
    process_cnt: int = 0
    schedule_cnt: int = 0


class MTPVarlen:
    def __init__(self, curve=()):
        self.set_decode_perf_curve(curve)
        self.forward = ForwardRuntimeEstimator()
        self.sampler_by_n = defaultdict(SamplerRuntimeEstimator)
        self.sampler_fallback = SamplerRuntimeEstimator()

    def set_decode_perf_curve(self, curve):
        curve = [(int(size), f32(ms)) for size, ms in curve]
        if any(size <= 0 or not math.isfinite(ms) or ms <= 0 for size, ms in curve):
            raise ValueError("Curve requires positive graph sizes and finite positive milliseconds")
        if any(a[0] >= b[0] for a, b in pairwise(curve)):
            raise ValueError("Graph sizes must be strictly increasing")
        self.curve = curve

    def update_runtime(self, num_seqs, graph_size, forward_ms, sampler_ms):
        if num_seqs < 1 or graph_size < num_seqs:
            raise ValueError("Runtime sample requires graph_size >= num_seqs >= 1")
        point = next((p for p in self.curve if p[0] >= graph_size), None)
        if point is None:
            raise ValueError("Runtime sample exceeds the startup curve")
        if not all(math.isfinite(v) and v >= 0 for v in (forward_ms, sampler_ms)):
            raise ValueError("Runtime samples must be finite nonnegative milliseconds")
        self.forward.update(graph_size, (forward_ms - point[1]) / graph_size)
        self.sampler_by_n[num_seqs].update(sampler_ms)
        self.sampler_fallback.update(sampler_ms)

    def schedule(self, states):
        if not self.curve:
            return [1 + len(s.draft_tokens) for s in states]
        if not states:
            return []  # Host boundary: upstream scheduler never calls an empty batch.
        if len(states) > self.curve[-1][0]:
            raise ValueError("Curve cannot accommodate all golden tokens")
        total_rates, allocation_rates = [], []
        for i, state in enumerate(states):
            if not state.process_cnt <= state.schedule_cnt <= state.process_cnt + 1:
                raise ValueError("schedule/process counters must differ by zero or one")
            nd = len(state.draft_tokens)
            paired = state.process_cnt == state.schedule_cnt
            has_draft = state.has_draft_info or not paired
            for queue, rates in (
                (total_rates, state.prev_accept_rates if paired else state.accept_rates),
                (allocation_rates, state.accept_rates),
            ):
                if rates and len(rates) != nd:
                    raise ValueError("Accept-rate vector must match available draft length")
                padded = rates or [float(has_draft)] * nd
                if any(not math.isfinite(r) or not 0 <= r <= 1 for r in padded):
                    raise ValueError("Acceptance probabilities must be finite and in [0, 1]")
                queue.extend((f32(r), i) for r in padded)
        total_rates.sort(key=lambda pair: (-pair[0], pair[1]))
        allocation_rates.sort(key=lambda pair: (-pair[0], pair[1]))
        n = len(states)
        sampler_ms = self.sampler_by_n[n].get()
        if sampler_ms is None:
            sampler_ms = self.sampler_fallback.get() or 0.0
        slope = max(0.0, self.forward.get())
        expected, size = f32(n), n
        best_size, best_throughput = n, 0.0
        index = 0
        for rate in [None, *total_rates]:
            if rate is not None:
                expected = f32(expected + rate[0])
                size += 1
            while index < len(self.curve) and size > self.curve[index][0]:
                index += 1
            if index == len(self.curve):
                break
            graph_size, baseline_ms = self.curve[index]
            throughput = expected / (baseline_ms + slope * graph_size + sampler_ms)
            if throughput > best_throughput:
                best_throughput, best_size = throughput, size
        result = [1] * n
        for _, index in allocation_rates[: best_size - n]:
            result[index] += 1
        return result
