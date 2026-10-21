"""Opt-in Mac cost adapter: lagged arithmetic means of real complete cycles.

MTPVarlen itself is unchanged. A hybrid restore/replay cycle has a mixture of
full-accept and reject costs; the throughput objective needs expected cost.
This adapter supplies a rolling sample mean after a minimum sample count,
retaining replay/catch-up/host costs. Every entire observation is withheld for
one completed cycle, so neither N-1 acceptance nor its correlated runtime can
leak into the N-2 total-budget decision. This is an estimated cost curve, never
synthetic measured performance or an upstream DS calibration implementation.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, replace

from ..policy import AdaptivePolicy, DSAdaptivePolicy


class LaggedMeanRecoveryPolicy(DSAdaptivePolicy):
    def __init__(self, candidates=(0, 1, 2, 3, 4), min_samples=5, window=9, refresh_steps=64):
        if not 1 <= min_samples <= window:
            raise ValueError("Require 1 <= min_samples <= window")
        super().__init__(candidates, exploration_samples=min_samples, refresh_steps=refresh_steps)
        self.min_samples = min_samples
        self.window = window
        self.available = defaultdict(lambda: deque(maxlen=window))
        for budget in self.candidates:
            self.available[budget]
        self.pending_observation = None
        self.last_published_step = {}
        self.last_cost_curve = None

    def choose(self, remaining, memory_pressure=False):
        eligible = [k for k in self.candidates if k <= remaining]
        if not eligible:
            raise ValueError("remaining must allow at least one token")
        self.last_replay = None
        self.last_cost_curve = None
        if memory_pressure:
            budget = 0
        else:
            explore = next((k for k in eligible if len(self.available[k]) < self.min_samples), None)
            if explore is None:
                explore = next((k for k in eligible
                                if self.steps - self.last_published_step[k] >= self.refresh_steps), None)
            if explore is not None:
                budget = explore
            else:
                points = []
                for k in eligible:
                    samples = self.available[k]
                    # Current decision is N; process_cnt records N-1. Every
                    # input must therefore be no newer than N-2.
                    if any(s["cycle"] > self.state.process_cnt - 1 for s in samples):
                        raise RuntimeError("Cost feedback violated N-2 isolation")
                    points.append((k + 1, statistics.mean(s["cycle_ms"] for s in samples)))
                # No residual observations are fed to this core. This empirical
                # curve already includes the full contemporaneous cost; adding
                # residuals against an older/different curve would double count.
                # Nothing is cleared to conceal a residual or a failed sample.
                self.scheduler.set_decode_perf_curve(points)
                nd = max(eligible)
                state = replace(self.state, draft_tokens=self.state.draft_tokens[:nd],
                                accept_rates=self.state.accept_rates[:nd],
                                prev_accept_rates=self.state.prev_accept_rates[:nd])
                budget = self.scheduler.schedule([state])[0] - 1
                if self.capture_replay:
                    self.last_cost_curve = {
                        "kind": "estimated lagged arithmetic mean of measured full cycles",
                        "feedback_cutoff_cycle": self.state.process_cnt - 1,
                        "points": [{"graph_size": k + 1,
                                    "measured_samples": list(self.available[k]),
                                    "estimated_cost_ms": cost} for k, (_, cost) in zip(eligible, points)],
                        "residual": "not used: full empirical curve already contains drift/replay/host cost",
                    }
                    self.last_replay = {
                        "curve": self.scheduler.curve, "state": asdict(state), "budget": budget,
                        "runtime_projection": {"forward_slope_ms": 0.0, "sampler_ms": 0.0},
                        "cost_model": self.last_cost_curve,
                    }
        self.state.schedule_cnt += 1
        self.decisions += 1
        self.last_budget = budget
        return budget

    def observe_cycle(self, budget, accepted, draft_ms, verify_ms, produced, *, cycle_ms,
                      stage_ms=None, **kwargs):
        if self.state.schedule_cnt != self.state.process_cnt + 1:
            raise ValueError("A complete observation requires one pending scheduled cycle")
        if (
            budget not in self.candidates or not 0 <= accepted <= budget
            or not accepted <= produced <= accepted + 1 or produced < 1
        ):
            raise ValueError("Invalid completed cycle token accounting")
        if not math.isfinite(cycle_ms) or cycle_ms <= 0:
            raise ValueError("Complete-cycle measurement must be finite positive milliseconds")
        if stage_ms is not None and any(not math.isfinite(v) or v < 0 for v in stage_ms.values()):
            raise ValueError("Stage measurements must be finite nonnegative milliseconds")
        if stage_ms is not None and sum(stage_ms.values()) > cycle_ms + 1e-6:
            raise ValueError("Measured stages cannot exceed the measured complete cycle")
        if self.pending_observation is not None:
            previous = self.pending_observation
            # Early close has no correction/final golden output; its work is
            # still reported end-to-end but is not a full steady-state cycle.
            if previous["complete_output_cycle"]:
                self.available[previous["budget"]].append(previous)
                self.last_published_step[previous["budget"]] = previous["cycle"]
        complete = AdaptivePolicy.observe(self, budget, accepted, 0.0, cycle_ms, produced, cycle_ms=cycle_ms)
        if complete:
            self.costs[budget].append(cycle_ms)  # Never N-1 decision inputs.
        self.pending_observation = {
            "cycle": self.steps, "budget": budget, "cycle_ms": float(cycle_ms),
            "accepted": accepted, "produced": produced,
            "stage_ms": dict(stage_ms) if stage_ms is not None else None,
            "complete_output_cycle": complete,
            "catchup_tokens": kwargs.get("catchup_tokens", 0),
        }
        self.state.prev_accept_rates = self.state.accept_rates
        if complete:
            self.state.accept_rates = [(self.positions[p][0] + 1) / (self.positions[p][1] + 2)
                                       for p in range(max(self.candidates))]
        self.state.process_cnt += 1

    def observe(self, budget, accepted, draft_ms, verify_ms, produced, *, cycle_ms=None, **kwargs):
        if cycle_ms is None:
            raise ValueError("lagged_mean requires the measured complete cycle, including recovery")
        self.observe_cycle(budget, accepted, draft_ms, verify_ms, produced, cycle_ms=cycle_ms, **kwargs)

    def summary(self):
        return {
            **super().summary(),
            "cost_adapter": "opt-in lagged_mean; Mac adaptation, not upstream calibration",
            "startup_curve_scope": "estimated rolling mean of measured complete cycles; N-2 feedback",
            "min_samples": self.min_samples, "window": self.window,
            "feedback_age": "all cost observations through N-2; latest N-1 withheld",
            "estimated_cost_ms_by_graph": {
                str(k + 1): statistics.mean(s["cycle_ms"] for s in samples)
                for k, samples in self.available.items() if len(samples) >= self.min_samples
            },
            "known_samples_by_graph": {str(k + 1): len(v) for k, v in self.available.items()},
            "unknown_graphs": [k + 1 for k in self.candidates if len(self.available[k]) < self.min_samples],
            "runtime_residual": "not used; no residual updates or history resets",
            "performance_gain_claim": False,
        }
