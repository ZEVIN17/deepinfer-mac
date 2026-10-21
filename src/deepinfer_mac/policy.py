"""Independent cost-based greedy draft budgeting; no confidence/MTP claim."""

from __future__ import annotations

import statistics
from collections import defaultdict, deque


class AdaptivePolicy:
    """Measure complete cycles, explore a bounded curve, then minimize ms/token.

    Budget zero is a genuine target-only fallback. Timings are synchronized by the
    adapter. Acceptance survival at each position is tracked for diagnostics; the
    decision uses actual cycle costs rather than acceptance rate alone.
    """

    def __init__(self, candidates=(0, 1, 2, 3, 4), exploration_samples=2):
        self.candidates = tuple(sorted(set(candidates)))
        if not self.candidates or self.candidates[0] != 0:
            raise ValueError("Candidates must include zero fallback")
        self.samples = defaultdict(lambda: deque(maxlen=9))
        self.positions = defaultdict(lambda: [0, 0])
        self.exploration_samples = exploration_samples
        self.last_budget = 0
        self.steps = 0

    def choose(self, remaining, memory_pressure=False):
        if memory_pressure:
            return 0
        eligible = [k for k in self.candidates if k <= remaining]
        for budget in eligible:
            if len(self.samples[budget]) < self.exploration_samples:
                self.last_budget = budget
                return budget
        costs = {k: statistics.median(self.samples[k]) for k in eligible}
        best = min(costs, key=costs.get)
        # A 5% hysteresis avoids switching on small timing fluctuations.
        if self.last_budget in costs and costs[self.last_budget] <= costs[best] * 1.05:
            best = self.last_budget
        self.last_budget = best
        return best

    def observe(self, budget, accepted, draft_ms, verify_ms, produced):
        if produced > 0:
            self.samples[budget].append((draft_ms + verify_ms) / produced)
        for position in range(budget):
            self.positions[position][0] += int(position < accepted)
            self.positions[position][1] += 1
        self.steps += 1

    def summary(self):
        return {
            "cycle_ms_per_token": {str(k): statistics.median(v) for k, v in self.samples.items() if v},
            "acceptance_by_position": {str(k): a / n for k, (a, n) in self.positions.items() if n},
        }
