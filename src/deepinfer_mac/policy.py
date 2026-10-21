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
        self.partial_cycles = deque(maxlen=9)

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

    def observe(self, budget, accepted, draft_ms, verify_ms, produced, *, cycle_ms=None):
        # A completed greedy verification cycle emits accepted drafts followed
        # by one target/golden token. If the consumer closes before that token,
        # later draft positions are censored, not observed rejections. Retain
        # the actual work for diagnostics but exclude BOTH cost and acceptance
        # from persistent estimators. Full cycles retain their prior behavior.
        if produced <= accepted:
            self.partial_cycles.append({
                "budget": budget, "accepted": accepted, "produced": produced,
                "cycle_ms": cycle_ms if cycle_ms is not None else draft_ms + verify_ms,
                "reason": "output closed before the target/golden token",
            })
            self.steps += 1
            return False
        if produced > 0:
            self.samples[budget].append((draft_ms + verify_ms) / produced)
        for position in range(budget):
            self.positions[position][0] += int(position < accepted)
            self.positions[position][1] += 1
        self.steps += 1
        return True

    def observe_cycle(self, budget, accepted, draft_ms, verify_ms, produced, *, cycle_ms, **kwargs):
        # Full measured wall cost, including host handling, replaces partial GPU cost.
        self.observe(budget, accepted, 0.0, cycle_ms, produced)

    def summary(self):
        return {
            "cycle_ms_per_token": {str(k): statistics.median(v) for k, v in self.samples.items() if v},
            "acceptance_by_position": {str(k): a / n for k, (a, n) in self.positions.items() if n},
            "partial_cycles_excluded": list(self.partial_cycles),
        }


class FixedPolicy:
    """Same execution loop control for baseline and tuned fixed-budget ablations."""

    def __init__(self, budget):
        self.budget = budget

    def choose(self, remaining, memory_pressure=False):
        return 0 if memory_pressure else min(self.budget, remaining)

    def observe(self, *args, **kwargs):
        pass


class DSAdaptivePolicy(AdaptivePolicy):
    """DeepInfer MTPVarlen-derived scheduler with empirical acceptance.

    Independent draft execution is not free on MLX: startup curve points measure
    the whole draft/verify/host cycle, not only the target forward. Curve startup
    and periodic exploration are platform adapters, separate from MTPVarlen.
    """

    def __init__(self, candidates=(0, 1, 2, 3, 4), exploration_samples=2, refresh_steps=64):
        from .ds import MTPState, MTPVarlen

        super().__init__(candidates, exploration_samples)
        if self.candidates != tuple(range(max(self.candidates) + 1)):
            raise ValueError("DS adapter requires contiguous draft budgets starting at zero")
        self.costs = defaultdict(lambda: deque(maxlen=9))
        self.state = MTPState(draft_tokens=list(range(max(self.candidates))), has_draft_info=True)
        self.scheduler = MTPVarlen()
        self.refresh_steps = refresh_steps
        self.last_sample_step = {}
        self.decisions = 0
        self.catchup_cycles_ms = deque(maxlen=9)
        self.last_replay = None
        self.capture_replay = True

    def choose(self, remaining, memory_pressure=False):
        from dataclasses import asdict, replace

        eligible = [k for k in self.candidates if k <= remaining]
        self.last_replay = None
        if memory_pressure:
            budget = 0
        else:
            explore = next((k for k in eligible if len(self.costs[k]) < self.exploration_samples), None)
            if explore is None:
                # Expiring fallback / task changes can never permanently disable a budget.
                explore = next(
                    (k for k in eligible if self.steps - self.last_sample_step[k] >= self.refresh_steps), None
                )
            if explore is not None:
                budget = explore
            else:
                curve = [(k + 1, statistics.median(self.costs[k])) for k in eligible]
                if not self.scheduler.curve or curve[-1][0] > self.scheduler.curve[-1][0]:
                    # Fixed startup curve, as upstream. Extending a previously
                    # short request's curve starts a fresh residual estimator.
                    from .ds import MTPVarlen

                    self.scheduler = MTPVarlen(curve)
                nd = max(eligible)
                state = replace(
                    self.state,
                    draft_tokens=self.state.draft_tokens[:nd],
                    accept_rates=self.state.accept_rates[:nd],
                    prev_accept_rates=self.state.prev_accept_rates[:nd],
                )
                budget = self.scheduler.schedule([state])[0] - 1
                if self.capture_replay:
                    self.last_replay = {
                        "curve": self.scheduler.curve, "state": asdict(state), "budget": budget,
                        "runtime_projection": {
                            "forward_slope_ms": max(0.0, self.scheduler.forward.get()),
                            "sampler_ms": self.scheduler.sampler_by_n[1].get()
                            or self.scheduler.sampler_fallback.get() or 0.0,
                        },
                    }
        self.state.schedule_cnt += 1
        self.decisions += 1
        self.last_budget = budget
        return budget

    def observe(self, budget, accepted, draft_ms, verify_ms, produced, *, cycle_ms=None, **kwargs):
        complete_ms = cycle_ms if cycle_ms is not None else draft_ms + verify_ms
        complete = super().observe(budget, accepted, draft_ms, verify_ms, produced, cycle_ms=complete_ms)
        catchup = kwargs.get("catchup_tokens", 0) > 0
        if complete and catchup:
            self.catchup_cycles_ms.append(complete_ms)
        if complete and produced and not catchup:
            self.costs[budget].append(max(complete_ms, 1e-6))
            self.last_sample_step[budget] = self.steps
            if self.scheduler.curve and budget + 1 <= self.scheduler.curve[-1][0]:
                # Curve already includes sampler/host cost: do not count it twice.
                self.scheduler.update_runtime(1, budget + 1, complete_ms, 0)
        self.state.prev_accept_rates = self.state.accept_rates
        if complete:
            self.state.accept_rates = [
                (self.positions[p][0] + 1) / (self.positions[p][1] + 2)
                for p in range(max(self.candidates))
            ]
        self.state.process_cnt += 1

    def observe_cycle(self, budget, accepted, draft_ms, verify_ms, produced, *, cycle_ms, **kwargs):
        self.observe(budget, accepted, draft_ms, verify_ms, produced, cycle_ms=cycle_ms, **kwargs)

    def abort_cycle(self):
        """Discard a scheduled forward that never yielded a completed token.

        A device/model failure provides no acceptance or runtime observation.
        The failed request's mutable caches are discarded by the caller, while
        this policy can outlive the request and must remain schedulable.
        """
        if self.state.schedule_cnt == self.state.process_cnt + 1:
            self.state.schedule_cnt -= 1
        self.last_replay = None

    def summary(self):
        return {
            **super().summary(),
            "implementation": "DeepInfer MTPVarlen-derived scheduler with empirical acceptance",
            "source_commit": "774ce244b87cc0eb8d4b94ab9f5de2ab22049f01",
            "startup_curve_scope": "complete draft+verify+host cycle; sampler residual counted once",
            "cycle_ms_by_graph": {str(k + 1): statistics.median(v) for k, v in self.costs.items() if v},
            "decisions": self.decisions,
            "refresh_steps": self.refresh_steps,
            "confidence_head": False,
            "catchup_cycle_ms": list(self.catchup_cycles_ms),
        }
