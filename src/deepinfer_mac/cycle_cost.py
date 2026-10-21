"""Bounded full-cycle cost observations and conservative strategy decisions.

This is a Mac platform extension, separate from the DS reference optimizer.
Shape and context composition remain part of the key; equal token totals do not
imply equal work. No speculative strategy is selected without quality admission.
"""
from __future__ import annotations

import math
import statistics
from collections import OrderedDict, deque
from dataclasses import dataclass

from .ds.joint_budget import CostKey


@dataclass(frozen=True)
class CycleShape:
    identity: CostKey
    cache_classes: tuple[str, ...]
    contexts: tuple[int, ...]
    calls: tuple[tuple[str, int, int], ...]

    def __post_init__(self):
        if len(self.contexts) != self.identity.batch_size or any(type(c) is not int or c < 0 for c in self.contexts):
            raise ValueError('Context composition must match the batch')
        if not self.cache_classes or any(not isinstance(c, str) or not c for c in self.cache_classes):
            raise ValueError('Concrete cache identity required')
        if any(not phase or type(batch) is not int or batch < 1 or type(q) is not int or q < 1
               for phase, batch, q in self.calls):
            raise ValueError('Actual model-call shapes required')


class CycleCostTable:
    def __init__(self, max_keys=128, window=32, min_samples=3):
        if any(type(n) is not int or n < 1 for n in (max_keys, window, min_samples)) or min_samples > window:
            raise ValueError('Positive bounded sample sizes required')
        self.max_keys, self.window, self.min_samples = max_keys, window, min_samples
        self.samples = OrderedDict()

    def observe(self, shape, full_cycle_ms, emitted_tokens):
        if not isinstance(shape, CycleShape):
            raise TypeError('Shape-bound cycle identity required')
        if (type(full_cycle_ms) not in (int, float) or not math.isfinite(full_cycle_ms) or full_cycle_ms <= 0
                or type(emitted_tokens) is not int or emitted_tokens < 1):
            raise ValueError('Successful complete cycle with positive elapsed time and output required')
        values = self.samples.setdefault(shape, deque(maxlen=self.window))
        values.append((full_cycle_ms, emitted_tokens))
        self.samples.move_to_end(shape)
        while len(self.samples) > self.max_keys:
            self.samples.popitem(last=False)

    def estimate(self, shape):
        values = self.samples.get(shape, ())
        if len(values) < self.min_samples:
            return None
        return {'samples': len(values), 'full_cycle_ms': statistics.median(ms for ms, _ in values),
                'ms_per_emitted_token': statistics.median(ms / n for ms, n in values)}


class ExecutionPolicy:
    """Select among admitted complete-cost estimates; native is the cold default.

    Estimates are provided for one exact workload identity by the caller. This
    class never migrates state: a changed decision must be executed at a safe
    boundary. Native handoff is separately transactional and one-way per epoch.
    """
    def __init__(self, margin=0.15, confirmations=2):
        if not math.isfinite(margin) or not 0 <= margin < 1 or type(confirmations) is not int or confirmations < 1:
            raise ValueError('Invalid switching margin/confirmation count')
        self.margin, self.confirmations = margin, confirmations
        self.current, self.pending, self.streak, self.identity = 'native_batch', None, 0, None

    def choose(self, identity, estimates, *, quality_admitted=()):
        if identity != self.identity:
            self.current, self.pending, self.streak = 'native_batch', None, 0
            self.identity = identity
        costs = {name: value for name, value in estimates.items()
                 if name in {'native_batch', *quality_admitted}
                 and type(value) in (int, float) and math.isfinite(value) and value > 0}
        # Never infer a speedup without an actual native cost observation.
        if 'native_batch' not in costs or self.current not in costs:
            self.current, self.pending, self.streak = 'native_batch', None, 0
            return self.current
        best = min(costs, key=lambda name: (costs[name], name != 'native_batch', name))
        if best == self.current or costs[best] >= costs[self.current] * (1 - self.margin):
            self.pending, self.streak = None, 0
            return self.current
        self.streak = self.streak + 1 if self.pending == best else 1
        self.pending = best
        if self.streak >= self.confirmations:
            self.current, self.pending, self.streak = best, None, 0
        return self.current
