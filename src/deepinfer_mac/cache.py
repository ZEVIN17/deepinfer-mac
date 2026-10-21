"""Exact prefix LRU with byte accounting and branch-safe cache cloning."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Entry:
    identity: str
    tokens: tuple[int, ...]
    cache: list
    nbytes: int


def cache_bytes(cache):
    return sum(int(c.nbytes) for c in cache)


class PromptCacheStore:
    def __init__(self, budget_bytes=512 * 1024**2):
        if budget_bytes < 0:
            raise ValueError("Cache budget cannot be negative")
        self.budget_bytes = budget_bytes
        self.entries = OrderedDict()
        self.evictions = 0

    @property
    def nbytes(self):
        return sum(e.nbytes for e in self.entries.values())

    def clear(self):
        self.entries.clear()

    def get(self, identity, tokens):
        """Reuse only a fully matching stored prefix, always leave one input token.

        Caches are snapshots at the end of prefill, not generated continuations.
        No trimming or tensor slicing of unknown custom cache types is attempted.
        """
        selected = None
        for key, entry in self.entries.items():
            if (
                entry.identity == identity
                and len(entry.tokens) < len(tokens)
                and tuple(tokens[: len(entry.tokens)]) == entry.tokens
                and (selected is None or len(entry.tokens) > len(selected[1].tokens))
            ):
                selected = key, entry
        if selected is None:
            return None, 0
        key, entry = selected
        self.entries.move_to_end(key)
        # MLX arrays are immutable; deepcopy of cache containers protects offsets
        # and subsequent update/assignment on branches (same as upstream server).
        return copy.deepcopy(entry.cache), len(entry.tokens)

    def put(self, identity, tokens, cache):
        if not tokens or self.budget_bytes == 0:
            return
        nbytes = cache_bytes(cache)
        if nbytes > self.budget_bytes:
            return
        key = identity, tuple(tokens)
        self.entries.pop(key, None)
        while self.entries and self.nbytes + nbytes > self.budget_bytes:
            self.entries.popitem(last=False)
            self.evictions += 1
        self.entries[key] = Entry(identity, tuple(tokens), copy.deepcopy(cache), nbytes)
