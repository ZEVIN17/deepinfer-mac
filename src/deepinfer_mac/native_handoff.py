"""One-way committed-prefix transfer to pinned MLX-LM native batching.

The caller must finish emitting the previous verified round before switching.
A native epoch owns its caches until drained; speculative re-entry requires a
new epoch. Draft state is discarded only after successful insertion of all rows.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class CommittedPrefix:
    uid: int
    cache: list
    tail: int
    remaining: int


def transfer_to_native(model, prefixes, *, stop_tokens, max_active, prefill_chunk, generator_factory=None):
    """Prepare on private caches and return generator plus native-to-owner UID map.

    Failure leaves every source cache untouched. No model forward occurs here;
    the unconsumed tail is submitted as a one-token prompt for the next tick.
    """
    if len({p.uid for p in prefixes}) != len(prefixes):
        raise ValueError("Duplicate committed request identity")
    if len(prefixes) > max_active or max_active < 1 or prefill_chunk < 1:
        raise ValueError("Invalid native batch capacity")
    for p in prefixes:
        if type(p.tail) is not int or p.tail < 0 or type(p.remaining) is not int or p.remaining < 1:
            raise ValueError("Committed prefix requires a pending tail and positive remaining budget")
    if generator_factory is None:
        from importlib.metadata import version
        if version('mlx-lm') != '0.32.2':
            raise ValueError('Native handoff requires audited mlx-lm==0.32.2')
        from mlx_lm.generate import BatchGenerator
        generator_factory = BatchGenerator
    generator = generator_factory(model, stop_tokens=stop_tokens, completion_batch_size=max_active,
                                  prefill_batch_size=1, prefill_step_size=prefill_chunk)
    try:
        uids = generator.insert([[p.tail] for p in prefixes],
                                max_tokens=[p.remaining for p in prefixes],
                                caches=[copy.deepcopy(p.cache) for p in prefixes]) if prefixes else []
        if len(uids) != len(prefixes) or len(set(uids)) != len(uids):
            raise RuntimeError('Native generator returned invalid request mapping')
        return generator, dict(zip(uids, (p.uid for p in prefixes)))
    except BaseException:
        generator.close()
        raise
