# Portions adapted from MLX-LM 0.32.2 generate.py speculative_generate_step.
# Copyright © 2023 Apple Inc. MIT license; see third_party/licenses/MLX-LM.txt.
"""Version-isolated greedy dynamic-budget adapter, retaining upstream rewind semantics."""

from __future__ import annotations

import time


def adaptive_generate_step(
    prompt,
    model,
    draft_model,
    model_cache,
    draft_cache,
    policy,
    max_tokens,
    rounds,
    cancel=None,
    memory_pressure=None,
):
    import mlx.core as mx
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

    if not can_trim_prompt_cache(model_cache) or not can_trim_prompt_cache(draft_cache):
        raise ValueError("Adaptive decoding requires trimmable target and draft caches")
    y = mx.array(prompt, mx.uint32)
    draft_y = y
    emitted = 0
    budget = accepted = 0

    def step(network, cache, inputs, n=1):
        logits = network(inputs[None], cache=cache)[:, -n:, :].squeeze(0)
        tokens = mx.argmax(logits, axis=-1).astype(mx.uint32)
        mx.eval(tokens)
        return tokens

    def rewind():
        trim_prompt_cache(model_cache, budget - accepted)
        trim_prompt_cache(draft_cache, max(budget - accepted - 1, 0))

    try:
        while emitted < max_tokens:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Request cancelled")
            budget = min(
                policy.choose(
                    max_tokens - emitted, memory_pressure=bool(memory_pressure and memory_pressure())
                ),
                max_tokens - emitted,
            )
            accepted = 0
            begin = time.perf_counter()
            draft = []
            current = draft_y
            for _ in range(budget):
                current = step(draft_model, draft_cache, current)
                draft.append(int(current.item()))
            draft_ms = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            verify_inputs = mx.concatenate([y, mx.array(draft, mx.uint32)])
            predicted = step(model, model_cache, verify_inputs, budget + 1).tolist()
            verify_ms = (time.perf_counter() - begin) * 1000
            n = 0
            while n < budget and predicted[n] == draft[n]:
                n += 1
            produced = min(n + 1, max_tokens - emitted)
            policy.observe(budget, n, draft_ms, verify_ms, produced)
            rounds.append(
                {
                    "budget": budget,
                    "accepted": min(n, produced),
                    "produced": produced,
                    "draft_ms": draft_ms,
                    "verify_ms": verify_ms,
                    "fallback": budget == 0,
                }
            )
            while accepted < n and emitted < max_tokens:
                accepted += 1
                emitted += 1
                yield predicted[accepted - 1], True
            if emitted < max_tokens:
                emitted += 1
                yield predicted[accepted], False
            if emitted == max_tokens:
                break
            y = mx.array([predicted[accepted]], mx.uint32)
            if budget == 0:
                # Zero-budget cycles leave draft state untouched. Catch up only
                # if a later measured cycle actually requests draft work.
                draft_y = mx.concatenate([draft_y, y])
            elif accepted == budget:
                draft_y = mx.array(draft[-1:] + [predicted[accepted]], mx.uint32)
            else:
                draft_y = y
            rewind()
            budget = accepted = 0
    finally:
        # This runs on EOS, cancellation and generator.close mid accepted block.
        rewind()
