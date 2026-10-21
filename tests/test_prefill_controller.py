"""Regression checks for workload-scoped prefill feedback and pinned bridge."""
from types import SimpleNamespace

import pytest

from deepinfer_mac.continuous import ContinuousConfig, PrefillController, set_prefill_chunk


def responses(done=False):
    return [SimpleNamespace(end_of_prompt=done)]


def test_decode_and_transfer_ticks_do_not_collapse_or_inflate_prefill():
    control = PrefillController(ContinuousConfig(prefill_chunk=128))
    for _ in range(20):
        assert control.observe(1000, []) == 128
        assert control.observe(0, responses(done=True)) == 128
    assert control.observe(100, responses()) == 64
    assert control.observe(1, []) == 64
    assert control.observe(100, responses(done=True)) == 64


def test_recovery_requires_consecutive_prefill_evidence_and_respects_budget():
    control = PrefillController(ContinuousConfig(max_active=4, tick_token_budget=100, prefill_chunk=128))
    assert control.chunk == 96
    assert control.observe(40, responses()) == 48
    assert control.observe(1, responses()) == 48
    assert control.observe(1, []) == 48
    assert control.observe(1, responses()) == 48
    assert control.observe(1, responses()) == 64
    for _ in range(20):
        control.observe(1, responses())
    assert control.chunk == 96
    for _ in range(20):
        control.observe(40, responses())
    assert control.chunk == 32


def test_upstream_control_is_fixed():
    control = PrefillController(ContinuousConfig(policy="upstream", prefill_chunk=64))
    for elapsed in (1, 1, 1000, 1000):
        assert control.observe(elapsed, responses()) == 64


@pytest.mark.parametrize("elapsed", [float("nan"), float("inf"), -1])
def test_invalid_measurement_rejected(elapsed):
    with pytest.raises(ValueError):
        PrefillController(ContinuousConfig()).observe(elapsed, responses())


def test_bridge_updates_current_and_future_batches():
    generator = SimpleNamespace(prefill_step_size=128, _prompt_batch=SimpleNamespace(prefill_step_size=128))
    set_prefill_chunk(generator, 32)
    assert generator.prefill_step_size == generator._prompt_batch.prefill_step_size == 32
    generator._prompt_batch = SimpleNamespace(prefill_step_size=generator.prefill_step_size)
    set_prefill_chunk(generator, 16)
    assert generator.prefill_step_size == generator._prompt_batch.prefill_step_size == 16


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_bridge_rejects_invalid_chunk_without_mutation(chunk):
    generator = SimpleNamespace(prefill_step_size=128, _prompt_batch=SimpleNamespace(prefill_step_size=128))
    with pytest.raises(ValueError):
        set_prefill_chunk(generator, chunk)
    assert generator.prefill_step_size == generator._prompt_batch.prefill_step_size == 128


def test_bridge_checks_current_batch_before_mutation():
    generator = SimpleNamespace(prefill_step_size=128, _prompt_batch=object())
    with pytest.raises(RuntimeError):
        set_prefill_chunk(generator, 32)
    assert generator.prefill_step_size == 128
