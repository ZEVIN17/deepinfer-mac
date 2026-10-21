"""Transactional identity and failure tests without device imports."""
import pytest

from deepinfer_mac.native_handoff import CommittedPrefix, transfer_to_native


def test_handoff_transfers_only_pending_tail_on_private_state():
    cache = [{'tokens': [1, 2]}]
    class Generator:
        def __init__(self, model, **kwargs):
            pass
        def insert(self, prompts, max_tokens, caches):
            assert prompts == [[3]] and max_tokens == [7]
            caches[0][0]['tokens'].append(3)
            return [100]
    _, mapping = transfer_to_native(None, [CommittedPrefix(8, cache, 3, 7)], stop_tokens=[],
                                     max_active=2, prefill_chunk=8, generator_factory=Generator)
    assert mapping == {100: 8}
    assert cache == [{'tokens': [1, 2]}]


def test_partial_insert_failure_does_not_change_source_and_closes_generator():
    cache, closed = [{'tokens': [1]}], []
    class Generator:
        def __init__(self, model, **kwargs):
            pass
        def insert(self, prompts, max_tokens, caches):
            caches[0][0]['tokens'].append(2)
            raise RuntimeError('injected insertion failure')
        def close(self):
            closed.append(True)
    with pytest.raises(RuntimeError, match='injected'):
        transfer_to_native(None, [CommittedPrefix(8, cache, 2, 7)], stop_tokens=[],
                           max_active=2, prefill_chunk=8, generator_factory=Generator)
    assert cache == [{'tokens': [1]}] and closed == [True]


def test_rejects_duplicate_or_terminal_rows_before_constructing_generator():
    def forbidden(*args, **kwargs):
        raise AssertionError('must reject before construction')
    for rows in ([CommittedPrefix(1, [], 2, 0)], [CommittedPrefix(1, [], 2, 1)] * 2):
        with pytest.raises(ValueError):
            transfer_to_native(None, rows, stop_tokens=[], max_active=2, prefill_chunk=8,
                               generator_factory=forbidden)
