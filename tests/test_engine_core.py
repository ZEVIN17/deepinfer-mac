"""Portable behavior checks, with no import of MLX at collection time."""

import pytest

from deepinfer_mac.cache import PromptCacheStore
from deepinfer_mac.policy import AdaptivePolicy
from deepinfer_mac.state import GenerationRequest


class Cache:
    def __init__(self, size=8):
        self.nbytes = size
        self.offset = 4


def test_cache_isolation_identity_and_eviction():
    store = PromptCacheStore(16)
    original = [Cache()]
    store.put("model-a", [1, 2], original)
    original[0].offset = 99
    branch, hit = store.get("model-a", [1, 2, 3])
    assert hit == 2 and branch[0].offset == 4
    branch[0].offset = 20
    assert store.get("model-a", [1, 2, 4])[0][0].offset == 4
    assert store.get("model-b", [1, 2, 3]) == (None, 0)
    assert store.get("model-a", [1, 3, 2]) == (None, 0)
    store.put("model-a", [3], [Cache()])
    store.put("model-a", [4], [Cache()])
    assert store.nbytes == 16 and store.evictions == 1
    assert store.get("model-a", [1, 2, 3]) == (None, 0)
    store.put("model-a", [5], [Cache(30)])
    assert store.nbytes == 16


def test_policy_uses_measured_cost_not_high_acceptance():
    policy = AdaptivePolicy(candidates=(0, 1, 4), exploration_samples=1)
    for budget, accepted, ms, produced in [(0, 0, 2, 1), (1, 1, 5, 2), (4, 4, 15, 5)]:
        assert policy.choose(10) == budget
        policy.observe(budget, accepted, ms / 2, ms / 2, produced)
    assert policy.choose(10) == 0
    assert policy.choose(10, memory_pressure=True) == 0
    assert policy.summary()["acceptance_by_position"]["0"] == 1


def test_request_rejects_unimplemented_modes_and_bounds():
    for params in ({"mode": "dspark"}, {"max_tokens": 0}, {"num_draft_tokens": 0}, {"prefill_step_size": 0}):
        with pytest.raises(ValueError):
            GenerationRequest(prompt="hi", **params)
    with pytest.raises(ValueError):
        GenerationRequest()


def test_resource_budget_rejects_weights_before_loading(monkeypatch):
    from deepinfer_mac.models import get_profile, validate_initial_budget

    with pytest.raises(MemoryError, match="Pinned weights"):
        validate_initial_budget("qwen3-4b", memory_limit_bytes=1024**3)
    with monkeypatch.context() as patch:
        patch.setitem(get_profile("qwen3.5-9b-8bit"), "draft", None)
        with pytest.raises(ValueError, match="compatible draft"):
            validate_initial_budget("qwen3.5-9b-8bit", speculative=True)
    assert validate_initial_budget("qwen3-0.6b", memory_limit_bytes=2 * 1024**3) < 2 * 1024**3


def test_swap_unavailable_is_explicit(monkeypatch):
    import psutil

    from deepinfer_mac.engine import read_swap_bytes

    def denied():
        raise OSError("not permitted")

    monkeypatch.setattr(psutil, "swap_memory", denied)
    assert read_swap_bytes() is None


@pytest.mark.parametrize("text", ["tail ", "结尾", "suffix\n"])
def test_final_segment_is_consumed_once(text):
    from deepinfer_mac.engine import finalize_segment

    class ConsumingDetokenizer:
        def __init__(self):
            self.reads = 0
            self.finalized = False

        def finalize(self):
            self.finalized = True

        @property
        def last_segment(self):
            assert self.finalized
            self.reads += 1
            return text if self.reads == 1 else ""

    detokenizer = ConsumingDetokenizer()
    assert finalize_segment(detokenizer) == text
    assert detokenizer.reads == 1
