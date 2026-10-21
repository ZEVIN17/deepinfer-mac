"""Portable production-prefill controls; these do not prove MLX numerics."""

from types import SimpleNamespace

import pytest
from test_native_cache import NativeDouble

from deepinfer_mac.cache import NativePromptCacheStore
from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.state import GenerationRequest


class LayerState:
    def __init__(self, tokens=(), trimmable=False):
        self.tokens = list(tokens)
        self.trimmable = trimmable

    @property
    def state(self):
        return tuple(self.tokens)

    @property
    def nbytes(self):
        return 16 + 8 * len(self.tokens)

    def trim(self, count):
        assert self.trimmable, "Hybrid states must not be arbitrarily trimmed"
        count = min(count, len(self.tokens))
        del self.tokens[len(self.tokens) - count :]
        return count


class RecordingNative(NativeDouble):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.insertions = []

    def insert_cache(self, identity, tokens, cache, *, cache_type="assistant"):
        self.insertions.append((tuple(tokens), cache_type))
        super().insert_cache(identity, tokens, cache, cache_type=cache_type)


def trim(cache, count):
    removed = [state.trim(count) for state in cache]
    assert len(set(removed)) == 1
    return removed[0]


CACHE_RUNTIME = SimpleNamespace(
    LRUPromptCache=RecordingNative,
    can_trim_prompt_cache=lambda states: all(c.trimmable for c in states),
    trim_prompt_cache=trim,
)


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        tokens = []
        for message in messages:
            role = {"system": 10, "user": 20, "assistant": 30}[message["role"]]
            tokens += [role, *message["content"], 99]
        return tokens + ([70, 71] if add_generation_prompt else [])


class NoPartialTokenization:
    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("Disabled or uncached path must not render partial messages")


class Chunk:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def __getitem__(self, axis):
        assert axis is None
        return self


class ArrayRuntime:
    uint32 = object()

    def __init__(self):
        self.chunks = []
        self.evaluations = 0

    def array(self, tokens, dtype):
        assert dtype is self.uint32
        self.chunks.append(list(tokens))
        return Chunk(tokens)

    def eval(self, states):
        self.evaluations += 1


def model(chunk, *, cache):
    for state in cache:
        state.tokens.extend(chunk.tokens)


MESSAGES = [{"role": "system", "content": [1, 2]}, {"role": "user", "content": [3, 4, 5]}]
TOKENS = Tokenizer().apply_chat_template(
    MESSAGES, tokenize=True, add_generation_prompt=True, enable_thinking=False
)


def setup(enabled, use_cache=True, speculative=False):
    engine = InferenceEngine(message_checkpoints=enabled, checkpoint_interval_tokens=3)
    engine.model = model
    engine.draft_model = model
    engine.tokenizer = Tokenizer() if enabled and use_cache else NoPartialTokenization()
    engine.cache = NativePromptCacheStore(4096, _runtime=CACHE_RUNTIME)
    request = GenerationRequest(messages=MESSAGES, use_cache=use_cache, prefill_step_size=3)
    target = [LayerState()]
    draft = [LayerState()] if speculative else []
    return engine, request, target, draft, ArrayRuntime()


@pytest.mark.parametrize("enabled,use_cache", [(False, False), (False, True), (True, False)])
def test_disabled_and_uncached_prefill_preserve_original_shapes(enabled, use_cache):
    engine, request, target, draft, arrays = setup(enabled, use_cache)
    _, metrics = engine._prefill_prompt(request, TOKENS, target, target, draft, "model", 0, False, arrays)
    assert arrays.chunks == [TOKENS[:3], TOKENS[3:6], TOKENS[6:9], TOKENS[9:-1]]
    assert target[0].tokens == TOKENS[:-1] and metrics is None
    if use_cache:
        assert [len(tokens) for tokens, _ in engine.cache._store.insertions] == [3, 6, 9, 10]
    else:
        assert engine.cache.nbytes == 0


@pytest.mark.parametrize("speculative", [False, True])
def test_ordered_exact_message_snapshots_include_nontrimmable_target_and_draft(speculative):
    engine, request, target, draft, arrays = setup(True, speculative=speculative)
    caches = target + draft
    _, metrics = engine._prefill_prompt(
        request, TOKENS, caches, target, draft, "model", 0, speculative, arrays
    )
    assert arrays.chunks == [TOKENS[:3], TOKENS[3:4], TOKENS[4:6], TOKENS[6:9], TOKENS[9:-1]]
    assert [state.tokens for state in caches] == [TOKENS[:-1]] * len(caches)
    writes = engine.cache._store.insertions
    assert [(len(tokens), role) for tokens, role in writes] == [
        (3, "assistant"),
        (4, "system"),
        (6, "assistant"),
        (9, "user"),
        (10, "assistant"),
    ]
    for (_, tokens), states in engine.cache._store.entries.items():
        assert all(state.tokens == list(tokens) for state in states)
    assert metrics["prefill_shapes_changed"] and metrics["requires_real_model_sampling_validation"]
    assert metrics["logical_snapshot_bytes_written"] == sum(
        s["logical_bytes"] for s in metrics["snapshots"] if s["stored"]
    )


def test_hybrid_branch_restores_exact_system_prefix_and_replays_only_real_suffix():
    engine, request, target, draft, arrays = setup(True)
    engine._prefill_prompt(request, TOKENS, target, target, draft, "model", 0, False, arrays)
    branch_tokens = TOKENS[:5] + [8, 4, 5] + TOKENS[8:]
    branch_request = GenerationRequest(
        messages=[MESSAGES[0], {"role": "user", "content": [8, 4, 5]}], use_cache=True, prefill_step_size=3
    )
    restored, hit = engine.cache.get("model", branch_tokens)
    assert hit == 4 and restored[0].tokens == TOKENS[:4]
    arrays = ArrayRuntime()
    engine._prefill_prompt(branch_request, branch_tokens, restored, restored, [], "model", hit, False, arrays)
    assert [token for chunk in arrays.chunks for token in chunk] == branch_tokens[hit:-1]
    assert restored[0].tokens == branch_tokens[:-1]
    assert target[0].tokens == TOKENS[:-1]


def test_full_exact_trimmable_hit_rewinds_one_input_without_extra_prefill():
    engine, request, _, _, arrays = setup(True)
    engine.cache.put("model", TOKENS, [LayerState(TOKENS, trimmable=True)])
    cache, hit = engine.cache.get("model", TOKENS)
    assert hit == len(TOKENS) - 1 and cache[0].tokens == TOKENS[:-1]
    engine._prefill_prompt(request, TOKENS, cache, cache, [], "model", hit, False, arrays)
    assert arrays.chunks == []
    assert cache[0].tokens == TOKENS[:-1]


def test_warm_final_message_boundary_preserves_user_priority_without_forward():
    class OneTailTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            tokens = super().apply_chat_template(messages, **kwargs)
            return tokens[:-1] if kwargs["add_generation_prompt"] else tokens

    engine, request, target, _, arrays = setup(True)
    engine.tokenizer = OneTailTokenizer()
    tokens = engine.tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    engine._prefill_prompt(request, tokens, target, target, [], "model", 0, False, arrays)
    assert engine.cache._store.insertions[-1] == (tuple(tokens[:-1]), "user")
    restored, hit = engine.cache.get("model", tokens)
    assert hit == len(tokens) - 1
    warm_arrays = ArrayRuntime()
    _, metrics = engine._prefill_prompt(
        request, tokens, restored, restored, [], "model", hit, False, warm_arrays
    )
    assert warm_arrays.chunks == []
    assert restored[0].tokens == target[0].tokens == tokens[:-1]
    assert engine.cache._store.insertions[-1] == (tuple(tokens[:-1]), "user")
    assert metrics["snapshots"][-1]["cache_type"] == "user"
    assert not metrics["prefill_shapes_changed"]


def test_message_checkpoint_opt_in_requires_native_backend():
    assert not InferenceEngine().message_checkpoints
    with pytest.raises(ValueError, match="native cache"):
        InferenceEngine(cache_backend="exact", message_checkpoints=True)
