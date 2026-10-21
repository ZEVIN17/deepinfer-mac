"""Real-device check of full-history penalties after prefix/cache reuse."""
import pytest

pytestmark = pytest.mark.metal


def test_full_prompt_presence_penalty_includes_consumed_prefix():
    import mlx.core as mx

    from deepinfer_mac.sampling import generation_sampling
    from deepinfer_mac.state import GenerationRequest

    request = GenerationRequest(prompt='test', temperature=.7, top_p=.8, top_k=2, presence_penalty=1.5)
    processors = generation_sampling(request, [2, 2, 7])['logits_processors']
    actual = processors[0](mx.array([4, 5], mx.uint32), mx.zeros((1, 8)))
    assert actual.tolist() == [[0, 0, -1.5, 0, -1.5, -1.5, 0, -1.5]]
