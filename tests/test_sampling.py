"""Portable sampler validation and supported API transport."""
import pytest
from fastapi.testclient import TestClient

from deepinfer_mac.server.app import create_app
from deepinfer_mac.state import GenerationRequest


@pytest.mark.parametrize('values', [
    {'temperature': float('nan')}, {'temperature': 3}, {'top_p': 0}, {'top_k': True},
    {'presence_penalty': float('inf')}, {'seed': True}, {'seed': -1}, {'seed': 2**32},
])
def test_invalid_sampling_values(values):
    with pytest.raises(ValueError):
        GenerationRequest(prompt='test', **values)


@pytest.mark.parametrize('mode', ['fixed_speculative', 'adaptive'])
@pytest.mark.parametrize('values', [{'temperature': .7}, {'top_p': .8}, {'top_k': 20}, {'presence_penalty': 1.5}])
def test_speculative_modes_reject_unverified_sampling(mode, values):
    with pytest.raises(ValueError, match='only greedy'):
        GenerationRequest(prompt='test', mode=mode, **values)


def test_api_preserves_sampling_seed_and_cache_flags():
    captured = []

    class Capture:
        def generate(self, request, emit=None, cancel=None):
            captured.append(request)
            return {'mode': request.mode, 'text': 'ok', 'token_ids': [2], 'prompt_tokens': 3,
                    'finish_reason': 'stop', 'metrics': {}}

    with TestClient(create_app(Capture)) as client:
        body = {'messages': [{'role': 'user', 'content': 'test'}], 'temperature': .7,
                'top_p': .8, 'top_k': 20, 'presence_penalty': 1.5, 'seed': 2**32-1,
                'enable_thinking': True, 'use_cache': False}
        assert client.post('/v1/chat/completions', json=body).status_code == 200
        request = captured[-1]
        assert (request.temperature, request.top_p, request.top_k, request.presence_penalty) == (.7, .8, 20, 1.5)
        assert request.seed == 2**32-1 and request.enable_thinking and not request.use_cache
        assert client.post('/v1/chat/completions', json={**body, 'top_k': 151936}).status_code == 422
        for mode in ('adaptive', 'fixed_speculative'):
            assert client.post('/v1/chat/completions', json={**body, 'mode': mode}).status_code == 422
    assert GenerationRequest(prompt='x', seed=0).seed == 0
