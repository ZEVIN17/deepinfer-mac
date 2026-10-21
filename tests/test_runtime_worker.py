import pytest
from fastapi.testclient import TestClient
from test_continuous_worker import Engine, Session, config, wait
from test_joint_service import MessageSession

from deepinfer_mac.continuous import UnsupportedBatchCache
from deepinfer_mac.server.app import create_app
from deepinfer_mac.server.runtime_worker import RuntimeInferenceWorker
from deepinfer_mac.state import GenerationRequest


def test_auto_cold_start_uses_native_without_speculative_probe():
    worker = RuntimeInferenceWorker(Engine, config=config(), session_factory=Session)
    try:
        job = wait(worker.submit(GenerationRequest(prompt='x', max_tokens=2)))
        assert job.status == 'completed'
        assert job.result['metrics']['selected_execution_strategy'] == 'native_batch'
        assert job.result['metrics']['speculative_auto_admitted'] is False
    finally:
        worker.close()


@pytest.mark.parametrize('failure', [UnsupportedBatchCache('unsupported'), RuntimeError('load failed')])
def test_only_capability_failure_falls_back_to_serial(failure):
    def factory(*args):
        raise failure
    worker = RuntimeInferenceWorker(Engine, config=config(), session_factory=factory)
    try:
        job = wait(worker.submit(GenerationRequest(prompt='x', max_tokens=2)))
        if isinstance(failure, UnsupportedBatchCache):
            assert job.status == 'completed'
            assert job.result['metrics']['selected_execution_strategy'] == 'serial'
        else:
            assert job.status == 'failed' and job.error == 'load failed'
    finally:
        worker.close()


def test_auto_api_health_and_greedy_result():
    app = create_app(engine_factory=Engine, execution_backend='auto', continuous_config=config(),
                     batch_session_factory=MessageSession)
    with TestClient(app) as client:
        assert client.get('/health').json()['execution_backend'] == 'auto'
        response = client.post('/v1/chat/completions', json={'model': 'qwen3-4b',
                               'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 2})
        assert response.status_code == 200
        assert response.json()['deepinfer']['metrics']['selected_execution_strategy'] == 'native_batch'
