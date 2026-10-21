"""Conservative automatic runtime: native batch, serial capability fallback.

Speculative selection stays unavailable until quality and exact-identity cost
admission are delivered. This class never probes speculative work on user jobs.
"""
from ..continuous import ContinuousConfig, UnsupportedBatchCache
from .continuous_worker import ContinuousInferenceWorker


class RuntimeInferenceWorker(ContinuousInferenceWorker):
    backend_name = 'auto'

    def __init__(self, engine_factory, *, config=None, **kwargs):
        super().__init__(engine_factory, config=config or ContinuousConfig(policy='upstream'), **kwargs)

    def create_session(self, job):
        try:
            return super().create_session(job)
        except UnsupportedBatchCache:
            # Constructor rejects before creating a generator or admitting a
            # UID. The loaded target is reusable by the exclusive serial path.
            job.emit({'type': 'execution_strategy', 'run_id': job.id,
                      'requested': 'auto', 'selected': 'serial', 'reason': 'unsupported_batch_cache'})
            return None

    def decorate_result(self, result):
        result['metrics'].update(requested_execution_strategy='auto',
                                 selected_execution_strategy='native_batch',
                                 speculative_auto_admitted=False)
        return result

    def decorate_serial_result(self, result):
        result['metrics'].update(requested_execution_strategy='auto',
                                 selected_execution_strategy='serial',
                                 speculative_auto_admitted=False)
        return result
