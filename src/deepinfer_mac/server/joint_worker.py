"""Joint service reuses the bounded continuous owner and terminal-state logic."""
from functools import partial

from ..joint_service import MLXJointBatchSession, public_round_metrics
from .continuous_worker import ContinuousInferenceWorker


class JointInferenceWorker(ContinuousInferenceWorker):
    backend_name = "joint_experimental"

    def __init__(self, engine_factory, *, joint_config, session_factory=None, **kwargs):
        self.joint_config = joint_config
        super().__init__(engine_factory, session_factory=session_factory or partial(
            MLXJointBatchSession, joint=joint_config), **kwargs)

    def submit(self, request, compare=False):
        self.joint_config.validate_request(request, compare)
        return super().submit(request, compare)

    def decorate_result(self, result):
        result["mode"] = "joint_experimental"
        for name in ("continuous_policy", "tick_token_budget", "prefill_batch_size"):
            result["metrics"].pop(name, None)
        result["metrics"].update(
            requested_mode="baseline", effective_mode="joint_experimental",
            experimental_joint_verification=True, confidence_source="empirical_uncalibrated",
            trained_confidence=False, prefill_scheduling="synchronous_chunked_no_preemption",
            cancellation_boundary="verified_round", strict_serial_equivalence_validated=False,
            draft_depth=self.joint_config.depth, verification_strategy=self.joint_config.strategy,
            last_joint_round=public_round_metrics(getattr(self.session, "last_report", None) or {}),
            cost_curve_available=bool((getattr(self.session, "last_report", None) or {}).get("cost_curve_available", False)),
        )
        return result
