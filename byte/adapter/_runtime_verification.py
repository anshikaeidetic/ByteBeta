"""Thin verification-layer facade for the split adapter runtime."""

from byte.adapter.pipeline.consensus import (
    _run_cheap_consensus_async,
    _run_cheap_consensus_sync,
    _should_attempt_cheap_consensus,
)
from byte.adapter.pipeline.escalation import (
    _escalation_action_for_tier,
    _make_escalated_decision,
    _resolve_escalation_target,
    _should_escalate_routed_response,
    record_escalation,
)
from byte.adapter.pipeline.verifier import (
    _assess_and_repair_response,
    _repair_cached_answer,
    _run_verifier_model_async,
    _run_verifier_model_sync,
    _should_run_verifier_model,
    _verification_bands,
)

__all__ = [
    "_assess_and_repair_response",
    "_escalation_action_for_tier",
    "_make_escalated_decision",
    "_repair_cached_answer",
    "_resolve_escalation_target",
    "_run_cheap_consensus_async",
    "_run_cheap_consensus_sync",
    "_run_verifier_model_async",
    "_run_verifier_model_sync",
    "_should_attempt_cheap_consensus",
    "_should_escalate_routed_response",
    "_should_run_verifier_model",
    "_verification_bands",
    "record_escalation",
]
