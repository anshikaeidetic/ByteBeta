from typing import Any

from byte.processor.model_router import (
    ModelRouteDecision,
)

from .verifier import _verification_bands


def _should_escalate_routed_response(
    chat_cache, route_decision, assessment, context, task_policy=None
) -> bool:
    if route_decision is None or assessment is None:
        return False
    if assessment.reason == "verifier_model_rejected":
        return route_decision.tier in {"cheap", "coder", "reasoning"}
    if route_decision.tier == "cheap":
        if not getattr(chat_cache.config, "routing_verify_cheap_responses", True):
            return False
        _, threshold = _verification_bands(
            chat_cache,
            route_decision,
            assessment,
            context,
            task_policy=task_policy,
        )
        if assessment.constraint == "freeform":
            return False
        return (not assessment.accepted) or assessment.score < threshold
    if route_decision.tier not in {"coder", "reasoning"}:
        return False
    _, threshold = _verification_bands(
        chat_cache,
        route_decision,
        assessment,
        context,
        task_policy=task_policy,
    )
    if not assessment.accepted:
        return True
    if assessment.constraint == "freeform":
        return False
    return assessment.score < threshold


def _pick_escalation_target(current_model: str, *candidates: tuple) -> tuple:
    for model_name, tier in candidates:
        if model_name and model_name != current_model:
            return model_name, tier
    return "", ""


def _resolve_escalation_target(chat_cache, route_decision, request_kwargs) -> tuple:
    current_model = str(
        request_kwargs.get("model", "") or getattr(route_decision, "selected_model", "") or ""
    )
    category = str(getattr(route_decision, "category", "") or "")
    signals = dict(getattr(route_decision, "signals", {}) or {})
    coder_model = str(getattr(chat_cache.config, "routing_coder_model", "") or "")
    reasoning_model = str(getattr(chat_cache.config, "routing_reasoning_model", "") or "")
    expensive_model = str(getattr(chat_cache.config, "routing_expensive_model", "") or "")
    default_model = str(getattr(chat_cache.config, "routing_default_model", "") or "")

    if route_decision.tier == "cheap":
        if category in {"code_fix", "code_refactor", "test_generation", "documentation"}:
            return _pick_escalation_target(
                current_model,
                (coder_model, "coder"),
                (reasoning_model, "reasoning"),
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
        if signals.get("needs_reasoning"):
            return _pick_escalation_target(
                current_model,
                (reasoning_model, "reasoning"),
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
        return _pick_escalation_target(
            current_model,
            (expensive_model, "expensive"),
            (default_model, "default"),
        )

    if route_decision.tier == "coder":
        if signals.get("needs_reasoning"):
            return _pick_escalation_target(
                current_model,
                (reasoning_model, "reasoning"),
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
        return _pick_escalation_target(
            current_model,
            (expensive_model, "expensive"),
            (default_model, "default"),
        )

    if route_decision.tier == "reasoning":
        return _pick_escalation_target(
            current_model,
            (expensive_model, "expensive"),
            (default_model, "default"),
        )

    return "", ""


def _escalation_action_for_tier(tier: str) -> str:
    return {
        "coder": "direct_coder",
        "reasoning": "direct_reasoning",
        "expensive": "direct_expensive",
        "tool": "tool_first",
    }.get(str(tier or ""), "direct_expensive")


def _make_escalated_decision(route_decision, escalated_model: str, tier: str, reason: str) -> Any:
    return ModelRouteDecision(
        original_model=route_decision.original_model,
        selected_model=escalated_model,
        tier=tier,
        reason=reason,
        category=route_decision.category,
        route_key=route_decision.route_key,
        prompt_chars=route_decision.prompt_chars,
        message_count=route_decision.message_count,
        has_tools=route_decision.has_tools,
        applied=bool(escalated_model and escalated_model != route_decision.original_model),
        signals=dict(route_decision.signals or {}),
    )


__all__ = [name for name in globals() if not name.startswith("__")]
