
"""Model-routing policy selection."""

from __future__ import annotations

from typing import Any

from byte.processor._model_router_signals import (
    _CHEAP_CATEGORIES,
    _CODER_CATEGORIES,
    _is_hard_request,
    _pick_route_target,
    _request_size,
    _slot_count,
)
from byte.processor._model_router_tracker import (
    _cheap_latency_beats_expensive,
    _meets_budget_quality_floor,
    _route_tracker,
)
from byte.processor._model_router_types import ModelRouteDecision
from byte.processor.intent import extract_request_intent
from byte.processor.quality import extract_output_contract
from byte.processor.route_signals import extract_route_signals
from byte.processor.task_policy import resolve_task_policy


def route_request_model(
    request_kwargs: dict[str, Any] | None,
    config: Any,
) -> ModelRouteDecision | None:
    request_kwargs = request_kwargs or {}
    if not getattr(config, "model_routing", False):
        return None

    forced_model = str(request_kwargs.pop("byte_route_model", "") or "")
    disable_routing = bool(request_kwargs.pop("byte_disable_routing", False))
    original_model = str(request_kwargs.get("model", "") or "")
    if disable_routing:
        return ModelRouteDecision(
            original_model=original_model,
            selected_model=original_model,
            tier="disabled",
            reason="routing_disabled",
            category="",
            route_key="",
            prompt_chars=0,
            message_count=0,
            has_tools=False,
            applied=False,
            signals={},
        )

    if forced_model:
        applied = forced_model != original_model
        if applied:
            request_kwargs["model"] = forced_model
        intent = extract_request_intent(request_kwargs)
        prompt_chars, message_count = _request_size(request_kwargs)
        signals = extract_route_signals(
            request_kwargs,
            long_prompt_chars=max(
                1, int(getattr(config, "routing_long_prompt_chars", 1200) or 1200)
            ),
            multi_turn_threshold=max(
                1, int(getattr(config, "routing_multi_turn_threshold", 6) or 6)
            ),
        )
        return ModelRouteDecision(
            original_model=original_model,
            selected_model=forced_model,
            tier="forced",
            reason="explicit_override",
            category=intent.category,
            route_key=intent.route_key,
            prompt_chars=prompt_chars,
            message_count=message_count,
            has_tools=bool(intent.tool_signature),
            applied=applied,
            signals=signals.to_dict(),
        )

    intent = extract_request_intent(request_kwargs)
    task_policy = resolve_task_policy(request_kwargs, config)
    prompt_chars, message_count = _request_size(request_kwargs)
    signals = extract_route_signals(
        request_kwargs,
        long_prompt_chars=max(1, int(getattr(config, "routing_long_prompt_chars", 1200) or 1200)),
        multi_turn_threshold=max(1, int(getattr(config, "routing_multi_turn_threshold", 6) or 6)),
    )
    output_contract = extract_output_contract(request_kwargs)
    has_tools = bool(intent.tool_signature)
    cheap_model = str(getattr(config, "routing_cheap_model", "") or "")
    expensive_model = str(getattr(config, "routing_expensive_model", "") or "")
    tool_model = str(getattr(config, "routing_tool_model", "") or "")
    coder_model = str(getattr(config, "routing_coder_model", "") or "")
    reasoning_model = str(getattr(config, "routing_reasoning_model", "") or "")
    default_model = str(getattr(config, "routing_default_model", "") or original_model)
    long_prompt_chars = max(1, int(getattr(config, "routing_long_prompt_chars", 1200) or 1200))
    multi_turn_threshold = max(1, int(getattr(config, "routing_multi_turn_threshold", 6) or 6))

    selected_model = default_model
    tier = "default"
    reason = "default_model"
    route_preference = str(request_kwargs.pop("byte_route_preference", "") or "")
    budget_strategy = str(getattr(config, "budget_strategy", "balanced") or "balanced")

    if route_preference == "expensive" and (expensive_model or default_model):
        selected_model, tier = _pick_route_target(
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "workflow_preference_expensive"
    elif route_preference == "reasoning" and (reasoning_model or expensive_model or default_model):
        selected_model, tier = _pick_route_target(
            (reasoning_model, "reasoning"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "workflow_preference_reasoning"
    elif route_preference == "coder" and (
        coder_model or reasoning_model or expensive_model or default_model
    ):
        selected_model, tier = _pick_route_target(
            (coder_model, "coder"),
            (reasoning_model, "reasoning"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "workflow_preference_coder"
    elif route_preference == "tool" and (tool_model or expensive_model or default_model):
        selected_model, tier = _pick_route_target(
            (tool_model, "tool"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "workflow_preference_tool"
    elif route_preference == "cheap" and (cheap_model or default_model):
        selected_model, tier = _pick_route_target(
            (cheap_model, "cheap"),
            (default_model, "default"),
        )
        reason = "workflow_preference_cheap"

    if (
        tier == "default"
        and signals.recommended_route == "tool"
        and (tool_model or expensive_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (tool_model, "tool"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "signal_tool_request"
    elif (
        tier == "default"
        and signals.recommended_route == "expensive"
        and (expensive_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        if signals.jailbreak_risk:
            reason = "signal_jailbreak_guard"
        elif signals.pii_risk:
            reason = "signal_pii_guard"
        elif signals.has_multimodal_input:
            reason = "signal_multimodal_request"
        elif signals.factual_risk:
            reason = "signal_factual_request"
        else:
            reason = "signal_complex_request"
    elif tier == "default" and has_tools and (tool_model or expensive_model or default_model):
        selected_model, tier = _pick_route_target(
            (tool_model, "tool"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "tool_or_function_request"
    elif (
        tier == "default"
        and _is_hard_request(
            request_kwargs,
            category=intent.category,
            prompt_chars=prompt_chars,
            message_count=message_count,
            long_prompt_chars=long_prompt_chars,
            multi_turn_threshold=multi_turn_threshold,
        )
        and (coder_model or reasoning_model or expensive_model or default_model)
    ):
        if intent.category in _CODER_CATEGORIES:
            selected_model, tier = _pick_route_target(
                (coder_model, "coder"),
                (reasoning_model, "reasoning"),
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
        else:
            selected_model, tier = _pick_route_target(
                (reasoning_model, "reasoning"),
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
        reason = "complex_or_long_request"
    elif (
        tier == "default"
        and intent.category == "classification"
        and _slot_count(intent.slots.get("labels"))
        > int(getattr(config, "routing_max_cheap_labels", 6) or 6)
        and (expensive_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "large_label_space"
    elif (
        tier == "default"
        and intent.category == "extraction"
        and _slot_count(intent.slots.get("fields"))
        > int(getattr(config, "routing_max_cheap_fields", 6) or 6)
        and (expensive_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "wide_extraction_schema"
    elif (
        tier == "default"
        and intent.category == "code_explanation"
        and str(intent.slots.get("style") or "") == "complexity"
        and (reasoning_model or expensive_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (reasoning_model, "reasoning"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "code_complexity_accuracy_priority"
    elif (
        tier == "default"
        and intent.category == "documentation"
        and output_contract.strict
        and (coder_model or expensive_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (coder_model, "coder"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "documentation_contract_accuracy_priority"
    elif (
        tier == "default"
        and signals.recommended_route == "coder"
        and (coder_model or reasoning_model or expensive_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (coder_model, "coder"),
            (reasoning_model, "reasoning"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "signal_coding_request"
    elif (
        tier == "default"
        and signals.recommended_route == "reasoning"
        and (reasoning_model or expensive_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (reasoning_model, "reasoning"),
            (expensive_model, "expensive"),
            (default_model, "default"),
        )
        reason = "signal_reasoning_request"
    elif (
        tier == "default"
        and signals.recommended_route == "cheap"
        and (cheap_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (cheap_model, "cheap"),
            (default_model, "default"),
        )
        reason = "signal_low_cost_request"
    elif (
        tier == "default"
        and intent.category in _CHEAP_CATEGORIES
        and (cheap_model or default_model)
    ):
        selected_model, tier = _pick_route_target(
            (cheap_model, "cheap"),
            (default_model, "default"),
        )
        reason = f"{intent.category}_request"
    elif tier == "default" and intent.category == "summarization":
        if prompt_chars <= (long_prompt_chars // 2) and (cheap_model or default_model):
            selected_model, tier = _pick_route_target(
                (cheap_model, "cheap"),
                (default_model, "default"),
            )
            reason = "short_summarization"
        elif expensive_model or default_model:
            selected_model, tier = _pick_route_target(
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
            reason = "long_summarization"
    elif (
        tier == "default"
        and cheap_model
        and message_count <= 2
        and prompt_chars < max(256, long_prompt_chars // 3)
    ):
        selected_model = cheap_model
        tier = "cheap"
        reason = "short_single_turn_request"

    if tier == "default":
        policy_bias = str(task_policy.get("route_bias", "") or "")
        if policy_bias == "tool" and (tool_model or expensive_model or default_model):
            selected_model, tier = _pick_route_target(
                (tool_model, "tool"),
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
            reason = "task_policy_tool"
        elif policy_bias == "coder" and (
            coder_model or reasoning_model or expensive_model or default_model
        ):
            selected_model, tier = _pick_route_target(
                (coder_model, "coder"),
                (reasoning_model, "reasoning"),
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
            reason = "task_policy_coder"
        elif policy_bias == "reasoning" and (reasoning_model or expensive_model or default_model):
            selected_model, tier = _pick_route_target(
                (reasoning_model, "reasoning"),
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
            reason = "task_policy_reasoning"
        elif policy_bias == "expensive" and (expensive_model or default_model):
            selected_model, tier = _pick_route_target(
                (expensive_model, "expensive"),
                (default_model, "default"),
            )
            reason = "task_policy_expensive"
        elif policy_bias == "cheap" and (cheap_model or default_model):
            selected_model, tier = _pick_route_target(
                (cheap_model, "cheap"),
                (default_model, "default"),
            )
            reason = "task_policy_cheap"

    if (
        budget_strategy == "lowest_cost"
        and cheap_model
        and _meets_budget_quality_floor(intent.route_key, config)
    ):
        selected_model = cheap_model
        tier = "cheap"
        reason = "budget_lowest_cost"
    elif budget_strategy == "quality_first" and (expensive_model or default_model):
        selected_model = expensive_model or default_model
        tier = "expensive"
        reason = "budget_quality_first"
    elif (
        budget_strategy == "low_latency"
        and cheap_model
        and _cheap_latency_beats_expensive(intent.route_key)
    ):
        selected_model = cheap_model
        tier = "cheap"
        reason = "budget_low_latency"

    if (
        tier == "cheap"
        and expensive_model
        and _route_tracker.prefer_expensive(intent.route_key, config)
    ):
        selected_model = expensive_model
        tier = "expensive"
        reason = "adaptive_quality_guard"

    if not selected_model:
        selected_model = original_model
        tier = "original"
        reason = "no_route_target_available"

    applied = bool(selected_model and selected_model != original_model)
    if applied:
        request_kwargs["model"] = selected_model

    return ModelRouteDecision(
        original_model=original_model,
        selected_model=selected_model or original_model,
        tier=tier,
        reason=reason,
        category=intent.category,
        route_key=intent.route_key,
        prompt_chars=prompt_chars,
        message_count=message_count,
        has_tools=has_tools,
        applied=applied,
        signals=signals.to_dict(),
    )

__all__ = ["route_request_model"]
