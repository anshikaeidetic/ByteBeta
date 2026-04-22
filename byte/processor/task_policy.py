from copy import deepcopy
from typing import Any

from byte.processor.intent import extract_request_intent
from byte.processor.route_signals import extract_route_signals
from byte.processor.workflow import has_request_source_context, request_requests_source_context

_DEFAULT_TASK_POLICIES: dict[str, dict[str, Any]] = {
    "classification": {
        "route_bias": "cheap",
        "verify_min_score": 0.85,
        "cache_admission_min_score": 0.6,
        "context_max_chars": 3200,
        "native_prompt_cache": True,
    },
    "translation": {
        "route_bias": "cheap",
        "verify_min_score": 0.85,
        "cache_admission_min_score": 0.7,
        "context_max_chars": 3200,
        "native_prompt_cache": True,
    },
    "exact_answer": {
        "route_bias": "cheap",
        "verify_min_score": 0.9,
        "cache_admission_min_score": 0.8,
        "context_max_chars": 2400,
        "native_prompt_cache": True,
    },
    "extraction": {
        "route_bias": "cheap",
        "verify_min_score": 0.88,
        "cache_admission_min_score": 0.7,
        "context_max_chars": 4200,
        "evidence_required": True,
        "evidence_min_support": 0.78,
        "evidence_structured_min_support": 0.78,
        "native_prompt_cache": True,
    },
    "summarization": {
        "route_bias": "cheap",
        "verify_min_score": 0.78,
        "cache_admission_min_score": 0.45,
        "context_max_chars": 5600,
        "evidence_min_support": 0.28,
        "evidence_summary_min_support": 0.28,
        "native_prompt_cache": True,
    },
    "question_answer": {
        "route_bias": "",
        "verify_min_score": 0.75,
        "cache_admission_min_score": 0.4,
        "context_max_chars": 5200,
        "evidence_min_support": 0.35,
        "native_prompt_cache": True,
    },
    "documentation": {
        "route_bias": "coder",
        "verify_min_score": 0.75,
        "cache_admission_min_score": 0.45,
        "context_max_chars": 6200,
        "native_prompt_cache": True,
    },
    "code_explanation": {
        "route_bias": "cheap",
        "verify_min_score": 0.8,
        "cache_admission_min_score": 0.55,
        "context_max_chars": 6800,
        "native_prompt_cache": True,
    },
    "code_fix": {
        "route_bias": "coder",
        "verify_min_score": 0.9,
        "cache_admission_min_score": 0.8,
        "context_max_chars": 9000,
        "native_prompt_cache": True,
    },
    "code_refactor": {
        "route_bias": "coder",
        "verify_min_score": 0.88,
        "cache_admission_min_score": 0.7,
        "context_max_chars": 8600,
        "native_prompt_cache": True,
    },
    "test_generation": {
        "route_bias": "coder",
        "verify_min_score": 0.85,
        "cache_admission_min_score": 0.7,
        "context_max_chars": 8600,
        "native_prompt_cache": True,
    },
    "tool_call": {
        "route_bias": "tool",
        "verify_min_score": 0.8,
        "cache_admission_min_score": 0.55,
        "context_max_chars": 5200,
        "native_prompt_cache": False,
    },
}


def resolve_task_policy(
    request_kwargs: dict[str, Any] | None,
    config: Any,
    *,
    global_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_kwargs = request_kwargs or {}
    intent = extract_request_intent(request_kwargs)
    merged = deepcopy(_DEFAULT_TASK_POLICIES.get(intent.category, {}))
    custom = getattr(config, "task_policies", {}) or {}
    if isinstance(custom.get(intent.category), dict):
        merged.update(custom[intent.category])
    if isinstance(custom.get("*"), dict):
        base = dict(custom["*"])
        base.update(merged)
        merged = base

    signals = extract_route_signals(
        request_kwargs,
        long_prompt_chars=max(1, int(getattr(config, "routing_long_prompt_chars", 1200) or 1200)),
        multi_turn_threshold=max(1, int(getattr(config, "routing_multi_turn_threshold", 6) or 6)),
    )
    if intent.category in {"code_fix", "code_refactor", "test_generation", "documentation"}:
        merged["route_bias"] = "coder"
    elif signals.needs_reasoning:
        merged["route_bias"] = "reasoning"
    if signals.has_multimodal_input or signals.factual_risk:
        merged["route_bias"] = "expensive"
    if signals.jailbreak_risk or signals.pii_risk:
        merged["route_bias"] = "expensive"
        merged["cache_admission_min_score"] = max(
            float(merged.get("cache_admission_min_score", 0.0) or 0.0), 0.85
        )
    if intent.category == "classification":
        label_count = _slot_count(intent.slots.get("labels"))
        if label_count > int(getattr(config, "routing_max_cheap_labels", 6) or 6):
            merged["route_bias"] = "expensive"
            merged["verify_min_score"] = max(
                float(merged.get("verify_min_score", 0.0) or 0.0), 0.92
            )
            merged["cache_admission_min_score"] = max(
                float(merged.get("cache_admission_min_score", 0.0) or 0.0), 0.82
            )
    if intent.category == "extraction":
        field_count = _slot_count(intent.slots.get("fields"))
        if field_count > int(getattr(config, "routing_max_cheap_fields", 6) or 6):
            merged["route_bias"] = "expensive"
            merged["verify_min_score"] = max(float(merged.get("verify_min_score", 0.0) or 0.0), 0.9)
            merged["cache_admission_min_score"] = max(
                float(merged.get("cache_admission_min_score", 0.0) or 0.0), 0.8
            )
            merged["evidence_min_support"] = max(
                float(merged.get("evidence_min_support", 0.0) or 0.0), 0.82
            )
    if intent.category == "comparison":
        merged["route_bias"] = "expensive"
        merged["verify_min_score"] = max(float(merged.get("verify_min_score", 0.0) or 0.0), 0.84)
        merged["cache_admission_min_score"] = max(
            float(merged.get("cache_admission_min_score", 0.0) or 0.0), 0.68
        )
        merged["evidence_required"] = True
        merged["evidence_min_support"] = max(
            float(merged.get("evidence_min_support", 0.0) or 0.0), 0.38
        )
    requests_source = request_requests_source_context(request_kwargs, intent=intent)
    has_source = has_request_source_context(request_kwargs)
    if requests_source and not has_source:
        merged["clarify_first"] = True
        merged["verify_min_score"] = max(float(merged.get("verify_min_score", 0.0) or 0.0), 0.9)
    if requests_source and has_source:
        merged["evidence_required"] = True
        if intent.category in {"question_answer", "extraction", "comparison"}:
            merged["route_bias"] = "expensive"
            merged["verify_min_score"] = max(
                float(merged.get("verify_min_score", 0.0) or 0.0), 0.86
            )
            merged["cache_admission_min_score"] = max(
                float(merged.get("cache_admission_min_score", 0.0) or 0.0), 0.72
            )
        elif intent.category == "summarization":
            merged["verify_min_score"] = max(float(merged.get("verify_min_score", 0.0) or 0.0), 0.8)
            merged["cache_admission_min_score"] = max(
                float(merged.get("cache_admission_min_score", 0.0) or 0.0), 0.5
            )
    if global_hint:
        if global_hint.get("prefer_tool_context"):
            merged["route_bias"] = "tool"
        elif global_hint.get("prefer_expensive"):
            merged["route_bias"] = "expensive"
        if global_hint.get("clarify_first"):
            merged["clarify_first"] = True
    merged["category"] = intent.category
    merged["route_key"] = intent.route_key
    return merged


def _slot_count(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (list, tuple, set)):
        return len([item for item in value if str(item).strip()])
    if isinstance(value, str):
        return len([item for item in value.split("|") if item.strip()])
    return 1
