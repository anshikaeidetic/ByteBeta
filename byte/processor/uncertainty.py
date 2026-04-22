from dataclasses import asdict, dataclass, field
from typing import Any

from byte.processor.intent import extract_request_intent
from byte.processor.route_signals import extract_route_signals

_STRUCTURED_CATEGORIES = {
    "classification",
    "extraction",
    "translation",
    "exact_answer",
    "summarization",
}

_HARD_CATEGORIES = {
    "code_fix",
    "code_refactor",
    "test_generation",
    "comparison",
}


@dataclass(frozen=True)
class UncertaintyAssessment:
    score: float
    band: str
    recommended_context_chars: int
    structured: bool
    requires_consensus: bool
    reasons: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_request_uncertainty(
    request_kwargs: dict[str, Any] | None,
    config: Any,
    *,
    failure_hint: dict[str, Any] | None = None,
) -> UncertaintyAssessment:
    request_kwargs = request_kwargs or {}
    failure_hint = dict(failure_hint or {})
    intent = extract_request_intent(request_kwargs)
    signals = extract_route_signals(
        request_kwargs,
        long_prompt_chars=max(1, int(getattr(config, "routing_long_prompt_chars", 1200) or 1200)),
        multi_turn_threshold=max(1, int(getattr(config, "routing_multi_turn_threshold", 6) or 6)),
    )

    reasons: dict[str, float] = {}
    score = 0.0

    if intent.category in _HARD_CATEGORIES:
        reasons["hard_category"] = 0.28
    elif intent.category in _STRUCTURED_CATEGORIES:
        reasons["structured_task"] = 0.1

    if signals.needs_reasoning:
        reasons["reasoning"] = 0.22
    if signals.factual_risk:
        reasons["factual"] = 0.14
    if signals.has_multimodal_input:
        reasons["multimodal"] = 0.12
    if signals.jailbreak_risk or signals.pii_risk:
        reasons["policy_risk"] = 0.2

    label_count = _slot_count(intent.slots.get("labels"))
    field_count = _slot_count(intent.slots.get("fields"))
    if label_count > 0:
        reasons["label_breadth"] = min(0.18, max(0.0, (label_count - 3) * 0.03))
    if field_count > 0:
        reasons["field_breadth"] = min(0.18, max(0.0, (field_count - 3) * 0.03))

    if failure_hint.get("prefer_expensive"):
        reasons["historical_failures"] = 0.16
    if failure_hint.get("prefer_tool_context"):
        reasons["tool_context_needed"] = 0.08
    if failure_hint.get("clarify_first"):
        reasons["clarify_first"] = 0.12

    score = max(0.0, min(1.0, round(sum(reasons.values()), 4)))

    if score >= 0.67:
        band = "high"
        recommended_context_chars = int(
            getattr(config, "context_budget_high_risk_chars", 7600) or 7600
        )
    elif score >= 0.34:
        band = "medium"
        recommended_context_chars = int(
            getattr(config, "context_budget_medium_risk_chars", 4800) or 4800
        )
    else:
        band = "low"
        recommended_context_chars = int(
            getattr(config, "context_budget_low_risk_chars", 2200) or 2200
        )

    structured = intent.category in _STRUCTURED_CATEGORIES
    requires_consensus = bool(
        getattr(config, "cheap_consensus_enabled", False)
        and structured
        and band in {"medium", "high"}
    )

    return UncertaintyAssessment(
        score=score,
        band=band,
        recommended_context_chars=recommended_context_chars,
        structured=structured,
        requires_consensus=requires_consensus,
        reasons=reasons,
    )


def _slot_count(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (list, tuple, set)):
        return len([item for item in value if str(item).strip()])
    if isinstance(value, str):
        return len([item for item in value.split("|") if item.strip()])
    return 1
