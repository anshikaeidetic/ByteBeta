import re
from dataclasses import asdict, dataclass, field
from typing import Any

from byte.processor.intent import IntentRecord, extract_request_intent
from byte.processor.pre import normalize_text
from byte.processor.route_signals import extract_route_signals

_DEICTIC_MARKERS = (
    "this",
    "that",
    "it",
    "same issue",
    "same bug",
    "same error",
    "nearby file",
    "above",
    "below",
    "tell me more",
)

_SOURCE_CONTEXT_FIELDS = (
    "byte_retrieval_context",
    "byte_document_context",
    "byte_support_articles",
    "byte_tool_result_context",
    "byte_repo_summary",
    "byte_repo_snapshot",
    "byte_changed_files",
    "byte_changed_hunks",
    "byte_prompt_pieces",
    "byte_repo_fingerprint",
    "byte_workspace_fingerprint",
    "byte_codebase_fingerprint",
)

_SOURCE_CONTEXT_HINTS = (
    "according to",
    "based on",
    "from the docs",
    "from the doc",
    "from the document",
    "from the article",
    "from the report",
    "from the policy",
    "from the ticket",
    "from the clause",
    "from the text",
    "support article",
    "attached",
    "following",
    "below",
    "above",
)

_CODE_CATEGORIES = {
    "code_fix",
    "code_refactor",
    "test_generation",
    "code_explanation",
    "documentation",
}

_CODE_KEYWORDS = {"fix", "bug", "refactor", "docstring", "code", "patch"}
_CODE_TEST_HINTS = (
    "pytest",
    "unittest",
    "unit test",
    "unit tests",
    "integration test",
    "integration tests",
    "test case",
    "test cases",
)

_CLASSIFICATION_LABEL_SET_PATTERN = re.compile(r"(?is)\{[^{}]*[,|][^{}]*\}")


@dataclass(frozen=True)
class AmbiguityAssessment:
    ambiguous: bool
    reason: str
    question: str
    category: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowDecision:
    action: str
    reason: str
    route_preference: str = ""
    response_text: str = ""
    planner_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["planner_hints"] = dict(self.planner_hints or {})
        return payload


def detect_ambiguity(
    request_kwargs: dict[str, Any] | None,
    *,
    min_chars: int = 24,
    context_hints: dict[str, Any] | None = None,
) -> AmbiguityAssessment:
    request_kwargs = request_kwargs or {}
    intent = extract_request_intent(request_kwargs)
    request_text = _extract_request_text(request_kwargs)
    normalized = normalize_text(request_text)
    words = set(normalized.split())
    message_count = len(request_kwargs.get("messages") or [])
    has_code = "```" in request_text
    has_source_context = has_request_source_context(request_kwargs, context_hints=context_hints)
    requests_source_context = request_requests_source_context(request_kwargs, intent=intent)
    has_structured_context = (
        has_code
        or has_source_context
        or any(token in normalized for token in ("labels", "fields", "keys", "selection"))
    )

    if intent.category in {"exact_answer", "translation"}:
        return AmbiguityAssessment(
            ambiguous=False,
            reason="clear_enough",
            question="",
            category=intent.category,
            score=0.0,
        )

    if requests_source_context and not has_source_context:
        return _assessment(
            intent,
            "missing_source_context",
            _missing_source_question(intent.category),
            0.9,
        )

    code_like_request = (
        intent.category in _CODE_CATEGORIES
        or bool(words & _CODE_KEYWORDS)
        or any(hint in normalized for hint in _CODE_TEST_HINTS)
        or "selected code" in normalized
    )

    if code_like_request and not has_structured_context:
        return _assessment(
            intent,
            "missing_code_context",
            "Which file or code snippet should I use for this coding task?",
            0.91,
        )

    if (
        len(normalized.split()) <= 3
        and message_count <= 1
        and not has_structured_context
        and any(_contains_marker(normalized, marker) for marker in _DEICTIC_MARKERS)
    ):
        return _assessment(intent, "too_short", "What exact task or input should I use?", 0.78)

    if (
        any(_contains_marker(normalized, marker) for marker in _DEICTIC_MARKERS)
        and message_count <= 1
        and not has_structured_context
        and len(words) <= 12
    ):
        return _assessment(intent, "missing_reference", _clarifying_question(intent.category), 0.84)

    if intent.category == "classification" and not _has_classification_labels(
        request_text, normalized
    ):
        return _assessment(
            intent,
            "missing_labels",
            "Which labels should I choose from for this classification task?",
            0.86,
        )

    if intent.category == "classification" and not has_source_context and message_count <= 1:
        return _assessment(
            intent,
            "missing_classification_input",
            "What text, ticket, or review should I classify?",
            0.82,
        )

    if intent.category == "extraction" and not any(
        token in normalized for token in ("field", "fields", "keys")
    ):
        return _assessment(
            intent,
            "missing_fields",
            "Which fields or keys should I extract?",
            0.88,
        )

    return AmbiguityAssessment(
        ambiguous=False,
        reason="clear_enough",
        question="",
        category=intent.category,
        score=0.0,
    )


def plan_request_workflow(
    request_kwargs: dict[str, Any] | None,
    config: Any,
    *,
    ambiguity: AmbiguityAssessment | None = None,
    failure_hint: dict[str, Any] | None = None,
    global_hint: dict[str, Any] | None = None,
    patch_candidate: dict[str, Any] | None = None,
) -> WorkflowDecision:
    request_kwargs = request_kwargs or {}
    intent = extract_request_intent(request_kwargs)
    signals = extract_route_signals(
        request_kwargs,
        long_prompt_chars=max(1, int(getattr(config, "routing_long_prompt_chars", 1200) or 1200)),
        multi_turn_threshold=max(1, int(getattr(config, "routing_multi_turn_threshold", 6) or 6)),
    )
    ambiguity = ambiguity or detect_ambiguity(
        request_kwargs,
        min_chars=getattr(config, "ambiguity_min_chars", 24),
    )
    failure_hint = dict(failure_hint or {})
    global_hint = dict(global_hint or {})
    if ambiguity.ambiguous and getattr(config, "ambiguity_detection", True):
        return WorkflowDecision(
            action="clarify",
            reason=ambiguity.reason,
            response_text=ambiguity.question,
            planner_hints={"ambiguity": ambiguity.to_dict()},
        )

    preferred_action = str(global_hint.get("preferred_action", "") or "")
    preferred_route = str(global_hint.get("route_preference", "") or "")
    avoid_action = str(global_hint.get("avoid_action", "") or "")
    counterfactual_action = str(global_hint.get("counterfactual_action", "") or "")
    if preferred_action == "reuse_verified_patch" and patch_candidate:
        return WorkflowDecision(
            action="reuse_verified_patch",
            reason="historical_workflow_patch_reuse",
            response_text=patch_candidate.get("patch_text", "")
            or patch_candidate.get("patched_code", ""),
            planner_hints={"patch_candidate": patch_candidate, "workflow_hint": global_hint},
        )
    if preferred_action == "tool_first" or preferred_route == "tool":
        return WorkflowDecision(
            action="tool_first",
            reason="historical_workflow_preference",
            route_preference="tool",
            planner_hints={"workflow_hint": global_hint},
        )
    if preferred_action == "direct_coder" or preferred_route == "coder":
        return WorkflowDecision(
            action="direct_coder",
            reason="historical_workflow_preference",
            route_preference="coder",
            planner_hints={"workflow_hint": global_hint},
        )
    if preferred_action == "direct_reasoning" or preferred_route == "reasoning":
        return WorkflowDecision(
            action="direct_reasoning",
            reason="historical_workflow_preference",
            route_preference="reasoning",
            planner_hints={"workflow_hint": global_hint},
        )
    if preferred_action == "direct_expensive" or preferred_route == "expensive":
        return WorkflowDecision(
            action="direct_expensive",
            reason="historical_workflow_preference",
            route_preference="expensive",
            planner_hints={"workflow_hint": global_hint},
        )
    if preferred_action == "cheap_then_verify" or preferred_route == "cheap":
        return WorkflowDecision(
            action="cheap_then_verify",
            reason="historical_workflow_preference",
            route_preference="cheap",
            planner_hints={"workflow_hint": global_hint},
        )
    if avoid_action == "cheap_then_verify" and counterfactual_action in {
        "direct_expensive",
        "tool_first",
        "direct_coder",
        "direct_reasoning",
    }:
        route_preference = {
            "direct_expensive": "expensive",
            "tool_first": "tool",
            "direct_coder": "coder",
            "direct_reasoning": "reasoning",
        }.get(counterfactual_action, "")
        return WorkflowDecision(
            action=counterfactual_action,
            reason="historical_counterfactual_preference",
            route_preference=route_preference,
            planner_hints={"workflow_hint": global_hint},
        )

    if signals.jailbreak_risk or signals.pii_risk:
        return WorkflowDecision(
            action="direct_expensive",
            reason="risk_guard",
            route_preference="expensive",
            planner_hints={"signals": signals.to_dict()},
        )

    if (
        patch_candidate
        and getattr(config, "delta_generation", True)
        and getattr(config, "planner_allow_verified_short_circuit", True)
        and request_kwargs.get("byte_allow_patch_reuse")
    ):
        return WorkflowDecision(
            action="reuse_verified_patch",
            reason="verified_patch_pattern_available",
            response_text=patch_candidate.get("patch_text", "")
            or patch_candidate.get("patched_code", ""),
            planner_hints={"patch_candidate": patch_candidate},
        )

    if signals.factual_risk and (request_kwargs.get("tools") or request_kwargs.get("functions")):
        return WorkflowDecision(
            action="tool_first",
            reason="factual_tool_preference",
            route_preference="tool",
            planner_hints={"signals": signals.to_dict()},
        )

    if failure_hint.get("prefer_tool_context") or global_hint.get("prefer_tool_context"):
        return WorkflowDecision(
            action="tool_first",
            reason="historical_tool_context_needed",
            route_preference="tool",
        )

    if failure_hint.get("prefer_expensive"):
        return WorkflowDecision(
            action="direct_expensive",
            reason="historical_quality_guard",
            route_preference="expensive",
        )

    if intent.category in _CODE_CATEGORIES:
        strategy = str(getattr(config, "budget_strategy", "balanced") or "balanced")
        if strategy == "lowest_cost":
            return WorkflowDecision(
                action="cheap_then_verify",
                reason="coding_task_cost_optimized",
                route_preference="cheap",
            )
        return WorkflowDecision(
            action="direct_coder",
            reason="coding_task_specialized_coder",
            route_preference="coder",
            planner_hints={"signals": signals.to_dict()},
        )

    if signals.recommended_route == "coder":
        return WorkflowDecision(
            action="route",
            reason="signal_coding_request",
            route_preference="coder",
            planner_hints={"signals": signals.to_dict()},
        )

    if signals.recommended_route == "reasoning":
        return WorkflowDecision(
            action="route",
            reason="signal_reasoning_request",
            route_preference="reasoning",
            planner_hints={"signals": signals.to_dict()},
        )

    if signals.recommended_route == "cheap":
        return WorkflowDecision(
            action="route",
            reason="signal_low_cost_request",
            route_preference="cheap",
            planner_hints={"signals": signals.to_dict()},
        )

    return WorkflowDecision(
        action="route",
        reason="default_workflow",
        route_preference="",
        planner_hints={"category": intent.category, "signals": signals.to_dict()},
    )


def _assessment(
    intent: IntentRecord, reason: str, question: str, score: float
) -> AmbiguityAssessment:
    return AmbiguityAssessment(
        ambiguous=True,
        reason=reason,
        question=question,
        category=intent.category,
        score=score,
    )


def _clarifying_question(category: str) -> str:
    if category in _CODE_CATEGORIES:
        return "Which file, code snippet, or repo context should I use?"
    if category == "summarization":
        return "Which text should I summarize?"
    if category == "classification":
        return "What are the allowed labels for this classification?"
    if category == "extraction":
        return "Which fields should I extract from the input?"
    return "Could you clarify the exact input or target for this request?"


def request_requests_source_context(
    request_kwargs: dict[str, Any] | None,
    *,
    intent: IntentRecord | None = None,
) -> bool:
    request_kwargs = request_kwargs or {}
    intent = intent or extract_request_intent(request_kwargs)
    normalized = normalize_text(_extract_request_text(request_kwargs))
    if any(hint in normalized for hint in _SOURCE_CONTEXT_HINTS):
        return True
    if intent.category in {"summarization", "extraction", "comparison"} and any(
        token in normalized
        for token in ("article", "document", "report", "clause", "text", "content")
    ):
        return True
    if intent.category == "question_answer" and any(
        token in normalized
        for token in ("document", "article", "policy", "report", "ticket", "support")
    ):
        return True
    return False


def has_request_source_context(
    request_kwargs: dict[str, Any] | None,
    *,
    context_hints: dict[str, Any] | None = None,
) -> bool:
    request_kwargs = request_kwargs or {}
    raw_context = dict(context_hints or {})
    if "_byte_raw_aux_context" in raw_context and isinstance(
        raw_context["_byte_raw_aux_context"], dict
    ):
        raw_context = dict(raw_context["_byte_raw_aux_context"])
    if any(request_kwargs.get(field) not in (None, "", [], {}) for field in _SOURCE_CONTEXT_FIELDS):
        return True
    if any(raw_context.get(field) not in (None, "", [], {}) for field in _SOURCE_CONTEXT_FIELDS):
        return True
    request_text = _extract_request_text(request_kwargs)
    if "```" in request_text:
        return True
    if re.search(
        r'(?is)(?:ticket|review|article|document|report|clause|text|content)\s*:\s*["\']?.{8,}',
        request_text,
    ):
        return True
    if re.search(r'(?is)"[^"]{8,}"', request_text):
        return True
    if len(re.findall(r"(?m)^\s*[-*]\s+\S.{6,}$", request_text)) >= 2:
        return True
    if re.search(
        r"(?is)\b(?:notes|drafting notes|source material|context)\s*:\s*(?:.+\n){2,}", request_text
    ):
        return True
    if re.search(r"(?is)\b(?:file|path|selection|diagnostic)\s*:", request_text):
        return True
    if _has_compiled_source_context_hints(request_text):
        return True
    return False


def _has_classification_labels(request_text: str, normalized: str) -> bool:
    if any(token in normalized for token in ("labels", "classes", "categories")):
        return True
    return bool(_CLASSIFICATION_LABEL_SET_PATTERN.search(request_text or ""))


def _missing_source_question(category: str) -> str:
    if category == "summarization":
        return "Which article, document, or text should I summarize?"
    if category == "extraction":
        return "Which document or text should I extract from?"
    if category == "comparison":
        return "Which two items or sources should I compare?"
    if category == "question_answer":
        return "Which document, article, or support context should I answer from?"
    return "Which source material should I use?"


def _extract_request_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            return " ".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt"))
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input"))
    return ""


def _has_compiled_source_context_hints(request_text: str) -> bool:
    text = str(request_text or "")
    if not text:
        return False
    markers = (
        "byte session delta",
        "repo summary",
        "repository summary",
        "repo snapshot",
        "repository snapshot",
        "changed files",
        "changed hunks",
        "prompt pieces",
        "support context",
        "document context",
        "tool result context",
        "retrieval context",
    )
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _contains_marker(normalized: str, marker: str) -> bool:
    normalized = f" {str(normalized or '').strip().lower()} "
    marker = str(marker or "").strip().lower()
    if not marker:
        return False
    return f" {marker} " in normalized
