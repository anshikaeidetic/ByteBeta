"""Evidence and style validation helpers for quality scoring."""

import re
from typing import Any

from byte.processor._quality_models import EvidenceAssessment, ResponseAssessment
from byte.processor.optimization_memory import compact_text
from byte.processor.pre import normalize_text
from byte.processor.workflow import request_requests_source_context

_EVIDENCE_FIELDS = (
    "byte_retrieval_context",
    "byte_document_context",
    "byte_support_articles",
    "byte_tool_result_context",
    "byte_repo_summary",
    "byte_repo_snapshot",
    "byte_changed_files",
    "byte_changed_hunks",
)
_EVIDENCE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "their",
    "there",
    "have",
    "has",
    "been",
    "were",
    "was",
    "will",
    "would",
    "should",
    "could",
    "only",
    "exactly",
    "answer",
    "reply",
    "respond",
    "return",
    "label",
    "json",
    "yaml",
    "csv",
}

def _collect_evidence_text(
    request_kwargs: dict[str, Any],
    context_hints: dict[str, Any] | None,
) -> str:
    payloads: list[str] = []
    raw_context = dict(context_hints or {})
    if "_byte_raw_aux_context" in raw_context and isinstance(
        raw_context["_byte_raw_aux_context"], dict
    ):
        raw_context = dict(raw_context["_byte_raw_aux_context"])
    for field in _EVIDENCE_FIELDS:
        for source in (raw_context, request_kwargs):
            value = source.get(field)
            if value in (None, "", [], {}):
                continue
            text = _evidence_value_text(value)
            if text:
                payloads.append(text)
                break
    if not payloads:
        return ""
    return "\n".join(payloads)


def _evidence_value_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, list):
        parts = [_evidence_value_text(item) for item in value[:12]]
        return "\n".join(part for part in parts if part)
    return compact_text(value, max_chars=2400)


def _merge_evidence_assessment(
    base: ResponseAssessment,
    evidence: EvidenceAssessment | None,
    *,
    constraint: str | None = None,
) -> ResponseAssessment:
    if evidence is None:
        return base
    merged_constraint = constraint or base.constraint
    accepted = bool(base.accepted and evidence.accepted)
    score = round(min(float(base.score or 0.0), float(evidence.score or 0.0)), 4)
    reason = base.reason if accepted else evidence.reason
    return ResponseAssessment(
        score=score,
        accepted=accepted,
        repaired_answer=base.repaired_answer,
        reason=reason,
        constraint=merged_constraint,
    )


def _evidence_threshold(category: str, config: Any, task_policy: dict[str, Any] | None) -> float:
    if isinstance(task_policy, dict) and task_policy.get("evidence_min_support") is not None:
        return float(task_policy.get("evidence_min_support") or 0.0)
    if category == "summarization":
        return _summary_evidence_threshold(config, task_policy)
    if category == "extraction":
        return _structured_evidence_threshold(category, config, task_policy)
    if category in {"question_answer", "comparison"}:
        return float(getattr(config, "evidence_min_support", 0.35) or 0.35)
    return float(getattr(config, "evidence_min_support", 0.35) or 0.35)


def _structured_evidence_threshold(
    category: str, config: Any, task_policy: dict[str, Any] | None
) -> float:
    if (
        isinstance(task_policy, dict)
        and task_policy.get("evidence_structured_min_support") is not None
    ):
        return float(task_policy.get("evidence_structured_min_support") or 0.0)
    default = 0.78 if category == "extraction" else 0.72
    return float(getattr(config, "evidence_structured_min_support", default) or default)


def _summary_evidence_threshold(config: Any, task_policy: dict[str, Any] | None) -> float:
    if (
        isinstance(task_policy, dict)
        and task_policy.get("evidence_summary_min_support") is not None
    ):
        return float(task_policy.get("evidence_summary_min_support") or 0.0)
    return float(getattr(config, "evidence_summary_min_support", 0.28) or 0.28)


def _assess_structured_evidence(
    *,
    intent: Any,
    request_kwargs: dict[str, Any],
    request_text: str,
    parsed: Any,
    evidence_text: str,
    structured_format: str,
    min_support: float,
    config: Any,
    task_policy: dict[str, Any] | None,
) -> EvidenceAssessment | None:
    if not getattr(config, "evidence_verification", True):
        return None
    if not evidence_text:
        return None
    if not _should_require_evidence(intent, request_kwargs, request_text, task_policy):
        return None
    support = _structured_value_support(parsed, evidence_text, request_text)
    accepted = support >= min_support
    return EvidenceAssessment(
        score=round(max(0.05, min(0.98, support)), 4),
        accepted=accepted,
        reason=f"{structured_format}_grounded"
        if accepted
        else f"{structured_format}_unsupported_values",
        constraint=structured_format,
    )


def _assess_evidence_support(
    *,
    intent: Any,
    request_kwargs: dict[str, Any],
    request_text: str,
    answer_text: str,
    evidence_text: str,
    constraint: str,
    min_support: float,
    config: Any,
    task_policy: dict[str, Any] | None,
) -> EvidenceAssessment | None:
    if not getattr(config, "evidence_verification", True):
        return None
    if not evidence_text:
        return None
    if not _should_require_evidence(intent, request_kwargs, request_text, task_policy):
        return None
    if str(getattr(intent, "category", "") or "") in {"classification", "translation"}:
        return None
    support = _text_support_score(answer_text, evidence_text, request_text)
    accepted = support >= min_support
    reason = f"{constraint}_grounded" if accepted else f"{constraint}_unsupported"
    return EvidenceAssessment(
        score=round(max(0.05, min(0.98, support)), 4),
        accepted=accepted,
        reason=reason,
        constraint=constraint,
    )


def _text_support_score(answer_text: str, evidence_text: str, request_text: str) -> float:
    normalized_answer = normalize_text(answer_text)
    normalized_corpus = normalize_text(f"{evidence_text} {request_text}")
    if not normalized_answer or not normalized_corpus:
        return 0.0
    if len(normalized_answer) <= 96 and normalized_answer in normalized_corpus:
        return 0.99
    answer_tokens = _support_tokens(answer_text)
    if not answer_tokens:
        return 0.0
    corpus_tokens = _support_tokens(normalized_corpus)
    matches = sum(1 for token in answer_tokens if token in corpus_tokens)
    score = matches / max(1, len(answer_tokens))
    numeric_tokens = re.findall(r"\b[$]?\d[\d,./:-]*\b", answer_text)
    if numeric_tokens:
        numeric_matches = sum(
            1 for token in numeric_tokens if normalize_text(token) in normalized_corpus
        )
        numeric_ratio = numeric_matches / max(1, len(numeric_tokens))
        score = score * (0.2 + 0.8 * numeric_ratio)
    return score


def _structured_value_support(parsed: Any, evidence_text: str, request_text: str) -> float:
    corpus = normalize_text(f"{evidence_text} {request_text}")
    values = _structured_scalar_values(parsed)
    if not values:
        return 0.0
    matches = 0.0
    for value in values:
        normalized = normalize_text(value)
        if not normalized:
            continue
        if normalized in corpus:
            matches += 1.0
            continue
        score = _text_support_score(value, evidence_text, request_text)
        if score >= 0.7:
            matches += 1.0
        elif score >= 0.35:
            matches += 0.5
    return matches / max(1, len(values))


def _structured_scalar_values(parsed: Any) -> list[str]:
    values: list[str] = []
    if isinstance(parsed, dict):
        for value in parsed.values():
            values.extend(_structured_scalar_values(value))
        return values
    if isinstance(parsed, list):
        for item in parsed[:20]:
            values.extend(_structured_scalar_values(item))
        return values
    if isinstance(parsed, (str, int, float)):
        text = str(parsed).strip()
        if text:
            values.append(text)
    return values


def _support_tokens(value: Any) -> list[str]:
    text = normalize_text(value)
    return [
        _normalize_support_token(token)
        for token in re.findall(r"[a-z0-9_./:-]+", text)
        if len(token) >= 3
        and _normalize_support_token(token)
        and _normalize_support_token(token) not in _EVIDENCE_STOPWORDS
    ]


def _normalize_support_token(token: str) -> str:
    token = str(token or "").strip().lower()
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif (len(token) > 4 and token.endswith("ed")) or (len(token) > 4 and token.endswith("es")):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token.strip("_-./:")


def _should_require_evidence(
    intent: Any,
    request_kwargs: dict[str, Any],
    request_text: str,
    task_policy: dict[str, Any] | None,
) -> bool:
    if isinstance(task_policy, dict):
        explicit = task_policy.get("evidence_required")
        if explicit is not None:
            return bool(explicit)
    category = str(getattr(intent, "category", "") or "")
    if category in {"extraction", "comparison"}:
        return True
    if request_requests_source_context(request_kwargs, intent=intent):
        return True
    if category == "question_answer" and any(
        token in normalize_text(request_text)
        for token in ("document", "article", "support", "policy", "report")
    ):
        return True
    return False


def _assess_style_constraint(intent, answer_text: str) -> ResponseAssessment | None:
    slots = getattr(intent, "slots", {}) or {}
    category = str(getattr(intent, "category", "") or "")
    style = str(slots.get("style") or "")
    if category != "summarization" or not style:
        return None

    if style == "one_sentence":
        sentence_count = _sentence_count(answer_text)
        accepted = sentence_count <= 1
        return ResponseAssessment(
            score=0.88 if accepted else 0.58,
            accepted=True,
            repaired_answer=answer_text,
            reason="summary_style_one_sentence" if accepted else "summary_style_too_long",
            constraint="summary_style",
        )
    if style == "bullets":
        accepted = _looks_like_bullets(answer_text)
        return ResponseAssessment(
            score=0.86 if accepted else 0.56,
            accepted=True,
            repaired_answer=answer_text,
            reason="summary_style_bullets" if accepted else "summary_style_not_bullets",
            constraint="summary_style",
        )
    if style == "headline":
        accepted = "\n" not in answer_text and len(answer_text) <= 120
        return ResponseAssessment(
            score=0.84 if accepted else 0.55,
            accepted=True,
            repaired_answer=answer_text,
            reason="summary_style_headline" if accepted else "summary_style_not_headline",
            constraint="summary_style",
        )
    if style == "concise":
        accepted = len(answer_text) <= 240
        return ResponseAssessment(
            score=0.82 if accepted else 0.57,
            accepted=True,
            repaired_answer=answer_text,
            reason="summary_style_concise" if accepted else "summary_style_not_concise",
            constraint="summary_style",
        )
    return None


def _sentence_count(answer_text: str) -> int:
    text = re.sub(r"\s+", " ", answer_text or "").strip()
    if not text:
        return 0
    parts = [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return max(1, len(parts))


def _looks_like_bullets(answer_text: str) -> bool:
    lines = [line.strip() for line in (answer_text or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    bullet_lines = sum(
        1
        for line in lines
        if line.startswith(("-", "*", "+"))
        or bool(re.match(r"^\d+\.\s+", line))
    )
    return bullet_lines >= max(2, len(lines) // 2)


__all__ = [
    "_assess_evidence_support",
    "_assess_structured_evidence",
    "_assess_style_constraint",
    "_collect_evidence_text",
    "_evidence_threshold",
    "_merge_evidence_assessment",
    "_structured_evidence_threshold",
    "_summary_evidence_threshold",
]
