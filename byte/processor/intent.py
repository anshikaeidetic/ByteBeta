import hashlib
from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from byte.processor.fingerprint import ConversationFingerprinter
from byte.processor.pre import canonicalize_text, normalize_text
from byte.processor.tool_calls import request_tool_signature

_SUMMARY_STYLE_HINTS = (
    ("one sentence", "one_sentence"),
    ("single sentence", "one_sentence"),
    ("bullet", "bullets"),
    ("bullets", "bullets"),
    ("concise", "concise"),
    ("brief", "concise"),
    ("headline", "headline"),
)

_EXTRACT_FORMAT_HINTS = (
    ("json", "json"),
    ("yaml", "yaml"),
    ("csv", "csv"),
)


@dataclass(frozen=True)
class IntentRecord:
    category: str
    route_key: str
    canonical_key: str
    payload_digest: str = ""
    tool_signature: str = ""
    slots: dict[str, Any] = field(default_factory=dict)


class IntentGraph:
    """Track reusable intent flows across sessions and providers."""

    def __init__(self, window_size: int = 3) -> None:
        self._fingerprinter = ConversationFingerprinter(window_size=window_size)
        self._nodes = Counter()
        self._edges = Counter()
        self._active_tracks: dict[str, str] = {}
        self._recent_paths: dict[str, list[str]] = {}
        self._records = 0
        self._lock = Lock()

    def record(
        self,
        request_kwargs: dict[str, Any] | None = None,
        *,
        intent: IntentRecord | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        request_kwargs = request_kwargs or {}
        intent = intent or extract_request_intent(request_kwargs)
        messages = request_kwargs.get("messages") or []
        track_id = _resolve_track_id(self._fingerprinter, messages, session_id=session_id)

        with self._lock:
            self._records += 1
            self._nodes[intent.route_key] += 1
            if track_id:
                previous = self._active_tracks.get(track_id)
                if previous:
                    self._edges[(previous, intent.route_key)] += 1
                self._active_tracks[track_id] = intent.route_key
                history = self._recent_paths.setdefault(track_id, [])
                history.append(intent.route_key)
                if len(history) > 10:
                    del history[:-10]

        return {
            "category": intent.category,
            "intent": intent.route_key,
            "track_id": track_id,
        }

    def stats(self, top_n: int = 10) -> dict[str, Any]:
        with self._lock:
            top_intents = [
                {"intent": intent, "count": count}
                for intent, count in self._nodes.most_common(top_n)
            ]
            top_transitions = [
                {"from": src, "to": dst, "count": count}
                for (src, dst), count in self._edges.most_common(top_n)
            ]
            return {
                "total_records": self._records,
                "unique_intents": len(self._nodes),
                "transition_count": sum(self._edges.values()),
                "tracked_conversations": len(self._active_tracks),
                "top_intents": top_intents,
                "top_transitions": top_transitions,
            }

    def merge(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._records += int(payload.get("total_records", 0) or 0)
            for item in payload.get("top_intents", []):
                self._nodes[item["intent"]] += int(item.get("count", 0) or 0)
            for item in payload.get("top_transitions", []):
                edge = (item["from"], item["to"])
                self._edges[edge] += int(item.get("count", 0) or 0)

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._active_tracks.clear()
            self._recent_paths.clear()
            self._records = 0


def extract_request_intent(request_kwargs: dict[str, Any]) -> IntentRecord:
    request_text = _extract_request_text(request_kwargs)
    normalized = normalize_text(request_text)
    canonical = canonicalize_text(request_text)
    tool_signature = request_tool_signature(request_kwargs)
    payload_digest = _structured_payload_digest(canonical, normalized)

    if tool_signature:
        route_key = f"tool_call::{tool_signature}"
        return IntentRecord(
            category="tool_call",
            route_key=route_key,
            canonical_key=canonical or normalized,
            payload_digest=payload_digest or _short_digest(normalized),
            tool_signature=tool_signature,
            slots={"tool_signature": tool_signature},
        )

    if canonical.startswith("classify::"):
        payload_key = _safe_part(canonical, 1)
        labels = _safe_part(canonical, 2)
        return IntentRecord(
            category="classification",
            route_key="classification",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={
                "payload_key": payload_key,
                "labels": labels,
            },
        )
    if canonical.startswith("translate::"):
        language = _safe_part(canonical, 1)
        return IntentRecord(
            category="translation",
            route_key=f"translation::{language}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={"language": language},
        )
    if canonical.startswith("exact_answer::"):
        token = _safe_part(canonical, 1)
        return IntentRecord(
            category="exact_answer",
            route_key="exact_answer",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={"token": token},
        )
    if canonical.startswith("summarize::"):
        style = _safe_part(canonical, 1)
        return IntentRecord(
            category="summarization",
            route_key=f"summarization::{style}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={"style": style},
        )
    if canonical.startswith("extract::"):
        fmt = _safe_part(canonical, 1)
        fields = _safe_part(canonical, 2)
        return IntentRecord(
            category="extraction",
            route_key=f"extraction::{fmt}::{fields}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={
                "format": fmt,
                "fields": fields,
            },
        )
    if canonical.startswith("qa_fact::"):
        relation = _safe_part(canonical, 1)
        entity = _safe_part(canonical, 2)
        return IntentRecord(
            category="question_answer",
            route_key=f"question_answer::{relation}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={
                "relation": relation,
                "entity": entity,
            },
        )
    if canonical.startswith("code_fix::"):
        language = _safe_part(canonical, 1)
        diagnostic = _safe_part(canonical, 2)
        return IntentRecord(
            category="code_fix",
            route_key=f"code_fix::{language}::{diagnostic}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={
                "language": language,
                "diagnostic": diagnostic,
            },
        )
    if canonical.startswith("code_tests::"):
        language = _safe_part(canonical, 1)
        framework = _safe_part(canonical, 2)
        return IntentRecord(
            category="test_generation",
            route_key=f"test_generation::{language}::{framework}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={
                "language": language,
                "framework": framework,
            },
        )
    if canonical.startswith("code_explain::"):
        language = _safe_part(canonical, 1)
        style = _safe_part(canonical, 2)
        return IntentRecord(
            category="code_explanation",
            route_key=f"code_explanation::{language}::{style}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={
                "language": language,
                "style": style,
            },
        )
    if canonical.startswith("code_refactor::"):
        language = _safe_part(canonical, 1)
        goal = _safe_part(canonical, 2)
        return IntentRecord(
            category="code_refactor",
            route_key=f"code_refactor::{language}::{goal}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={
                "language": language,
                "goal": goal,
            },
        )
    if canonical.startswith("code_doc::"):
        language = _safe_part(canonical, 1)
        return IntentRecord(
            category="documentation",
            route_key=f"documentation::{language}",
            canonical_key=canonical,
            payload_digest=_structured_payload_digest(canonical, normalized),
            slots={"language": language},
        )

    if any(token in normalized for token in ("compare", "difference", "versus", " vs ")):
        category = "comparison"
    elif any(token in normalized for token in ("summarize", "summary", "tldr", "tl dr")):
        category = "summarization"
    elif any(token in normalized for token in ("extract", "fields", "return json", "keys")):
        category = "extraction"
    elif normalized.endswith("?") or "?" in request_text:
        category = "question_answer"
    else:
        category = "instruction"

    route_bits = [category]
    slots: dict[str, Any] = {}
    if category == "summarization":
        style = _summary_style(normalized)
        route_bits.append(style)
        slots["style"] = style
    if category == "extraction":
        fmt = _extract_format(normalized)
        route_bits.append(fmt)
        slots["format"] = fmt
    route_key = "::".join(bit for bit in route_bits if bit)
    return IntentRecord(
        category=category,
        route_key=route_key,
        canonical_key=canonical or normalized,
        payload_digest=_short_digest(normalized),
        slots=slots,
    )


def _resolve_track_id(
    fingerprinter: ConversationFingerprinter,
    messages: list[dict[str, Any]],
    *,
    session_id: str | None,
) -> str | None:
    if session_id:
        return f"session::{session_id}"
    context_key = fingerprinter.context_key(messages)
    if context_key:
        return f"context::{context_key}"
    return None


def _extract_request_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return " ".join(part for part in parts if part)
        return str(content)
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt"))
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input"))
    return ""


def _structured_payload_digest(canonical: str, normalized: str) -> str:
    if "::" not in canonical:
        return _short_digest(normalized)
    return _safe_part(canonical, -1)


def _short_digest(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


def _safe_part(value: str, index: int) -> str:
    parts = value.split("::")
    try:
        return parts[index]
    except IndexError:
        return ""


def _summary_style(normalized: str) -> str:
    for needle, label in _SUMMARY_STYLE_HINTS:
        if needle in normalized:
            return label
    return "default"


def _extract_format(normalized: str) -> str:
    for needle, label in _EXTRACT_FORMAT_HINTS:
        if needle in normalized:
            return label
    return "plain"
