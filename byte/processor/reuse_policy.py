import re
from dataclasses import asdict, dataclass
from typing import Any

from byte.processor.intent import extract_request_intent
from byte.processor.pre import normalize_text
from byte.trust import evaluate_query_risk

_EXACT_TOKEN_PATTERNS = (
    re.compile(
        r"(?is)(?:return|reply|respond|answer)\s+with\s+exactly\s+(?P<token>[A-Za-z0-9_.:-]+)"
    ),
    re.compile(r"(?is)exactly\s+(?P<token>[A-Za-z0-9_.:-]+)\s+and\s+nothing\s+else"),
)

_UNIQUE_REQUEST_HINTS = (
    "one-off",
    "one off",
    "single-use",
    "single use",
    "unique prompt",
    "unique token",
    "stress unique",
)

_EXPLICIT_POLICIES = {"full_reuse", "context_only", "direct_only"}
_CONTEXT_BOUND_AUX_FIELDS = (
    "byte_changed_hunks",
    "byte_changed_files",
    "byte_repo_snapshot",
    "byte_repo_summary",
    "byte_tool_result_context",
    "byte_prompt_pieces",
)
_SESSION_SCOPED_AUX_FIELDS = (
    "byte_retrieval_context",
    "byte_document_context",
    "byte_support_articles",
)
_DEFAULT_GROUNDED_CATEGORIES = {
    "classification",
    "summarization",
    "extraction",
    "comparison",
    "question_answer",
    "instruction",
    "code_fix",
    "code_refactor",
    "test_generation",
    "code_explanation",
    "documentation",
}


@dataclass(frozen=True)
class ReusePolicy:
    mode: str = "full_reuse"
    reason: str = "default"
    explicit: bool = False
    unique_output: bool = False
    exact_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def allow_lookup(self) -> bool:
        return self.mode == "full_reuse"

    @property
    def allow_save(self) -> bool:
        return self.mode == "full_reuse"


def detect_reuse_policy(
    request_kwargs: dict[str, Any] | None,
    *,
    config: Any | None = None,
    context: dict[str, Any] | None = None,
) -> ReusePolicy:
    request_kwargs = request_kwargs or {}
    context = context or {}
    explicit = _explicit_policy(request_kwargs)
    if explicit:
        return ReusePolicy(
            mode=explicit,
            reason="request_override",
            explicit=True,
            unique_output=explicit != "full_reuse",
            exact_token=_extract_exact_token(_request_text(request_kwargs)),
        )

    request_text = _request_text(request_kwargs)
    normalized = normalize_text(request_text)
    intent = extract_request_intent(request_kwargs)
    exact_token = _extract_exact_token(request_text)
    if bool(getattr(config, "unique_output_guard", True)):
        if exact_token and _looks_like_unique_token(exact_token):
            mode = (
                "context_only"
                if bool(getattr(config, "context_only_unique_prompts", True))
                else "direct_only"
            )
            return ReusePolicy(
                mode=mode,
                reason="unique_exact_token",
                unique_output=True,
                exact_token=exact_token,
            )

        if intent.category == "exact_answer" and any(
            marker in normalized for marker in _UNIQUE_REQUEST_HINTS
        ):
            mode = (
                "context_only"
                if bool(getattr(config, "context_only_unique_prompts", True))
                else "direct_only"
            )
            return ReusePolicy(
                mode=mode,
                reason="unique_exact_request",
                unique_output=True,
                exact_token=exact_token,
            )

    if _should_use_context_only_for_grounded_request(
        request_kwargs, context, intent=intent, config=config
    ):
        return ReusePolicy(
            mode="context_only",
            reason="grounded_context_request",
            exact_token=exact_token,
        )

    trust_risk = evaluate_query_risk(request_kwargs, config, context=context)
    if trust_risk.direct_only:
        return ReusePolicy(
            mode="direct_only",
            reason=trust_risk.fallback_reason or "trust_direct_only",
            exact_token=exact_token,
        )
    if trust_risk.context_only and _should_use_context_only_for_grounded_request(
        request_kwargs, context, intent=intent, config=config
    ):
        return ReusePolicy(
            mode="context_only",
            reason=trust_risk.fallback_reason or "trust_context_only",
            exact_token=exact_token,
        )

    return ReusePolicy()


def _explicit_policy(request_kwargs: dict[str, Any]) -> str:
    raw = str(request_kwargs.get("byte_reuse_policy", "") or "").strip().lower()
    if raw in _EXPLICIT_POLICIES:
        return raw
    if bool(request_kwargs.get("byte_context_only_optimization")):
        return "context_only"
    if bool(request_kwargs.get("byte_disable_answer_reuse")):
        return "direct_only"
    return ""


def _request_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "") or item.get("content", "") or ""))
                else:
                    parts.append(str(item or ""))
            return " ".join(part for part in parts if part)
        return str(content or "")
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "")
    return ""


def _extract_exact_token(request_text: str) -> str:
    for pattern in _EXACT_TOKEN_PATTERNS:
        match = pattern.search(request_text or "")
        if match:
            return str(match.group("token") or "").strip()
    return ""


def _looks_like_unique_token(token: str) -> bool:
    lowered = normalize_text(token)
    if not lowered:
        return False
    if re.fullmatch(r"[a-f0-9]{10,}", lowered):
        return True
    if re.search(r"\d{3,}", token) and len(token) >= 8:
        return True
    if (
        token.upper() == token
        and any(separator in token for separator in ("_", "-", ":"))
        and re.search(r"\d", token)
    ):
        return True
    if any(
        hint in lowered
        for hint in ("uuid", "guid", "nonce", "digest", "hash", "trace", "request", "unique")
    ):
        return True
    return False


def _should_use_context_only_for_grounded_request(
    request_kwargs: dict[str, Any],
    context: dict[str, Any],
    *,
    intent: Any,
    config: Any,
) -> bool:
    if not bool(getattr(config, "grounded_context_only", True)):
        return False
    if not _has_grounded_aux_context(request_kwargs, context):
        return False
    grounded_categories = {
        str(item).strip()
        for item in (
            getattr(config, "grounded_context_categories", None) or _DEFAULT_GROUNDED_CATEGORIES
        )
        if str(item).strip()
    }
    return str(getattr(intent, "category", "") or "") in grounded_categories


def _has_grounded_aux_context(request_kwargs: dict[str, Any], context: dict[str, Any]) -> bool:
    if any(
        request_kwargs.get(field) not in (None, "", [], {}) for field in _CONTEXT_BOUND_AUX_FIELDS
    ):
        return True
    raw_aux = context.get("_byte_raw_aux_context", {}) or {}
    if not isinstance(raw_aux, dict):
        raw_aux = {}
    if any(raw_aux.get(field) not in (None, "", [], {}) for field in _CONTEXT_BOUND_AUX_FIELDS):
        return True
    if not str(request_kwargs.get("byte_session_id", "") or "").strip():
        memory_context = context.get("_byte_memory", {}) or {}
        if isinstance(memory_context, dict):
            session_id = str(
                memory_context.get("byte_session_id", "")
                or memory_context.get("session_id", "")
                or ""
            )
        else:
            session_id = ""
    else:
        session_id = str(request_kwargs.get("byte_session_id", "") or "").strip()
    if not session_id:
        return False
    if any(
        request_kwargs.get(field) not in (None, "", [], {}) for field in _SESSION_SCOPED_AUX_FIELDS
    ):
        return True
    return any(raw_aux.get(field) not in (None, "", [], {}) for field in _SESSION_SCOPED_AUX_FIELDS)
