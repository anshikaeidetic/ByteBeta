from dataclasses import asdict, dataclass
from typing import Any

from byte.processor.intent import extract_request_intent
from byte.processor.pre import normalize_text
from byte.processor.tool_calls import request_tool_signature

_REASONING_HINTS = (
    "step by step",
    "reason",
    "analyze",
    "analysis",
    "tradeoff",
    "compare",
    "debug",
    "root cause",
    "architecture",
    "design",
    "derive",
    "prove",
)

_PII_HINTS = (
    "ssn",
    "social security",
    "credit card",
    "cvv",
    "password",
    "secret",
    "api key",
    "private key",
    "passport",
    "bank account",
)

_JAILBREAK_HINTS = (
    "ignore previous instructions",
    "ignore the system prompt",
    "reveal your system prompt",
    "show hidden instructions",
    "jailbreak",
    "bypass policy",
    "developer message",
)

_FACTUALITY_HINTS = (
    "latest",
    "today",
    "current",
    "news",
    "price",
    "legal",
    "medical",
    "financial",
    "law",
    "stock",
    "weather",
)

_STRUCTURED_OUTPUT_HINTS = (
    "json",
    "yaml",
    "csv",
    "schema",
    "exactly",
    "one label",
    "valid",
)

_MEDIA_TYPES = {
    "image",
    "image_url",
    "input_image",
    "input_audio",
    "audio",
    "file",
    "input_file",
}

_CODER_CATEGORIES = {
    "code_fix",
    "code_refactor",
    "test_generation",
    "documentation",
}


@dataclass(frozen=True)
class RouteSignals:
    category: str
    route_key: str
    prompt_chars: int
    message_count: int
    has_tools: bool
    has_multimodal_input: bool
    has_image_input: bool
    has_audio_input: bool
    has_file_input: bool
    needs_reasoning: bool
    factual_risk: bool
    pii_risk: bool
    jailbreak_risk: bool
    structured_output: bool
    recommended_route: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_route_signals(
    request_kwargs: dict[str, Any] | None,
    *,
    long_prompt_chars: int = 1200,
    multi_turn_threshold: int = 6,
) -> RouteSignals:
    request_kwargs = request_kwargs or {}
    intent = extract_request_intent(request_kwargs)
    prompt_chars, message_count = _request_size(request_kwargs)
    request_text = _request_text(request_kwargs)
    normalized = normalize_text(request_text)
    tool_signature = request_tool_signature(request_kwargs)
    has_tools = bool(tool_signature)
    media_flags = _detect_multimodal_inputs(request_kwargs)
    needs_reasoning = _needs_reasoning(
        normalized,
        prompt_chars=prompt_chars,
        message_count=message_count,
        category=intent.category,
        long_prompt_chars=long_prompt_chars,
        multi_turn_threshold=multi_turn_threshold,
    )
    factual_risk = any(token in normalized for token in _FACTUALITY_HINTS)
    pii_risk = any(token in normalized for token in _PII_HINTS)
    jailbreak_risk = any(token in normalized for token in _JAILBREAK_HINTS)
    structured_output = any(token in normalized for token in _STRUCTURED_OUTPUT_HINTS)
    recommended_route = _recommended_route(
        category=intent.category,
        has_tools=has_tools,
        needs_reasoning=needs_reasoning,
        factual_risk=factual_risk,
        pii_risk=pii_risk,
        jailbreak_risk=jailbreak_risk,
        has_multimodal_input=media_flags["has_multimodal_input"],
        structured_output=structured_output,
        prompt_chars=prompt_chars,
    )
    return RouteSignals(
        category=intent.category,
        route_key=intent.route_key,
        prompt_chars=prompt_chars,
        message_count=message_count,
        has_tools=has_tools,
        has_multimodal_input=media_flags["has_multimodal_input"],
        has_image_input=media_flags["has_image_input"],
        has_audio_input=media_flags["has_audio_input"],
        has_file_input=media_flags["has_file_input"],
        needs_reasoning=needs_reasoning,
        factual_risk=factual_risk,
        pii_risk=pii_risk,
        jailbreak_risk=jailbreak_risk,
        structured_output=structured_output,
        recommended_route=recommended_route,
    )


def _request_size(request_kwargs: dict[str, Any]) -> tuple:
    messages = request_kwargs.get("messages") or []
    total_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total_chars += len(str(item.get("text", "") or item.get("content", "") or ""))
                else:
                    total_chars += len(str(item or ""))
        else:
            total_chars += len(str(content or ""))
    if total_chars == 0:
        if request_kwargs.get("prompt") is not None:
            total_chars = len(str(request_kwargs.get("prompt") or ""))
        elif request_kwargs.get("input") is not None:
            total_chars = len(str(request_kwargs.get("input") or ""))
    return total_chars, len(messages)


def _request_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(
                        str(
                            item.get("text", "")
                            or item.get("content", "")
                            or item.get("mime_type", "")
                            or ""
                        )
                    )
                else:
                    parts.append(str(item or ""))
            return " ".join(part for part in parts if part)
        return str(content or "")
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "")
    return ""


def _detect_multimodal_inputs(request_kwargs: dict[str, Any]) -> dict[str, bool]:
    flags = {
        "has_multimodal_input": False,
        "has_image_input": False,
        "has_audio_input": False,
        "has_file_input": False,
    }
    messages = request_kwargs.get("messages") or []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type in _MEDIA_TYPES:
                flags["has_multimodal_input"] = True
            if item_type in {"image", "image_url", "input_image"}:
                flags["has_image_input"] = True
            elif item_type in {"audio", "input_audio"}:
                flags["has_audio_input"] = True
            elif item_type in {"file", "input_file"}:
                flags["has_file_input"] = True
    if request_kwargs.get("file") not in (None, "", [], {}):
        flags["has_multimodal_input"] = True
        flags["has_file_input"] = True
    return flags


def _needs_reasoning(
    normalized: str,
    *,
    prompt_chars: int,
    message_count: int,
    category: str,
    long_prompt_chars: int,
    multi_turn_threshold: int,
) -> bool:
    if prompt_chars >= long_prompt_chars:
        return True
    if message_count >= multi_turn_threshold:
        return True
    if category in {"code_fix", "code_refactor", "test_generation", "comparison", "documentation"}:
        return True
    return any(keyword in normalized for keyword in _REASONING_HINTS)


def _recommended_route(
    *,
    category: str,
    has_tools: bool,
    needs_reasoning: bool,
    factual_risk: bool,
    pii_risk: bool,
    jailbreak_risk: bool,
    has_multimodal_input: bool,
    structured_output: bool,
    prompt_chars: int,
) -> str:
    if has_tools:
        return "tool"
    if jailbreak_risk or pii_risk:
        return "expensive"
    if factual_risk or has_multimodal_input:
        return "expensive"
    if category in _CODER_CATEGORIES:
        return "coder"
    if needs_reasoning:
        return "reasoning"
    if category in {"classification", "translation", "exact_answer", "extraction"}:
        return "cheap"
    if structured_output and prompt_chars <= 800:
        return "cheap"
    return "default"
