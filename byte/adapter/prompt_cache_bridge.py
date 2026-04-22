from typing import Any

from byte.processor.optimization_memory import extract_prompt_pieces, stable_digest
from byte.processor.task_policy import resolve_task_policy
from byte.utils.multimodal import content_signature


def apply_native_prompt_cache(
    provider: str,
    request_kwargs: dict[str, Any] | None,
    config: Any,
) -> dict[str, Any]:
    request_kwargs = dict(request_kwargs or {})
    if not getattr(config, "native_prompt_caching", True):
        return request_kwargs

    policy = resolve_task_policy(request_kwargs, config)
    if not bool(policy.get("native_prompt_cache", True)):
        return request_kwargs

    prompt_chars = _request_chars(request_kwargs)
    if prompt_chars < int(getattr(config, "native_prompt_cache_min_chars", 1200) or 0):
        return request_kwargs

    cache_key = _native_prompt_cache_key(provider, request_kwargs)
    if not cache_key:
        return request_kwargs

    request_kwargs["native_prompt_cache_key"] = cache_key
    if getattr(config, "native_prompt_cache_ttl", None):
        request_kwargs["native_prompt_cache_ttl"] = getattr(config, "native_prompt_cache_ttl", None)

    provider_name = str(provider or "").strip().lower()
    if provider_name == "openai":
        request_kwargs["prompt_cache_key"] = cache_key
        if request_kwargs.get("native_prompt_cache_ttl"):
            request_kwargs["prompt_cache_retention"] = request_kwargs["native_prompt_cache_ttl"]
    return request_kwargs


def strip_native_prompt_cache_hints(request_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    request_kwargs = dict(request_kwargs or {})
    request_kwargs.pop("native_prompt_cache_key", None)
    request_kwargs.pop("native_prompt_cache_ttl", None)
    return request_kwargs


def _native_prompt_cache_key(provider: str, request_kwargs: dict[str, Any]) -> str:
    distilled_modules = [
        str(item).strip()
        for item in (request_kwargs.get("byte_distilled_prompt_modules") or [])
        if str(item).strip()
    ]
    pieces = extract_prompt_pieces(request_kwargs)
    if not pieces and not distilled_modules:
        payload = {
            "provider": provider,
            "model": request_kwargs.get("model", ""),
            "prompt": _request_signature_text(request_kwargs),
        }
        return stable_digest(payload)

    normalized = []
    for piece in pieces:
        piece_type = str(piece.get("type") or piece.get("piece_type") or "piece")
        if piece_type in {"history_user"} and not request_kwargs.get(
            "byte_prompt_cache_include_history"
        ):
            continue
        normalized.append(
            {
                "type": piece_type,
                "content": _content_signature(piece.get("content")),
            }
        )
    if not normalized:
        return ""
    payload = {
        "provider": provider,
        "model": request_kwargs.get("model", ""),
        "pieces": normalized,
        "modules": distilled_modules,
    }
    return stable_digest(payload)


def _request_chars(request_kwargs: dict[str, Any]) -> int:
    return len(_request_signature_text(request_kwargs))


def _request_signature_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        parts = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                parts.append(content_signature(content))
            else:
                parts.append(str(content or ""))
        return "\n".join(parts)
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "")
    return ""


def _content_signature(value: Any) -> str:
    if isinstance(value, list):
        return content_signature(value)
    return str(value or "")
