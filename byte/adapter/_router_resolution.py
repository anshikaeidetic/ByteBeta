"""Provider and model resolution helpers for routed execution."""

from collections.abc import Sequence
from typing import Any

from byte import cache
from byte.processor.budget import estimate_request_cost
from byte.utils.error import ByteErrorCode, CacheError

from ._router_registry import _REGISTRY, RouteTarget

_PROVIDER_MODULES = {
    "openai": "byte._backends.openai",
    "deepseek": "byte._backends.deepseek",
    "anthropic": "byte._backends.anthropic",
    "gemini": "byte._backends.gemini",
    "groq": "byte._backends.groq",
    "openrouter": "byte._backends.openrouter",
    "ollama": "byte._backends.ollama",
    "mistral": "byte._backends.mistral",
    "cohere": "byte._backends.cohere",
    "bedrock": "byte._backends.bedrock",
    "huggingface": "byte._backends.huggingface",
}

_SURFACE_CAPABILITIES = {
    "chat_completion": "chat_completion",
    "text_completion": "text_completion",
    "image": "image_generation",
    "audio_transcribe": "audio_transcription",
    "audio_translate": "audio_translation",
    "speech": "speech_generation",
    "moderation": "moderation",
}

def resolve_provider_model(
    model: str,
    *,
    provider_hint: str = "",
    aliases: dict[str, Sequence[str]] | None = None,
    allow_backend_target: bool = False,
) -> list[RouteTarget]:
    """Resolve a public route name or alias into concrete provider targets."""

    raw_model = str(model or "").strip()
    if not raw_model:
        raise CacheError("A model name is required.", code=ByteErrorCode.PROVIDER_CONFIG)

    alias_targets = _coerce_targets((aliases or {}).get(raw_model)) or _REGISTRY.resolve_alias(
        raw_model
    )
    if alias_targets:
        return [
            _target_from_string(
                target, source="alias", alias=raw_model, provider_hint=provider_hint
            )
            for target in alias_targets
        ]

    if "/" in raw_model and not allow_backend_target:
        raise CacheError(
            "Byte routes must use Byte route names. Backend-qualified model selectors are not part of the public API.",
            code=ByteErrorCode.PROVIDER_CONFIG,
        )

    return [_target_from_string(raw_model, source="direct", provider_hint=provider_hint)]


def resolve_model_name_for_provider(
    model: str,
    provider: str,
    *,
    cache_obj: Any | None = None,
    overrides: dict[str, Any] | None = None,
) -> str:
    """Return the backend model name that should be sent to one provider."""

    provider = str(provider or "").strip().lower()
    if not provider:
        return str(model or "")
    config = getattr((cache_obj if cache_obj is not None else cache), "config", None)
    aliases: dict[str, Sequence[str]] = {}
    if config is not None:
        aliases.update(getattr(config, "routing_model_aliases", {}) or {})
    aliases.update((overrides or {}).get("routing_model_aliases", {}) or {})
    targets = resolve_provider_model(
        str(model or ""),
        provider_hint=provider,
        aliases=aliases,
    )
    for target in targets:
        if target.provider in ("", provider):
            return target.model
    return str(model or "")

def _target_from_string(
    value: str,
    *,
    source: str,
    alias: str = "",
    provider_hint: str = "",
) -> RouteTarget:
    normalized = str(value or "").strip()
    if not normalized:
        raise CacheError("Invalid provider target: empty value.", code=ByteErrorCode.ROUTER_NO_TARGET)
    if "/" in normalized:
        maybe_provider, actual_model = normalized.split("/", 1)
        if maybe_provider in _PROVIDER_MODULES:
            return RouteTarget(
                provider=maybe_provider,
                model=actual_model,
                source=source,
                alias=alias,
            )
    return RouteTarget(
        provider=str(provider_hint or "").strip().lower(),
        model=normalized,
        source=source,
        alias=alias,
    )


def _coerce_targets(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _enrich_target(target: RouteTarget, *, request_kwargs: dict[str, Any] | None) -> RouteTarget:
    estimated_cost = estimate_request_cost(target.qualified_model or target.model, request_kwargs)
    metadata = dict(target.metadata or {})
    metadata.update(
        {
            "estimated_cost_usd": estimated_cost,
            "avg_latency_ms": round(_REGISTRY.avg_latency(target), 2),
            "health_score": round(_REGISTRY.health_score(target), 4),
        }
    )
    return RouteTarget(
        provider=target.provider,
        model=target.model,
        source=target.source,
        alias=target.alias,
        metadata=metadata,
    )


def _provider_request_kwargs(provider: str, kwargs: dict[str, Any], config: Any) -> dict[str, Any]:
    payload = dict(kwargs)
    provider_keys: dict[str, Any] = {}
    if config is not None:
        provider_keys.update(getattr(config, "routing_provider_keys", {}) or {})
    provider_keys.update(payload.pop("byte_provider_keys", {}) or {})
    generic_api_key = payload.get("api_key")
    provider_entry = provider_keys.get(provider)

    if isinstance(provider_entry, dict):
        payload.update(provider_entry)
    elif provider_entry not in (None, "", [], {}):
        if provider == "ollama":
            payload["host"] = provider_entry
        else:
            payload["api_key"] = provider_entry
    elif generic_api_key and provider != "ollama":
        payload["api_key"] = generic_api_key

    return payload

def _supports_surface(provider: str, surface: str) -> bool:
    capability = _SURFACE_CAPABILITIES.get(surface, "")
    if not capability:
        return False
    from byte.adapter.api import supports_capability  # pylint: disable=C0415

    return supports_capability(provider, capability)
