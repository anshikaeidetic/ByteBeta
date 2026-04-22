"""Gateway bootstrap and provider-resolution helpers for the Byte server."""

from __future__ import annotations

import os
from typing import Any, cast

from fastapi import HTTPException

from byte import Cache, Config, __version__
from byte.adapter.api import (
    init_exact_cache,
    init_hybrid_cache,
    init_normalized_cache,
    init_safe_semantic_cache,
)
from byte.core import LazyCacheProxy
from byte.processor.pre import last_content, normalized_last_content
from byte_server._server_state import ServerServices

_CREDENTIAL_FREE_PROVIDERS = {"huggingface"}


def _prewarm_embedding(cache_obj: Cache) -> None:
    """Run one dummy embedding so the ONNX model is loaded before first request."""
    try:
        dm = getattr(cache_obj, "data_manager", None)
        if dm is None:
            return
        embedding = getattr(dm, "embedding_func", None) or getattr(dm, "embedding", None)
        if embedding is None:
            return
        to_embeddings = getattr(embedding, "to_embeddings", None)
        if callable(to_embeddings):
            to_embeddings("warmup")
    except Exception:
        pass


def _init_gateway_cache(
    mode: str,
    cache_dir: str,
    cache_obj: Cache | None = None,
    config: Config | None = None,
) -> Cache:
    """Initialize the Byte gateway cache with a built-in strategy."""
    mode = (mode or "normalized").lower()
    target_cache = cache_obj if cache_obj is not None else Cache()
    base_config = config if config is not None else Config(enable_token_counter=False)

    if mode == "semantic":
        init_safe_semantic_cache(
            data_dir=cache_dir,
            pre_func=last_content,
            cache_obj=target_cache,
            config=base_config,
        )
        _prewarm_embedding(target_cache)
        return target_cache

    if mode == "exact":
        init_exact_cache(
            data_dir=cache_dir,
            cache_obj=target_cache,
            pre_func=last_content,
            config=base_config,
        )
        return target_cache

    if mode == "normalized":
        init_normalized_cache(
            data_dir=cache_dir,
            cache_obj=target_cache,
            pre_func=last_content,
            normalized_pre_func=normalized_last_content,
            config=base_config,
        )
        return target_cache

    if mode == "hybrid":
        result = cast(
            Cache,
            init_hybrid_cache(
                data_dir=cache_dir,
                cache_obj=target_cache,
                pre_func=last_content,
                normalized_pre_func=normalized_last_content,
                config=base_config,
            ),
        )
        # Hybrid chains exact → normalized → semantic; warm the semantic layer.
        _prewarm_embedding(result)
        return result

    raise ValueError(f"Unsupported gateway cache mode: {mode}")


def _active_cache(services: ServerServices) -> Cache | LazyCacheProxy:
    """Use the gateway cache when enabled, else fall back to the default cache."""
    return services.active_cache()


def _configure_server_telemetry(
    services: ServerServices,
    args: Any,
    *,
    telemetry_runtime_cls: Any,
    telemetry_settings_cls: Any,
    readiness_provider: Any,
) -> None:
    runtime = services.runtime_state()
    if runtime.telemetry_runtime is not None:
        runtime.telemetry_runtime.shutdown()
        runtime.telemetry_runtime = None

    settings = telemetry_settings_cls.from_sources(
        service_name="byteai-cache-server",
        service_version=__version__,
        enabled=args.otel_enabled,
        endpoint=args.otel_endpoint,
        protocol=args.otel_protocol,
        headers=args.otel_headers,
        insecure=args.otel_insecure,
        disable_traces=args.otel_disable_traces,
        disable_metrics=args.otel_disable_metrics,
        export_interval_ms=args.otel_export_interval_ms,
        service_namespace=args.otel_service_namespace,
        environment=args.otel_environment,
        resource_attributes=args.otel_resource_attributes,
        datadog_enabled=args.datadog_enabled,
        datadog_agent_host=args.datadog_agent_host,
        datadog_service=args.datadog_service,
        datadog_env=args.datadog_env,
        datadog_version=args.datadog_version,
    )
    runtime.telemetry_runtime = telemetry_runtime_cls(
        settings,
        get_active_cache=lambda: _active_cache(services),
        get_readiness=readiness_provider,
        get_router_summary=services.router_registry_summary,
    ).start()
    runtime.telemetry_runtime.bind_cache(services.base_cache, cache_role="base")
    if runtime.gateway_cache is not None:
        runtime.telemetry_runtime.bind_cache(runtime.gateway_cache, cache_role="gateway")


async def _call_gateway_completion(services: ServerServices, **kwargs: Any) -> Any:
    """Run the default gateway backend call off the event loop."""
    return await services.run_in_threadpool()(services.default_backend.ChatCompletion.create, **kwargs)


async def _call_chat_completion(
    services: ServerServices, handler: Any, **kwargs: Any
) -> Any:
    """Run the blocking adapter call off the event loop."""
    return await services.run_in_threadpool()(handler, **kwargs)


def _resolve_gateway_handler(
    services: ServerServices,
    model_name: str,
    *,
    backend_handler: Any,
    routed_handler: Any,
) -> tuple[bool, Any]:
    use_routed_gateway = _should_use_routed_gateway(services, model_name)
    handler = routed_handler if use_routed_gateway else backend_handler
    return use_routed_gateway, handler


def _resolve_gateway_auth_kwargs(
    services: ServerServices,
    headers: Any,
    model_name: str,
    *,
    use_routed_gateway: bool,
) -> dict[str, Any]:
    return (
        _resolve_routed_gateway_kwargs(services, headers, model_name)
        if use_routed_gateway
        else {"api_key": _resolve_gateway_key(headers)}
    )


def _response_cache_hit(payload: Any) -> bool:
    return bool(isinstance(payload, dict) and payload.get("byte", False))


def _speech_media_type(response_format: str) -> str:
    mapping = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/L16",
    }
    return mapping.get(str(response_format or "mp3").strip().lower(), "application/octet-stream")


def _server_backend_api_key() -> str | None:
    return str(
        os.getenv("BYTE_BACKEND_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    ).strip() or None


def _resolve_gateway_key(headers: Any) -> str:
    """Resolve a generic Byte gateway backend key."""
    auth_header = headers.get("authorization")
    if auth_header:
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header. Use 'Bearer <API_KEY>'.",
            )
        return str(token).strip()

    server_key = _server_backend_api_key()
    if server_key:
        return server_key

    raise HTTPException(
        status_code=401,
        detail=(
            "Missing Byte gateway credentials. Provide 'Authorization: Bearer <API_KEY>' "
            "or configure BYTE_BACKEND_API_KEY on the ByteAI Cache server."
        ),
    )


def _env_provider_keys() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if os.getenv("OPENAI_API_KEY"):
        payload["openai"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("DEEPSEEK_API_KEY"):
        payload["deepseek"] = os.getenv("DEEPSEEK_API_KEY")
    if os.getenv("ANTHROPIC_API_KEY"):
        payload["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        payload["gemini"] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if os.getenv("GROQ_API_KEY"):
        payload["groq"] = os.getenv("GROQ_API_KEY")
    if os.getenv("OPENROUTER_API_KEY"):
        payload["openrouter"] = os.getenv("OPENROUTER_API_KEY")
    if os.getenv("OLLAMA_HOST"):
        payload["ollama"] = os.getenv("OLLAMA_HOST")
    if os.getenv("MISTRAL_API_KEY"):
        payload["mistral"] = os.getenv("MISTRAL_API_KEY")
    if os.getenv("COHERE_API_KEY"):
        payload["cohere"] = os.getenv("COHERE_API_KEY")
    if os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
        payload["bedrock"] = {
            "region_name": os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        }
    return payload


def _provider_from_model(model_name: str) -> str:
    model_name = str(model_name or "").strip()
    if "/" not in model_name:
        return ""
    provider, _, _ = model_name.partition("/")
    return provider.strip().lower()


def _alias_targets(services: ServerServices, model_name: str) -> list[str]:
    runtime = services.runtime_state()
    return list(runtime.gateway_routes.get(str(model_name or "").strip(), []) or [])


def _provider_requires_credentials(provider: str) -> bool:
    return bool(provider) and provider not in _CREDENTIAL_FREE_PROVIDERS


def _model_can_run_without_credentials(services: ServerServices, model_name: str) -> bool:
    provider = _provider_from_model(model_name)
    if provider:
        return not _provider_requires_credentials(provider)
    targets = _alias_targets(services, model_name)
    if not targets:
        return False
    providers = {_provider_from_model(target) for target in targets if _provider_from_model(target)}
    return bool(providers) and all(not _provider_requires_credentials(item) for item in providers)


def _should_use_routed_gateway(services: ServerServices, model_name: str) -> bool:
    runtime = services.runtime_state()
    model_name = str(model_name or "").strip()
    if runtime.gateway_mode == "adaptive":
        return True
    if runtime.gateway_mode == "backend":
        return False
    return model_name in set(runtime.gateway_routes.keys())


def _resolve_routed_gateway_kwargs(
    services: ServerServices, headers: Any, model_name: str
) -> dict[str, Any]:
    provider_keys: dict[str, Any] = _env_provider_keys()

    # Optional X-Byte-Provider header lets the caller tell the adaptive router
    # which backend module to dispatch to (e.g. "deepseek", "gemini", "groq").
    # Without this hint the router tries to resolve bare model names via aliases
    # and fails when no alias is configured, returning "No provider target supports".
    provider_hint = str(headers.get("x-byte-provider", "") or "").strip().lower()
    hint_kwargs: dict[str, Any] = (
        {"byte_provider_hint": provider_hint} if provider_hint else {}
    )

    auth_header = headers.get("authorization")
    if auth_header:
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header. Use 'Bearer <API_KEY>'.",
            )
        api_key = token.strip()
        # Register the provided key under the hinted provider too, so the router
        # can look it up when dispatching to that specific backend.
        keys = dict(provider_keys)
        if provider_hint and provider_hint not in keys:
            keys[provider_hint] = api_key
        return {
            "api_key": api_key,
            "byte_provider_keys": keys,
            **hint_kwargs,
        }

    if not provider_keys and _model_can_run_without_credentials(services, model_name):
        return {**hint_kwargs}

    if provider_keys:
        return {
            "byte_provider_keys": provider_keys,
            **hint_kwargs,
        }

    raise HTTPException(
        status_code=401,
        detail=(
            "Missing provider credentials. Provide 'Authorization: Bearer <API_KEY>' "
            "or configure backend credentials on the ByteAI Cache server."
        ),
    )


__all__ = [
    "_active_cache",
    "_call_chat_completion",
    "_call_gateway_completion",
    "_configure_server_telemetry",
    "_env_provider_keys",
    "_init_gateway_cache",
    "_model_can_run_without_credentials",
    "_resolve_gateway_auth_kwargs",
    "_resolve_gateway_handler",
    "_response_cache_hit",
    "_server_backend_api_key",
    "_should_use_routed_gateway",
    "_speech_media_type",
]
