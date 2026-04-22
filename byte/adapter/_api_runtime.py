"""Routing, policy, and research helpers for the public adapter API."""

from __future__ import annotations

from typing import Any

from byte import Cache, cache
from byte.adapter.router_runtime import (
    clear_model_aliases as clear_router_aliases,
)
from byte.adapter.router_runtime import (
    clear_route_runtime_stats,
    register_model_alias,
    route_runtime_stats,
)
from byte.adapter.router_runtime import (
    model_aliases as router_aliases,
)
from byte.processor.model_router import route_request_model
from byte.processor.policy import policy_stats
from byte.processor.workflow import detect_ambiguity, plan_request_workflow
from byte.research import research_registry, research_registry_summary


def preview_model_route(
    request_kwargs: dict[str, Any], cache_obj: Cache | None = None
) -> dict[str, Any] | None:
    """Preview ByteAI Cache's smart-routing decision without making a request."""
    cache_obj = cache_obj if cache_obj else cache
    decision = route_request_model(dict(request_kwargs), cache_obj.config)
    return decision.to_dict() if decision else None


def preview_workflow_plan(
    request_kwargs: dict[str, Any], cache_obj: Cache | None = None
) -> dict[str, Any]:
    """Preview ByteAI Cache's workflow planner without making a provider call."""
    cache_obj = cache_obj if cache_obj else cache
    ambiguity = detect_ambiguity(
        request_kwargs,
        min_chars=cache_obj.config.ambiguity_min_chars,
    )
    failure_hint = cache_obj.failure_memory_hint(
        request_kwargs,
        provider=str(request_kwargs.get("byte_provider", "") or ""),
        model=str(request_kwargs.get("model", "") or ""),
    )
    decision = plan_request_workflow(
        dict(request_kwargs),
        cache_obj.config,
        ambiguity=ambiguity,
        failure_hint=failure_hint,
    )
    return {
        "ambiguity": ambiguity.to_dict(),
        "failure_hint": failure_hint,
        "decision": decision.to_dict(),
    }


def remember_execution_result(
    request_kwargs: dict[str, Any],
    *,
    answer: Any,
    cache_obj: Cache | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Persist an execution result in ByteAI Cache's execution memory."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.remember_execution_result(request_kwargs, answer=answer, **kwargs)


def execution_memory_summary(cache_obj: Cache | None = None) -> dict[str, Any]:
    """Return summary stats for execution memory."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.execution_memory_stats()


def suggest_patch_pattern(
    request_kwargs: dict[str, Any],
    *,
    cache_obj: Cache | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Suggest a patch pattern from prior execution memory when available."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.suggest_patch_pattern(request_kwargs, **kwargs)


def policy_summary() -> dict[str, Any]:
    """Return in-process policy runtime stats."""
    return policy_stats()


def register_router_alias(alias: str, targets: Any) -> dict[str, Any]:
    """Register a model alias for unified multi-provider routing."""
    register_model_alias(alias, list(targets or []))
    return {"alias": alias, "targets": router_aliases().get(alias, [])}


def clear_router_alias_registry() -> dict[str, Any]:
    """Clear the in-process unified-router alias registry."""
    clear_router_aliases()
    return {"cleared": True}


def router_registry_summary() -> dict[str, Any]:
    """Return alias and route stats for the unified provider router."""
    return route_runtime_stats()


def clear_router_runtime() -> dict[str, Any]:
    """Reset in-process unified-router route stats without touching cache data."""
    clear_route_runtime_stats()
    return {"cleared": True}


def compression_runtime_summary() -> dict[str, Any]:
    """Return compression runtime stats for the H2O subsystem."""
    from byte.h2o.runtime import h2o_runtime_stats

    return h2o_runtime_stats()


def research_registry_overview() -> dict[str, Any]:
    """Return the summarized research registry surface."""
    return research_registry_summary()


def research_registry_catalog() -> dict[str, Any]:
    """Return the full research registry catalog."""
    return research_registry()
