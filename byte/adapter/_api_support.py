"""Public adapter-support helpers built from smaller internal runtime modules."""

from __future__ import annotations

from typing import Any

from byte.adapter._api_capabilities import (
    provider_capabilities,
    provider_capability_matrix,
    supports_capability,
)
from byte.adapter._api_memory import (
    batch_embeddings,
    batch_rerank,
    export_memory_artifact,
    export_memory_snapshot,
    import_memory_artifact,
    import_memory_snapshot,
    memory_summary,
    note_session_delta,
    recall_artifact,
    recall_rerank_result,
    recall_retrieval_result,
    recall_tool_result,
    recent_interactions,
    remember_artifact,
    remember_interaction,
    remember_rerank_result,
    remember_retrieval_result,
    remember_tool_result,
    remember_workflow_plan,
    run_rerank,
    run_retrieval,
    run_tool,
    workflow_plan_hint,
)
from byte.adapter._api_runtime import (
    clear_router_alias_registry,
    clear_router_runtime,
    compression_runtime_summary,
    execution_memory_summary,
    policy_summary,
    preview_model_route,
    preview_workflow_plan,
    register_router_alias,
    remember_execution_result,
    research_registry_catalog,
    research_registry_overview,
    router_registry_summary,
    suggest_patch_pattern,
)
from byte.adapter.adapter import adapt

__all__ = [
    "batch_embeddings",
    "batch_rerank",
    "clear_router_alias_registry",
    "clear_router_runtime",
    "compression_runtime_summary",
    "execution_memory_summary",
    "export_memory_artifact",
    "export_memory_snapshot",
    "get",
    "import_memory_artifact",
    "import_memory_snapshot",
    "memory_summary",
    "note_session_delta",
    "policy_summary",
    "preview_model_route",
    "preview_workflow_plan",
    "provider_capabilities",
    "provider_capability_matrix",
    "put",
    "recall_artifact",
    "recall_rerank_result",
    "recall_retrieval_result",
    "recall_tool_result",
    "recent_interactions",
    "register_router_alias",
    "remember_artifact",
    "remember_execution_result",
    "remember_interaction",
    "remember_rerank_result",
    "remember_retrieval_result",
    "remember_tool_result",
    "remember_workflow_plan",
    "research_registry_catalog",
    "research_registry_overview",
    "router_registry_summary",
    "run_rerank",
    "run_retrieval",
    "run_tool",
    "suggest_patch_pattern",
    "supports_capability",
    "workflow_plan_hint",
]


def _cache_data_converter(cache_data: Any) -> Any:
    """Return cache payloads unchanged when serving direct `put`/`get` helpers."""
    return cache_data


def _update_cache_callback_none(
    llm_data: Any,
    _update_cache_func: Any,
    *_args: Any,
    **_kwargs: Any,
) -> Any:
    """Return the LLM payload without persisting on cache hits."""
    return llm_data


def _llm_handle_none(*_llm_args: Any, **_llm_kwargs: Any) -> None:
    """Return no value on a cache miss for the direct `get` helper."""
    return None


def _update_cache_callback(
    llm_data: Any,
    update_cache_func: Any,
    *_args: Any,
    **_kwargs: Any,
) -> None:
    """Persist the LLM payload to cache storage."""
    update_cache_func(llm_data)


def put(prompt: str, data: Any, **kwargs: Any) -> None:
    """Put a prompt/value pair into ByteAI Cache."""

    def llm_handle(*_llm_args: Any, **_llm_kwargs: Any) -> Any:
        return data

    adapt(
        llm_handle,
        _cache_data_converter,
        _update_cache_callback,
        cache_skip=True,
        byte_disable_reasoning_shortcut=True,
        prompt=prompt,
        **kwargs,
    )


def get(prompt: str, **kwargs: Any) -> Any:
    """Get a cached value directly by prompt."""
    return adapt(
        _llm_handle_none,
        _cache_data_converter,
        _update_cache_callback_none,
        byte_disable_reasoning_shortcut=True,
        prompt=prompt,
        **kwargs,
    )
