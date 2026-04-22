"""Memory and memoization helpers for the public adapter API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from byte import Cache, cache


def remember_tool_result(tool_name: str, tool_args: Any, result: Any, **kwargs) -> dict[str, Any]:
    """Store a deterministic tool result in ByteAI Cache's shared tool-result memory."""
    cache_obj = kwargs.pop("cache_obj", cache)
    return cache_obj.remember_tool_result(tool_name, tool_args, result, **kwargs)


def recall_tool_result(tool_name: str, tool_args: Any, **kwargs) -> Any:
    """Recall a tool result from ByteAI Cache's provider-agnostic tool-result memory."""
    cache_obj = kwargs.pop("cache_obj", cache)
    return cache_obj.recall_tool_result(tool_name, tool_args, **kwargs)


def run_tool(tool_name: str, tool_args: Any, tool_func: Callable[..., Any], **kwargs) -> Any:
    """Run a deterministic tool through ByteAI Cache so repeated tool calls become cache hits."""
    cache_obj = kwargs.pop("cache_obj", cache)
    return cache_obj.run_tool(tool_name, tool_args, tool_func, **kwargs)


def remember_retrieval_result(query: Any, result: Any, **kwargs) -> dict[str, Any]:
    """Store retrieval outputs so unique prompts can still reuse expensive search work."""
    cache_obj = kwargs.pop("cache_obj", cache)
    scope = kwargs.pop("scope", None)
    metadata = kwargs.pop("metadata", None)
    return cache_obj.remember_tool_result(
        "retrieval.search",
        {"query": query, **kwargs},
        result,
        scope=scope,
        metadata=metadata,
    )


def recall_retrieval_result(query: Any, **kwargs) -> Any:
    """Recall a retrieval result from ByteAI Cache's retrieval memo store."""
    cache_obj = kwargs.pop("cache_obj", cache)
    scope = kwargs.pop("scope", None)
    include_metadata = kwargs.pop("include_metadata", False)
    return cache_obj.recall_tool_result(
        "retrieval.search",
        {"query": query, **kwargs},
        scope=scope,
        include_metadata=include_metadata,
    )


def run_retrieval(query: Any, retrieval_func: Callable[..., Any], **kwargs) -> Any:
    """Run retrieval through ByteAI Cache so repeated search calls hit shared memory."""
    cache_obj = kwargs.pop("cache_obj", cache)
    scope = kwargs.pop("scope", None)
    metadata = kwargs.pop("metadata", None)
    return cache_obj.run_tool(
        "retrieval.search",
        {"query": query, **kwargs},
        retrieval_func,
        scope=scope,
        metadata=metadata,
    )


def remember_rerank_result(query: Any, candidates: Any, result: Any, **kwargs) -> dict[str, Any]:
    """Store rerank outputs for repeated retrieval pipelines."""
    cache_obj = kwargs.pop("cache_obj", cache)
    scope = kwargs.pop("scope", None)
    metadata = kwargs.pop("metadata", None)
    return cache_obj.remember_tool_result(
        "retrieval.rerank",
        {"query": query, "candidates": candidates, **kwargs},
        result,
        scope=scope,
        metadata=metadata,
    )


def recall_rerank_result(query: Any, candidates: Any, **kwargs) -> Any:
    """Recall a rerank result from ByteAI Cache's retrieval memo store."""
    cache_obj = kwargs.pop("cache_obj", cache)
    scope = kwargs.pop("scope", None)
    include_metadata = kwargs.pop("include_metadata", False)
    return cache_obj.recall_tool_result(
        "retrieval.rerank",
        {"query": query, "candidates": candidates, **kwargs},
        scope=scope,
        include_metadata=include_metadata,
    )


def run_rerank(query: Any, candidates: Any, rerank_func: Callable[..., Any], **kwargs) -> Any:
    """Run reranking through ByteAI Cache so repeated ranking work becomes reusable."""
    cache_obj = kwargs.pop("cache_obj", cache)
    scope = kwargs.pop("scope", None)
    metadata = kwargs.pop("metadata", None)
    return cache_obj.run_tool(
        "retrieval.rerank",
        {"query": query, "candidates": candidates, **kwargs},
        rerank_func,
        scope=scope,
        metadata=metadata,
    )


def memory_summary(cache_obj: Cache | None = None) -> dict[str, Any]:
    """Return the shared/local memory-layer summary for a cache object."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.memory_summary()


def export_memory_snapshot(cache_obj: Cache | None = None, **kwargs) -> dict[str, Any]:
    """Export ByteAI Cache's provider-agnostic memory snapshot for cross-app sharing."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.export_memory_snapshot(**kwargs)


def export_memory_artifact(
    path: str, cache_obj: Cache | None = None, **kwargs
) -> dict[str, Any]:
    """Export ByteAI Cache memory to a JSON, CSV, ZIP, Parquet, or SQLite artifact."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.export_memory_artifact(path, **kwargs)


def import_memory_snapshot(
    snapshot: dict[str, Any], cache_obj: Cache | None = None
) -> dict[str, Any]:
    """Import a provider-agnostic memory snapshot into a cache object."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.import_memory_snapshot(snapshot)


def import_memory_artifact(
    path: str, cache_obj: Cache | None = None, **kwargs
) -> dict[str, Any]:
    """Import a memory artifact exported by ByteAI Cache."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.import_memory_artifact(path, **kwargs)


def remember_interaction(
    request_kwargs: dict[str, Any],
    *,
    answer: Any,
    cache_obj: Cache | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Persist a provider-agnostic AI-memory interaction entry."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.remember_interaction(request_kwargs, answer=answer, **kwargs)


def recent_interactions(limit: int = 10, cache_obj: Cache | None = None) -> Any:
    """Return the most recent AI-memory interaction entries."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.recent_interactions(limit=limit)


def remember_artifact(
    artifact_type: str, value: Any, cache_obj: Cache | None = None, **kwargs
) -> dict[str, Any]:
    """Store a compact artifact summary for repo, retrieval, and document contexts."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.remember_artifact(artifact_type, value, **kwargs)


def recall_artifact(
    artifact_type: str, *, fingerprint: str, cache_obj: Cache | None = None
) -> Any:
    """Recall a compact artifact entry by fingerprint."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.recall_artifact(artifact_type, fingerprint=fingerprint)


def remember_workflow_plan(
    request_kwargs: dict[str, Any],
    *,
    action: str,
    cache_obj: Cache | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Persist a successful workflow plan so similar requests can reuse it."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.remember_workflow_plan(request_kwargs, action=action, **kwargs)


def workflow_plan_hint(
    request_kwargs: dict[str, Any], cache_obj: Cache | None = None, **kwargs
) -> dict[str, Any]:
    """Fetch ByteAI Cache's best known workflow hint for a request."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.workflow_plan_hint(request_kwargs, **kwargs)


def note_session_delta(
    session_key: str,
    artifact_type: str,
    value: Any,
    cache_obj: Cache | None = None,
    **kwargs,
) -> Any:
    """Track whether an artifact changed since the last session turn."""
    cache_obj = cache_obj if cache_obj else cache
    return cache_obj.note_session_delta(session_key, artifact_type, value, **kwargs)


def batch_embeddings(inputs: Any, cache_obj: Cache | None = None, **kwargs) -> Any:
    """Batch embedding work with dedupe so warm/import flows stay efficient."""
    from byte.processor.batching import batch_embed

    cache_obj = cache_obj if cache_obj else cache
    extra_param = kwargs.pop("extra_param", None)
    return batch_embed(cache_obj.embedding_func, list(inputs or []), extra_param=extra_param)


def batch_rerank(query: Any, candidates: Any, rerank_func: Callable[..., Any], **kwargs) -> Any:
    """Batch rerank calls across candidate slices."""
    from byte.processor.batching import batch_rerank as _batch_rerank

    batch_size = kwargs.pop("batch_size", 16)
    extra_param = kwargs.pop("extra_param", None)
    return _batch_rerank(
        rerank_func,
        query=query,
        candidates=list(candidates or []),
        batch_size=batch_size,
        extra_param=extra_param,
    )
