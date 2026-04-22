"""Shared run-state types for the adapter pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheAnswerMatch:
    """Resolved cache candidate that survived ranking and validation."""

    rank: float
    answer: Any
    search_data: Any
    cache_data: Any


@dataclass
class PipelineRunState:
    """Mutable state carried through a single pipeline invocation."""

    llm_handler: Any
    cache_data_convert: Any
    update_cache_callback: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    chat_cache: Any
    context: dict[str, Any]
    start_time: float
    search_only_flag: bool
    user_temperature: bool
    user_top_k: bool
    requested_top_k: Any
    temperature: float
    session: Any
    require_object_store: bool
    cache_enable: bool
    cache_skip: bool
    cache_factor: float
    pre_store_data: Any = None
    pre_embedding_data: Any = None
    embedding_data: Any = None
    coalesce_key: str | None = None
    cache_stage_started_at: float | None = None
    llm_data: Any = None
    response_assessment: Any = None
    route_decision: Any = None
    task_policy: dict[str, Any] = field(default_factory=dict)
    pending_cache_tasks: list[Any] = field(default_factory=list)
    event_loop: Any = None


__all__ = ["CacheAnswerMatch", "PipelineRunState"]
