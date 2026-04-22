"""Intent and tool-result memory methods for the Cache mixin."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _CacheToolMemoryMixin:
    def record_intent(
        self, request_kwargs: dict[str, Any], session_id: str | None = None
    ) -> dict[str, Any]:
        """Record the request into the shared/local intent graph."""
        if self.intent_graph is None:
            self._init_memory_runtime(self.config)
        return self.intent_graph.record(request_kwargs, session_id=session_id)

    def intent_stats(self) -> dict[str, Any]:
        if self.intent_graph is None:
            return {}
        return self.intent_graph.stats()

    def remember_tool_result(
        self,
        tool_name: str,
        tool_args: Any,
        result: Any,
        *,
        ttl: float | None = None,
        scope: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.tool_result_store is None:
            self._init_memory_runtime(self.config)
        effective_ttl = self.config.tool_result_ttl if ttl is None else ttl
        effective_scope = self.memory_scope if scope is None else scope
        return self.tool_result_store.put(
            tool_name,
            tool_args,
            result,
            ttl=effective_ttl,
            scope=effective_scope or "",
            metadata=metadata,
        )

    def recall_tool_result(
        self,
        tool_name: str,
        tool_args: Any,
        *,
        scope: str | None = None,
        include_metadata: bool = False,
    ) -> Any:
        if self.tool_result_store is None:
            self._init_memory_runtime(self.config)
        effective_scope = self.memory_scope if scope is None else scope
        entry = self.tool_result_store.get(tool_name, tool_args, scope=effective_scope or "")
        if not entry:
            return None
        if include_metadata:
            return entry
        return entry["result"]

    def run_tool(
        self,
        tool_name: str,
        tool_args: Any,
        tool_func: Callable,
        *,
        ttl: float | None = None,
        scope: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, bool]:
        cached = self.recall_tool_result(
            tool_name,
            tool_args,
            scope=scope,
            include_metadata=True,
        )
        if cached is not None:
            return cached["result"], True

        if isinstance(tool_args, dict):
            result = tool_func(**tool_args)
        elif isinstance(tool_args, (list, tuple)):
            result = tool_func(*tool_args)
        elif tool_args is None:
            result = tool_func()
        else:
            result = tool_func(tool_args)

        self.remember_tool_result(
            tool_name,
            tool_args,
            result,
            ttl=ttl,
            scope=scope,
            metadata=metadata,
        )
        return result, False

    def tool_memory_stats(self) -> dict[str, Any]:
        if self.tool_result_store is None:
            return {}
        return self.tool_result_store.stats()

__all__ = ["_CacheToolMemoryMixin"]
