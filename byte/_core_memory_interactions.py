"""Interaction memory methods for the Cache mixin."""

from __future__ import annotations

from typing import Any

from byte.security import redact_memory_snapshot, sanitize_structure


class _CacheInteractionMemoryMixin:
    def remember_interaction(
        self,
        request_kwargs: dict[str, Any],
        *,
        answer: Any,
        reasoning: Any | None = None,
        tool_outputs: Any | None = None,
        embedding_data: Any | None = None,
        model: str = "",
        provider: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "llm",
    ) -> dict[str, Any]:
        if self.ai_memory_store is None:
            self._init_memory_runtime(self.config)
        if self.config.security_redact_memory:
            answer = sanitize_structure(answer)
            reasoning = sanitize_structure(reasoning)
            tool_outputs = sanitize_structure(tool_outputs)
            metadata = sanitize_structure(metadata or {})
        return self.ai_memory_store.remember(
            request_kwargs,
            answer=answer,
            reasoning=reasoning,
            tool_outputs=tool_outputs,
            embedding_data=embedding_data,
            model=model,
            provider=provider,
            scope=self.memory_scope or "",
            metadata=metadata,
            source=source,
        )

    def touch_interaction(
        self,
        request_kwargs: dict[str, Any],
        *,
        answer: Any | None = None,
        reasoning: Any | None = None,
        tool_outputs: Any | None = None,
        embedding_data: Any | None = None,
        model: str = "",
        provider: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "cache",
    ) -> dict[str, Any]:
        if self.ai_memory_store is None:
            self._init_memory_runtime(self.config)
        if self.config.security_redact_memory:
            answer = sanitize_structure(answer)
            reasoning = sanitize_structure(reasoning)
            tool_outputs = sanitize_structure(tool_outputs)
            metadata = sanitize_structure(metadata or {})
        return self.ai_memory_store.touch(
            request_kwargs,
            answer=answer,
            reasoning=reasoning,
            tool_outputs=tool_outputs,
            embedding_data=embedding_data,
            model=model,
            provider=provider,
            scope=self.memory_scope or "",
            metadata=metadata,
            source=source,
        )

    def ai_memory_stats(self) -> dict[str, Any]:
        if self.ai_memory_store is None:
            return {}
        return self.ai_memory_store.stats()

    def recent_interactions(self, limit: int = 10) -> list[dict[str, Any]]:
        if self.ai_memory_store is None:
            return []
        entries = self.ai_memory_store.recent(limit=limit)
        if self.config.security_redact_memory:
            payload = redact_memory_snapshot({"ai_memory": {"entries": entries, "stats": {}}})
            return payload["ai_memory"]["entries"]
        return entries

__all__ = ["_CacheInteractionMemoryMixin"]
