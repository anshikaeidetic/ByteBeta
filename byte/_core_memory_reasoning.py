"""Reasoning-memory methods for the Cache mixin."""

from __future__ import annotations

from typing import Any

from byte.security import sanitize_structure


class _CacheReasoningMemoryMixin:
    def remember_reasoning_result(
        self,
        *,
        kind: str,
        key: str,
        answer: Any,
        verified: bool = True,
        metadata: dict[str, Any] | None = None,
        source: str = "llm",
    ) -> dict[str, Any]:
        if self.reasoning_memory_store is None:
            self._init_memory_runtime(self.config)
        if self.config.security_redact_memory:
            answer = sanitize_structure(answer)
            metadata = sanitize_structure(metadata or {})
        return self.reasoning_memory_store.remember(
            kind=kind,
            key=key,
            answer=answer,
            verified=verified,
            metadata=metadata,
            source=source,
        )

    def lookup_reasoning_result(
        self,
        *,
        key: str,
        kind: str = "",
        verified_only: bool = True,
    ) -> dict[str, Any] | None:
        if self.reasoning_memory_store is None:
            self._init_memory_runtime(self.config)
        return self.reasoning_memory_store.lookup(
            key=key,
            kind=kind,
            verified_only=verified_only,
        )

    def reasoning_memory_stats(self) -> dict[str, Any]:
        if self.reasoning_memory_store is None:
            return {}
        return self.reasoning_memory_store.stats()

__all__ = ["_CacheReasoningMemoryMixin"]
