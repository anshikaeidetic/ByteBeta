"""Failure and patch-pattern memory methods for the Cache mixin."""

from __future__ import annotations

from typing import Any

from byte.security import sanitize_structure


class _CachePatternMemoryMixin:
    def remember_failure_pattern(
        self,
        request_kwargs: dict[str, Any],
        *,
        reason: str,
        provider: str = "",
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.failure_memory_store is None:
            self._init_memory_runtime(self.config)
        if self.config.security_redact_memory:
            metadata = sanitize_structure(metadata or {})
        return self.failure_memory_store.record(  # type: ignore
            request_kwargs,
            reason=reason,
            provider=provider,
            model=model,
            scope=self.memory_scope or "",
            metadata=metadata,
        )

    def failure_memory_hint(
        self,
        request_kwargs: dict[str, Any],
        *,
        provider: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        if self.failure_memory_store is None:
            self._init_memory_runtime(self.config)
        return self.failure_memory_store.hint(
            request_kwargs,
            provider=provider,
            model=model,
            scope=self.memory_scope or "",
        )

    def failure_memory_stats(self) -> dict[str, Any]:
        if self.failure_memory_store is None:
            return {}
        return self.failure_memory_store.stats()

    def remember_patch_pattern(
        self,
        request_kwargs: dict[str, Any],
        *,
        patch: Any,
        repo_fingerprint: str = "",
        verified: bool = False,
        model: str = "",
        provider: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self.patch_pattern_store is None:
            self._init_memory_runtime(self.config)
        return self.patch_pattern_store.remember(
            request_kwargs,
            patch=patch,
            repo_fingerprint=repo_fingerprint,
            verified=verified,
            model=model,
            provider=provider,
            scope=self.memory_scope or "",
            metadata=metadata,
        )

    def suggest_patch_pattern(
        self,
        request_kwargs: dict[str, Any],
        *,
        repo_fingerprint: str = "",
        verified_only: bool = True,
    ) -> dict[str, Any] | None:
        if self.patch_pattern_store is None:
            self._init_memory_runtime(self.config)
        return self.patch_pattern_store.suggest(
            request_kwargs,
            repo_fingerprint=repo_fingerprint,
            scope=self.memory_scope or "",
            verified_only=verified_only,
        )

    def patch_pattern_stats(self) -> dict[str, Any]:
        if self.patch_pattern_store is None:
            return {}
        return self.patch_pattern_store.stats()

__all__ = ["_CachePatternMemoryMixin"]
