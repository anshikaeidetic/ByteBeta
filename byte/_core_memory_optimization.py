"""Optimization-memory methods for the Cache mixin."""

from __future__ import annotations

from typing import Any

from byte.security import sanitize_structure


class _CacheOptimizationMemoryMixin:
    def remember_prompt_pieces(
        self,
        pieces: list[dict[str, Any]],
        *,
        source: str = "request",
    ) -> list[dict[str, Any]]:
        if self.prompt_piece_store is None:
            self._init_memory_runtime(self.config)
        return self.prompt_piece_store.remember_many(
            pieces,
            scope=self.memory_scope or "",
            source=source,
        )

    def prompt_piece_stats(self) -> dict[str, Any]:
        if self.prompt_piece_store is None:
            return {}
        return self.prompt_piece_store.stats()

    def prompt_module_stats(self) -> dict[str, Any]:
        if self.prompt_module_registry is None:
            return {}
        return self.prompt_module_registry.stats()

    def remember_artifact(
        self,
        artifact_type: str,
        value: Any,
        *,
        fingerprint: str = "",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "request",
    ) -> dict[str, Any]:
        if self.artifact_memory_store is None:
            self._init_memory_runtime(self.config)
        if self.config.security_redact_memory:
            value = sanitize_structure(value)
            metadata = sanitize_structure(metadata or {})
        return self.artifact_memory_store.remember(
            artifact_type,
            value,
            fingerprint=fingerprint,
            summary=summary,
            scope=self.memory_scope or "",
            metadata=metadata,
            source=source,
        )

    def recall_artifact(self, artifact_type: str, *, fingerprint: str) -> dict[str, Any] | None:
        if self.artifact_memory_store is None:
            self._init_memory_runtime(self.config)
        return self.artifact_memory_store.get(
            artifact_type,
            fingerprint=fingerprint,
            scope=self.memory_scope or "",
        )

    def artifact_memory_stats(self) -> dict[str, Any]:
        if self.artifact_memory_store is None:
            return {}
        return self.artifact_memory_store.stats()

    def remember_workflow_plan(
        self,
        request_kwargs: dict[str, Any],
        *,
        action: str,
        route_preference: str = "",
        counterfactual_action: str = "",
        counterfactual_reason: str = "",
        repo_fingerprint: str = "",
        artifact_fingerprint: str = "",
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.workflow_plan_store is None:
            self._init_memory_runtime(self.config)
        if self.config.security_redact_memory:
            metadata = sanitize_structure(metadata or {})
        return self.workflow_plan_store.remember(
            request_kwargs,
            action=action,
            route_preference=route_preference,
            counterfactual_action=counterfactual_action,
            counterfactual_reason=counterfactual_reason,
            repo_fingerprint=repo_fingerprint,
            artifact_fingerprint=artifact_fingerprint,
            scope=self.memory_scope or "",
            success=success,
            metadata=metadata,
        )

    def workflow_plan_hint(
        self,
        request_kwargs: dict[str, Any],
        *,
        repo_fingerprint: str = "",
        artifact_fingerprint: str = "",
    ) -> dict[str, Any]:
        if self.workflow_plan_store is None:
            self._init_memory_runtime(self.config)
        return self.workflow_plan_store.hint(
            request_kwargs,
            repo_fingerprint=repo_fingerprint,
            artifact_fingerprint=artifact_fingerprint,
            scope=self.memory_scope or "",
        )

    def workflow_plan_stats(self) -> dict[str, Any]:
        if self.workflow_plan_store is None:
            return {}
        return self.workflow_plan_store.stats()

    def note_session_delta(
        self,
        session_key: str,
        artifact_type: str,
        value: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.session_delta_store is None:
            self._init_memory_runtime(self.config)
        if self.config.security_redact_memory:
            metadata = sanitize_structure(metadata or {})
        return self.session_delta_store.note(
            session_key,
            artifact_type,
            value,
            scope=self.memory_scope or "",
            metadata=metadata,
        )

    def session_delta_stats(self) -> dict[str, Any]:
        if self.session_delta_store is None:
            return {}
        return self.session_delta_store.stats()

__all__ = ["_CacheOptimizationMemoryMixin"]
