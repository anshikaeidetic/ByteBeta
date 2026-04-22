"""Execution-result memory methods for the Cache mixin."""

from __future__ import annotations

from typing import Any

from byte.security import sanitize_structure


class _CacheExecutionMemoryMixin:
    def remember_execution_result(
        self,
        request_kwargs: dict[str, Any],
        *,
        answer: Any,
        verification: Any | None = None,
        patch: Any | None = None,
        test_command: Any | None = None,
        test_result: Any | None = None,
        lint_result: Any | None = None,
        schema_validation: Any | None = None,
        tool_checks: Any | None = None,
        repo_fingerprint: str = "",
        model: str = "",
        provider: str = "",
        metadata: dict[str, Any] | None = None,
        source: str = "llm",
    ) -> dict[str, Any]:
        if self.execution_memory_store is None:
            self._init_memory_runtime(self.config)
        if self.config.security_redact_memory:
            answer = sanitize_structure(answer)
            verification = sanitize_structure(verification)
            patch = sanitize_structure(patch)
            test_command = sanitize_structure(test_command)
            test_result = sanitize_structure(test_result)
            lint_result = sanitize_structure(lint_result)
            schema_validation = sanitize_structure(schema_validation)
            tool_checks = sanitize_structure(tool_checks)
            metadata = sanitize_structure(metadata or {})
        entry = self.execution_memory_store.remember(
            request_kwargs,
            answer=answer,
            verification=verification,
            patch=patch,
            test_command=test_command,
            test_result=test_result,
            lint_result=lint_result,
            schema_validation=schema_validation,
            tool_checks=tool_checks,
            repo_fingerprint=repo_fingerprint,
            model=model,
            provider=provider,
            scope=self.memory_scope or "",
            metadata=metadata,
            source=source,
        )
        if (
            patch not in (None, "")
            and self.patch_pattern_store is not None
            and entry.get("verified")
        ):
            self.patch_pattern_store.remember(
                request_kwargs,
                patch=patch,
                repo_fingerprint=repo_fingerprint,
                verified=True,
                model=model,
                provider=provider,
                scope=self.memory_scope or "",
                metadata=metadata,
            )
        return entry

    def lookup_execution_result(
        self,
        request_kwargs: dict[str, Any],
        *,
        answer: Any | None = None,
        repo_fingerprint: str = "",
        model: str = "",
        verified_only: bool = False,
    ) -> dict[str, Any] | None:
        if self.execution_memory_store is None:
            self._init_memory_runtime(self.config)
        return self.execution_memory_store.lookup(
            request_kwargs,
            answer=answer,
            repo_fingerprint=repo_fingerprint,
            model=model,
            scope=self.memory_scope or "",
            verified_only=verified_only,
        )

    def execution_memory_stats(self) -> dict[str, Any]:
        if self.execution_memory_store is None:
            return {}
        return self.execution_memory_store.stats()

__all__ = ["_CacheExecutionMemoryMixin"]
