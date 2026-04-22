"""Memory snapshot import and export methods for the Cache mixin."""

from __future__ import annotations

from typing import Any

from byte.research import research_registry_summary
from byte.security import redact_memory_snapshot, sanitize_structure


class _CacheMemorySnapshotMixin:
    def memory_summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "memory_scope": self.memory_scope or "",
            "intent_graph": self.intent_stats(),
            "research_registry": research_registry_summary(),
        }
        for spec, _ in self._iter_memory_stores():
            result[spec.summary_key] = getattr(self, spec.stats_method)()
        vector_store = getattr(self.data_manager, "v", None)
        if vector_store is not None and hasattr(vector_store, "compression_stats"):
            result["vector_compression"] = sanitize_structure(vector_store.compression_stats())
        return result

    def export_memory_snapshot(self, *, tool_result_limit: int | None = None) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "memory_scope": self.memory_scope or "",
            "intent_graph": self.intent_stats(),
        }
        for spec, store in self._iter_memory_stores():
            snapshot[spec.summary_key] = {
                "entries": [],
                "stats": getattr(self, spec.stats_method)(),
            }
            if store is None:
                continue
            if spec.supports_limit:
                snapshot[spec.summary_key] = store.snapshot(limit=tool_result_limit)
            else:
                snapshot[spec.summary_key] = store.snapshot()
        if self.config.security_redact_memory:
            snapshot = redact_memory_snapshot(snapshot)
        return snapshot

    def export_memory_artifact(
        self,
        path: str,
        *,
        format: str | None = None,
        tool_result_limit: int | None = None,
    ) -> dict[str, Any]:
        from byte.processor.memory_export import export_snapshot_artifact

        snapshot = self.export_memory_snapshot(tool_result_limit=tool_result_limit)
        return export_snapshot_artifact(
            snapshot,
            path,
            format=format,
            encryption_key=(
                self.config.security_encryption_key
                if self.config.security_encrypt_artifacts
                else None
            ),
        )

    def import_memory_snapshot(self, snapshot: dict[str, Any] | None) -> dict[str, Any]:
        snapshot = snapshot or {}
        self._ensure_memory_runtime()

        intent_payload = snapshot.get("intent_graph", {}) or {}
        if self.intent_graph is not None and intent_payload:
            self.intent_graph.merge(intent_payload)

        result = {
            "memory_scope": self.memory_scope or "",
            "intent_graph": self.intent_stats(),
        }
        for spec, store in self._iter_memory_stores():
            payload = snapshot.get(spec.summary_key, {}) or {}
            merge_result = {"imported": 0, "skipped": 0, "total_entries": 0}
            if store is not None and payload:
                merge_result = store.merge(payload)
            result[spec.summary_key] = merge_result
        return result

    def import_memory_artifact(self, path: str, *, format: str | None = None) -> dict[str, Any]:
        from byte.processor.memory_export import load_snapshot_artifact

        snapshot = load_snapshot_artifact(
            path,
            format=format,
            encryption_key=(
                self.config.security_encryption_key
                if self.config.security_encrypt_artifacts
                else None
            ),
        )
        return self.import_memory_snapshot(snapshot)

__all__ = ["_CacheMemorySnapshotMixin"]
