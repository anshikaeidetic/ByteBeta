"""Event, intent-graph, and replay mixins for the Byte control-plane store."""

from __future__ import annotations

import json
from typing import Any

from byte_server._control_plane_scope import RequestScope
from byte_server._control_plane_store_common import _json_dumps, _now


class _ControlPlaneStoreEventMixin:
    def record_cache_event(
        self,
        *,
        scope: RequestScope,
        route_key: str,
        event_type: str,
        reason: str,
        cache_hit: bool,
        latency_ms: float,
        worker_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        payload = {
            "tenant": str(scope.tenant or ""),
            "session_id": str(scope.session or ""),
            "workflow_id": str(scope.workflow or ""),
            "route_key": str(route_key or ""),
            "event_type": str(event_type or "request"),
            "reason": str(reason or ""),
            "cache_hit": bool(cache_hit),
            "latency_ms": round(float(latency_ms or 0.0), 4),
            "worker_id": str(worker_id or ""),
            "metadata": dict(metadata or {}),
            "created_at": now,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_inspection_events(
                    tenant, session_id, workflow_id, route_key, event_type,
                    reason, cache_hit, latency_ms, worker_id, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["tenant"],
                    payload["session_id"],
                    payload["workflow_id"],
                    payload["route_key"],
                    payload["event_type"],
                    payload["reason"],
                    1 if payload["cache_hit"] else 0,
                    payload["latency_ms"],
                    payload["worker_id"],
                    _json_dumps(payload["metadata"]),
                    payload["created_at"],
                ),
            )
            conn.commit()
        return payload

    def list_cache_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cache_inspection_events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit or 1)),),
            ).fetchall()
        return [
            {
                "tenant": str(row["tenant"]),
                "session_id": str(row["session_id"]),
                "workflow_id": str(row["workflow_id"]),
                "route_key": str(row["route_key"]),
                "event_type": str(row["event_type"]),
                "reason": str(row["reason"]),
                "cache_hit": bool(int(row["cache_hit"] or 0)),
                "latency_ms": float(row["latency_ms"] or 0.0),
                "worker_id": str(row["worker_id"] or ""),
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": float(row["created_at"] or 0.0),
            }
            for row in rows
        ]

    def record_intent_edge(self, *, tenant: str, source_intent: str, target_intent: str) -> None:
        tenant = str(tenant or "").strip()
        source_intent = str(source_intent or "").strip()
        target_intent = str(target_intent or "").strip()
        if not tenant or not source_intent or not target_intent:
            return
        now = _now()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT edge_count FROM intent_graph_edges
                WHERE tenant = ? AND source_intent = ? AND target_intent = ?
                """,
                (tenant, source_intent, target_intent),
            ).fetchone()
            count = int((row["edge_count"] if row else 0) or 0) + 1
            conn.execute(
                """
                INSERT INTO intent_graph_edges(tenant, source_intent, target_intent, edge_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tenant, source_intent, target_intent) DO UPDATE SET
                    edge_count = excluded.edge_count,
                    updated_at = excluded.updated_at
                """,
                (tenant, source_intent, target_intent, count, now),
            )
            conn.commit()

    def intent_graph_snapshot(self, limit: int = 25) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tenant, source_intent, target_intent, edge_count, updated_at
                FROM intent_graph_edges
                ORDER BY edge_count DESC, updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit or 1)),),
            ).fetchall()
        edges = [
            {
                "tenant": str(row["tenant"]),
                "source_intent": str(row["source_intent"]),
                "target_intent": str(row["target_intent"]),
                "edge_count": int(row["edge_count"] or 0),
                "updated_at": float(row["updated_at"] or 0.0),
            }
            for row in rows
        ]
        return {
            "edges": edges,
            "edge_count": len(edges),
            "tracked_tenants": len({edge["tenant"] for edge in edges}),
        }

    def create_replay_job(
        self,
        *,
        scope: RequestScope,
        route_key: str,
        feature: str,
        request_digest: str,
        projected_savings: float,
    ) -> dict[str, Any]:
        now = _now()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO replay_jobs(
                    tenant, session_id, workflow_id, route_key, feature,
                    status, request_digest, projected_savings, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(scope.tenant or ""),
                    str(scope.session or ""),
                    str(scope.workflow or ""),
                    str(route_key or ""),
                    str(feature or "shadow"),
                    "queued",
                    str(request_digest or ""),
                    float(projected_savings or 0.0),
                    now,
                ),
            )
            conn.commit()
            job_id = int(cursor.lastrowid or 0)
        return {
            "id": job_id,
            "tenant": scope.tenant,
            "session_id": scope.session,
            "workflow_id": scope.workflow,
            "route_key": route_key,
            "feature": feature,
            "status": "queued",
            "request_digest": request_digest,
            "projected_savings": float(projected_savings or 0.0),
            "created_at": now,
        }

    def complete_replay_job(
        self,
        *,
        job_id: int,
        deterministic_score: float,
        evidence_score: float,
        error_rate: float,
        error_rate_upper: float,
        judge_required: bool,
        projected_savings: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        payload = {
            "job_id": int(job_id or 0),
            "deterministic_score": round(
                max(0.0, min(1.0, float(deterministic_score or 0.0))), 4
            ),
            "evidence_score": round(
                max(0.0, min(1.0, float(evidence_score or 0.0))), 4
            ),
            "error_rate": round(max(0.0, min(1.0, float(error_rate or 0.0))), 6),
            "error_rate_upper": round(
                max(0.0, min(1.0, float(error_rate_upper or 0.0))), 6
            ),
            "judge_required": bool(judge_required),
            "projected_savings": round(float(projected_savings or 0.0), 6),
            "metadata": dict(metadata or {}),
            "created_at": now,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE replay_jobs SET status = ? WHERE id = ?",
                ("completed", payload["job_id"]),
            )
            conn.execute(
                """
                INSERT INTO replay_outcomes(
                    job_id, deterministic_score, evidence_score, error_rate,
                    error_rate_upper, judge_required, projected_savings, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    deterministic_score=excluded.deterministic_score,
                    evidence_score=excluded.evidence_score,
                    error_rate=excluded.error_rate,
                    error_rate_upper=excluded.error_rate_upper,
                    judge_required=excluded.judge_required,
                    projected_savings=excluded.projected_savings,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at
                """,
                (
                    payload["job_id"],
                    payload["deterministic_score"],
                    payload["evidence_score"],
                    payload["error_rate"],
                    payload["error_rate_upper"],
                    1 if payload["judge_required"] else 0,
                    payload["projected_savings"],
                    _json_dumps(payload["metadata"]),
                    payload["created_at"],
                ),
            )
            conn.commit()
        return payload

    def list_replays(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT j.*, o.deterministic_score, o.evidence_score, o.error_rate,
                       o.error_rate_upper, o.judge_required,
                       o.metadata_json AS outcome_metadata_json
                FROM replay_jobs AS j
                LEFT JOIN replay_outcomes AS o ON o.job_id = j.id
                ORDER BY j.created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit or 1)),),
            ).fetchall()
        payload = []
        for row in rows:
            payload.append(
                {
                    "id": int(row["id"] or 0),
                    "tenant": str(row["tenant"]),
                    "session_id": str(row["session_id"]),
                    "workflow_id": str(row["workflow_id"]),
                    "route_key": str(row["route_key"]),
                    "feature": str(row["feature"]),
                    "status": str(row["status"]),
                    "request_digest": str(row["request_digest"]),
                    "projected_savings": float(row["projected_savings"] or 0.0),
                    "created_at": float(row["created_at"] or 0.0),
                    "deterministic_score": float(row["deterministic_score"] or 0.0),
                    "evidence_score": float(row["evidence_score"] or 0.0),
                    "error_rate": float(row["error_rate"] or 0.0),
                    "error_rate_upper": float(row["error_rate_upper"] or 0.0),
                    "judge_required": bool(int(row["judge_required"] or 0)),
                    "outcome_metadata": json.loads(
                        row["outcome_metadata_json"] or "{}"
                    ),
                }
            )
        return payload
