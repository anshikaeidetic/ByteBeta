"""Schema and mirror-loading mixins for the Byte control-plane store."""

from __future__ import annotations

import json


class _ControlPlaneStoreSchemaMixin:
    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    health_score REAL NOT NULL,
                    queue_depth INTEGER NOT NULL,
                    free_vram_gb REAL NOT NULL,
                    model_inventory_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    last_error TEXT NOT NULL,
                    last_heartbeat REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worker_leases (
                    tenant TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant, session_id, model_family)
                );
                CREATE TABLE IF NOT EXISTS kv_residency (
                    tenant TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    compression_profile TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    prompt_digest TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant, session_id, model_name, compression_profile)
                );
                CREATE TABLE IF NOT EXISTS replay_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    route_key TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    projected_savings REAL NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS replay_outcomes (
                    job_id INTEGER PRIMARY KEY,
                    deterministic_score REAL NOT NULL,
                    evidence_score REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    error_rate_upper REAL NOT NULL,
                    judge_required INTEGER NOT NULL,
                    projected_savings REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES replay_jobs(id)
                );
                CREATE TABLE IF NOT EXISTS threshold_recommendations (
                    tenant TEXT NOT NULL,
                    route_key TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    suggested_threshold REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    projected_savings REAL NOT NULL,
                    error_rate REAL NOT NULL,
                    error_rate_upper REAL NOT NULL,
                    auto_apply_eligible INTEGER NOT NULL,
                    details_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant, route_key, feature)
                );
                CREATE TABLE IF NOT EXISTS cache_inspection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    route_key TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    worker_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS intent_graph_edges (
                    tenant TEXT NOT NULL,
                    source_intent TEXT NOT NULL,
                    target_intent TEXT NOT NULL,
                    edge_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (tenant, source_intent, target_intent)
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.commit()

    def _load_mirrors(self) -> None:
        with self._lock, self._connect() as conn:
            self._workers = {}
            for row in conn.execute("SELECT * FROM workers"):
                self._workers[str(row["worker_id"])] = {
                    "worker_id": str(row["worker_id"]),
                    "url": str(row["url"]),
                    "status": str(row["status"]),
                    "health_score": float(row["health_score"] or 0.0),
                    "queue_depth": int(row["queue_depth"] or 0),
                    "free_vram_gb": float(row["free_vram_gb"] or 0.0),
                    "model_inventory": json.loads(row["model_inventory_json"] or "[]"),
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "last_error": str(row["last_error"] or ""),
                    "last_heartbeat": float(row["last_heartbeat"] or 0.0),
                }
            self._leases = {}
            for row in conn.execute("SELECT * FROM worker_leases"):
                self._leases[
                    (str(row["tenant"]), str(row["session_id"]), str(row["model_family"]))
                ] = {
                    "worker_id": str(row["worker_id"]),
                    "updated_at": float(row["updated_at"] or 0.0),
                }
            self._kv_residency = {}
            for row in conn.execute("SELECT * FROM kv_residency"):
                self._kv_residency[
                    (
                        str(row["tenant"]),
                        str(row["session_id"]),
                        str(row["model_name"]),
                        str(row["compression_profile"]),
                    )
                ] = {
                    "worker_id": str(row["worker_id"]),
                    "prompt_digest": str(row["prompt_digest"]),
                    "updated_at": float(row["updated_at"] or 0.0),
                }
            self._recommendations = {}
            for row in conn.execute("SELECT * FROM threshold_recommendations"):
                self._recommendations[
                    (str(row["tenant"]), str(row["route_key"]), str(row["feature"]))
                ] = {
                    "tenant": str(row["tenant"]),
                    "route_key": str(row["route_key"]),
                    "feature": str(row["feature"]),
                    "suggested_threshold": float(row["suggested_threshold"] or 0.0),
                    "sample_count": int(row["sample_count"] or 0),
                    "projected_savings": float(row["projected_savings"] or 0.0),
                    "error_rate": float(row["error_rate"] or 0.0),
                    "error_rate_upper": float(row["error_rate_upper"] or 0.0),
                    "auto_apply_eligible": bool(int(row["auto_apply_eligible"] or 0)),
                    "details": json.loads(row["details_json"] or "{}"),
                    "updated_at": float(row["updated_at"] or 0.0),
                }
            self._settings_cache = {}
            for row in conn.execute("SELECT * FROM settings"):
                self._settings_cache[str(row["key"])] = json.loads(row["value_json"] or "{}")
