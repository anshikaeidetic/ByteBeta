"""Worker and KV residency mixins for the Byte control-plane store."""

from __future__ import annotations

from typing import Any

from byte_server._control_plane_routing import (
    WorkerSelection,
    _model_family,
    _short_digest,
    _worker_supports_model,
)
from byte_server._control_plane_scope import RequestScope
from byte_server._control_plane_store_common import _json_dumps, _now


class _ControlPlaneStoreWorkerMixin:
    def upsert_worker(self, heartbeat: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "worker_id": str(heartbeat.get("worker_id") or "").strip()
            or _short_digest(str(heartbeat)),
            "url": str(heartbeat.get("url") or "").strip(),
            "status": str(heartbeat.get("status", "ready") or "ready").strip(),
            "health_score": max(
                0.0, min(1.0, float(heartbeat.get("health_score", 1.0) or 1.0))
            ),
            "queue_depth": max(0, int(heartbeat.get("queue_depth", 0) or 0)),
            "free_vram_gb": max(
                0.0, float(heartbeat.get("free_vram_gb", 0.0) or 0.0)
            ),
            "model_inventory": [
                str(item).strip()
                for item in (heartbeat.get("model_inventory") or [])
                if str(item).strip()
            ],
            "metadata": dict(heartbeat.get("metadata", {}) or {}),
            "last_error": str(heartbeat.get("last_error", "") or ""),
            "last_heartbeat": float(heartbeat.get("last_heartbeat", _now()) or _now()),
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workers(
                    worker_id, url, status, health_score, queue_depth, free_vram_gb,
                    model_inventory_json, metadata_json, last_error, last_heartbeat
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    url=excluded.url,
                    status=excluded.status,
                    health_score=excluded.health_score,
                    queue_depth=excluded.queue_depth,
                    free_vram_gb=excluded.free_vram_gb,
                    model_inventory_json=excluded.model_inventory_json,
                    metadata_json=excluded.metadata_json,
                    last_error=excluded.last_error,
                    last_heartbeat=excluded.last_heartbeat
                """,
                (
                    payload["worker_id"],
                    payload["url"],
                    payload["status"],
                    payload["health_score"],
                    payload["queue_depth"],
                    payload["free_vram_gb"],
                    _json_dumps(payload["model_inventory"]),
                    _json_dumps(payload["metadata"]),
                    payload["last_error"],
                    payload["last_heartbeat"],
                ),
            )
            conn.commit()
            self._workers[payload["worker_id"]] = dict(payload)
        return payload

    def list_workers(self) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._workers.values())
        items.sort(
            key=lambda item: (
                float(item.get("health_score", 0.0) or 0.0),
                float(item.get("free_vram_gb", 0.0) or 0.0),
                -int(item.get("queue_depth", 0) or 0),
                float(item.get("last_heartbeat", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return items

    def record_worker_lease(
        self, *, tenant: str, session_id: str, model_family: str, worker_id: str
    ) -> None:
        now = _now()
        tenant = str(tenant or "").strip()
        session_id = str(session_id or "").strip()
        model_family = str(model_family or "").strip()
        worker_id = str(worker_id or "").strip()
        if not tenant or not session_id or not model_family or not worker_id:
            return
        key = (tenant, session_id, model_family)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO worker_leases(tenant, session_id, model_family, worker_id, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tenant, session_id, model_family) DO UPDATE SET
                    worker_id=excluded.worker_id,
                    updated_at=excluded.updated_at
                """,
                (tenant, session_id, model_family, worker_id, now),
            )
            conn.commit()
            self._leases[key] = {"worker_id": worker_id, "updated_at": now}

    def record_kv_residency(
        self,
        *,
        tenant: str,
        session_id: str,
        model_name: str,
        compression_profile: str,
        worker_id: str,
        prompt_digest: str,
    ) -> None:
        now = _now()
        tenant = str(tenant or "").strip()
        session_id = str(session_id or "").strip()
        model_name = str(model_name or "").strip()
        compression_profile = str(compression_profile or "disabled")
        worker_id = str(worker_id or "").strip()
        prompt_digest = str(prompt_digest or "").strip()
        if not tenant or not session_id or not model_name or not worker_id:
            return
        key = (tenant, session_id, model_name, compression_profile)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kv_residency(
                    tenant, session_id, model_name, compression_profile,
                    worker_id, prompt_digest, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant, session_id, model_name, compression_profile) DO UPDATE SET
                    worker_id=excluded.worker_id,
                    prompt_digest=excluded.prompt_digest,
                    updated_at=excluded.updated_at
                """,
                (
                    tenant,
                    session_id,
                    model_name,
                    compression_profile,
                    worker_id,
                    prompt_digest,
                    now,
                ),
            )
            conn.commit()
            self._kv_residency[key] = {
                "worker_id": worker_id,
                "prompt_digest": prompt_digest,
                "updated_at": now,
            }

    def choose_worker(
        self,
        *,
        scope: RequestScope,
        model_name: str,
        compression_profile: str,
    ) -> WorkerSelection | None:
        tenant = str(scope.tenant or "").strip()
        session_id = str(scope.session or "").strip()
        model_name = str(model_name or "").strip()
        if not tenant or not session_id or not model_name:
            return None
        model_family = _model_family(model_name)
        with self._lock:
            kv_hit = self._kv_residency.get(
                (tenant, session_id, model_name, compression_profile)
            )
            if kv_hit:
                worker = self._workers.get(str(kv_hit.get("worker_id") or ""))
                if worker and _worker_supports_model(worker, model_name):
                    return WorkerSelection(
                        worker_id=str(worker["worker_id"]),
                        url=str(worker["url"]),
                        source="kv_residency",
                        model_name=model_name,
                        score=1.0,
                        queue_depth=int(worker.get("queue_depth", 0) or 0),
                        free_vram_gb=float(worker.get("free_vram_gb", 0.0) or 0.0),
                        model_inventory=list(worker.get("model_inventory", []) or []),
                    )

            lease = self._leases.get((tenant, session_id, model_family))
            if lease:
                worker = self._workers.get(str(lease.get("worker_id") or ""))
                if worker and _worker_supports_model(worker, model_name):
                    return WorkerSelection(
                        worker_id=str(worker["worker_id"]),
                        url=str(worker["url"]),
                        source="session_affinity",
                        model_name=model_name,
                        score=max(
                            0.0, float(worker.get("health_score", 0.0) or 0.0)
                        ),
                        queue_depth=int(worker.get("queue_depth", 0) or 0),
                        free_vram_gb=float(worker.get("free_vram_gb", 0.0) or 0.0),
                        model_inventory=list(worker.get("model_inventory", []) or []),
                    )

            candidates = [
                worker
                for worker in self._workers.values()
                if _worker_supports_model(worker, model_name)
            ]
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(item.get("health_score", 0.0) or 0.0)
                + min(float(item.get("free_vram_gb", 0.0) or 0.0) / 100.0, 0.25)
                - min(int(item.get("queue_depth", 0) or 0) / 20.0, 0.35)
            ),
            reverse=True,
        )
        best = ranked[0]
        score = (
            float(best.get("health_score", 0.0) or 0.0)
            + min(float(best.get("free_vram_gb", 0.0) or 0.0) / 100.0, 0.25)
            - min(int(best.get("queue_depth", 0) or 0) / 20.0, 0.35)
        )
        return WorkerSelection(
            worker_id=str(best["worker_id"]),
            url=str(best["url"]),
            source="health_weighted",
            model_name=model_name,
            score=round(score, 4),
            queue_depth=int(best.get("queue_depth", 0) or 0),
            free_vram_gb=float(best.get("free_vram_gb", 0.0) or 0.0),
            model_inventory=list(best.get("model_inventory", []) or []),
        )
