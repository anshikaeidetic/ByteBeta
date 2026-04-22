"""Settings and recommendation mixins for the Byte control-plane store."""

from __future__ import annotations

from typing import Any

from byte_server._control_plane_store_common import _json_dumps, _now


class _ControlPlaneStoreSettingsMixin:
    def set_setting(self, key: str, value: Any) -> dict[str, Any]:
        now = _now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (str(key), _json_dumps(value), now),
            )
            conn.commit()
            self._settings_cache[str(key)] = value
        return {"key": str(key), "value": value, "updated_at": now}

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._settings_cache.get(str(key), default)

    def upsert_recommendation(
        self,
        *,
        tenant: str,
        route_key: str,
        feature: str,
        suggested_threshold: float,
        sample_count: int,
        projected_savings: float,
        error_rate: float,
        error_rate_upper: float,
        auto_apply_eligible: bool,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _now()
        payload = {
            "tenant": str(tenant or ""),
            "route_key": str(route_key or ""),
            "feature": str(feature or "shadow"),
            "suggested_threshold": round(
                max(0.0, min(1.0, float(suggested_threshold or 0.0))), 4
            ),
            "sample_count": max(0, int(sample_count or 0)),
            "projected_savings": round(float(projected_savings or 0.0), 6),
            "error_rate": round(max(0.0, min(1.0, float(error_rate or 0.0))), 6),
            "error_rate_upper": round(
                max(0.0, min(1.0, float(error_rate_upper or 0.0))), 6
            ),
            "auto_apply_eligible": bool(auto_apply_eligible),
            "details": dict(details or {}),
            "updated_at": now,
        }
        key = (payload["tenant"], payload["route_key"], payload["feature"])
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO threshold_recommendations(
                    tenant, route_key, feature, suggested_threshold, sample_count,
                    projected_savings, error_rate, error_rate_upper, auto_apply_eligible,
                    details_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant, route_key, feature) DO UPDATE SET
                    suggested_threshold=excluded.suggested_threshold,
                    sample_count=excluded.sample_count,
                    projected_savings=excluded.projected_savings,
                    error_rate=excluded.error_rate,
                    error_rate_upper=excluded.error_rate_upper,
                    auto_apply_eligible=excluded.auto_apply_eligible,
                    details_json=excluded.details_json,
                    updated_at=excluded.updated_at
                """,
                (
                    payload["tenant"],
                    payload["route_key"],
                    payload["feature"],
                    payload["suggested_threshold"],
                    payload["sample_count"],
                    payload["projected_savings"],
                    payload["error_rate"],
                    payload["error_rate_upper"],
                    1 if payload["auto_apply_eligible"] else 0,
                    _json_dumps(payload["details"]),
                    payload["updated_at"],
                ),
            )
            conn.commit()
            self._recommendations[key] = dict(payload)
        return payload

    def list_recommendations(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                self._recommendations.values(),
                key=lambda item: (
                    float(item.get("projected_savings", 0.0) or 0.0),
                    float(item.get("updated_at", 0.0) or 0.0),
                ),
                reverse=True,
            )

    def roi_snapshot(self) -> dict[str, Any]:
        events = self.list_cache_events(limit=500)
        replays = self.list_replays(limit=500)
        cache_hits = sum(1 for item in events if item.get("cache_hit"))
        total_requests = len(events)
        routed = sum(1 for item in events if str(item.get("worker_id", "")).strip())
        projected_savings = sum(
            float(item.get("projected_savings", 0.0) or 0.0) for item in replays
        )
        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_ratio": round(cache_hits / total_requests, 4)
            if total_requests
            else 0.0,
            "worker_routed_requests": routed,
            "projected_savings": round(projected_savings, 6),
            "active_workers": len(self.list_workers()),
            "recommendations": len(self._recommendations),
        }
