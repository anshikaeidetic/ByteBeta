"""Runtime orchestration helpers for the Byte control plane."""

from __future__ import annotations

import random
import threading
from collections.abc import Mapping
from typing import Any, cast

import requests

from byte import INTERNAL_AUTH_HEADER
from byte.processor.intent import extract_request_intent
from byte.trust.core import extract_contract
from byte_server._control_plane_routing import (
    WorkerSelection,
    _compression_profile,
    _model_family,
    _short_digest,
)
from byte_server._control_plane_scope import RequestScope, request_text, response_text
from byte_server._control_plane_store import ControlPlaneStore
from byte_server._control_plane_store_common import _now


class ControlPlaneRuntime:
    """Runtime wrapper around persisted control-plane state."""

    def __init__(
        self,
        *,
        db_path: str,
        worker_urls: list[str] | None = None,
        memory_service_url: str = "",
        internal_auth_token: str = "",
        replay_enabled: bool = False,
        replay_sample_rate: float = 0.05,
        request_timeout_s: float = 20.0,
    ) -> None:
        self.store = ControlPlaneStore(db_path)
        self._lock = threading.RLock()
        self._last_intents: dict[tuple[str, str], str] = {}
        self.request_timeout_s = max(1.0, float(request_timeout_s or 20.0))
        initial_settings = {
            "worker_urls": [
                str(item).strip().rstrip("/")
                for item in (worker_urls or [])
                if str(item).strip()
            ],
            "memory_service_url": str(memory_service_url or "").strip().rstrip("/"),
            "internal_auth_token": str(internal_auth_token or "").strip(),
            "replay_enabled": bool(replay_enabled),
            "replay_sample_rate": round(
                max(0.0, min(1.0, float(replay_sample_rate or 0.0))), 4
            ),
        }
        current = dict(self.store.get_setting("feature_flags", {}) or {})
        if not current:
            self.store.set_setting("feature_flags", initial_settings)
            self._feature_flags = initial_settings
        else:
            current.setdefault("worker_urls", initial_settings["worker_urls"])
            current.setdefault(
                "memory_service_url", initial_settings["memory_service_url"]
            )
            current.setdefault(
                "internal_auth_token", initial_settings["internal_auth_token"]
            )
            current.setdefault("replay_enabled", initial_settings["replay_enabled"])
            current.setdefault(
                "replay_sample_rate", initial_settings["replay_sample_rate"]
            )
            self.store.set_setting("feature_flags", current)
            self._feature_flags = current

    @property
    def feature_flags(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._feature_flags)

    def _worker_urls(self) -> list[str]:
        raw_urls = self.feature_flags.get("worker_urls", [])
        if not isinstance(raw_urls, list):
            return []
        return [str(item).strip().rstrip("/") for item in raw_urls if str(item).strip()]

    def _internal_auth_token(self) -> str:
        return str(self.feature_flags.get("internal_auth_token", "") or "").strip()

    def _internal_headers(self) -> dict[str, str]:
        token = self._internal_auth_token()
        if not token:
            return {}
        return {INTERNAL_AUTH_HEADER: token}

    def _feature_float(self, key: str, default: float = 0.0) -> float:
        value = self.feature_flags.get(key, default)
        return float(cast(float | int | str, value or default))

    def update_feature_flags(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            merged = dict(self._feature_flags)
            for key, value in dict(payload or {}).items():
                merged[str(key)] = value
            merged_worker_urls = merged.get("worker_urls", [])
            if not isinstance(merged_worker_urls, list):
                merged_worker_urls = []
            merged["worker_urls"] = [
                str(item).strip().rstrip("/")
                for item in merged_worker_urls
                if str(item).strip()
            ]
            if merged.get("memory_service_url") not in (None, ""):
                merged["memory_service_url"] = str(
                    merged.get("memory_service_url")
                ).strip().rstrip("/")
            if merged.get("internal_auth_token") not in (None, ""):
                merged["internal_auth_token"] = str(
                    merged.get("internal_auth_token")
                ).strip()
            merged["replay_enabled"] = bool(merged.get("replay_enabled", False))
            replay_sample_rate = merged.get("replay_sample_rate", 0.0)
            merged["replay_sample_rate"] = round(
                max(
                    0.0,
                    min(
                        1.0,
                        float(cast(float | int | str, replay_sample_rate or 0.0)),
                    ),
                ),
                4,
            )
            self.store.set_setting("feature_flags", merged)
            self._feature_flags = merged
            return dict(merged)

    def extract_scope(
        self,
        headers: Mapping[str, Any],
        payload: dict[str, Any] | None,
    ) -> RequestScope:
        """Derive tenant, session, and workflow scope from headers and payload."""

        payload = payload or {}
        tenant = str(
            headers.get("x-byte-tenant")
            or payload.get("byte_tenant")
            or payload.get("tenant")
            or payload.get("user")
            or "anonymous"
        ).strip()
        intent = extract_request_intent(payload)
        session = str(
            headers.get("x-byte-session")
            or payload.get("byte_session")
            or payload.get("session")
            or _short_digest(f"{tenant}:{intent.route_key}:{request_text(payload)}")
        ).strip()
        workflow = str(
            headers.get("x-byte-workflow")
            or payload.get("byte_workflow")
            or payload.get("workflow")
            or intent.route_key
            or "default"
        ).strip()
        source = (
            "headers"
            if headers.get("x-byte-session") or headers.get("x-byte-tenant")
            else "derived"
        )
        return RequestScope(
            tenant=tenant or "anonymous",
            session=session or "ephemeral",
            workflow=workflow or "default",
            source=source,
        )

    def refresh_workers(self) -> list[dict[str, Any]]:
        refreshed: list[dict[str, Any]] = []
        for url in self._worker_urls():
            try:
                response = requests.get(
                    f"{str(url).rstrip('/')}/internal/v1/runtime",
                    headers=self._internal_headers(),
                    timeout=self.request_timeout_s,
                )
                response.raise_for_status()
                payload = response.json() if response.content else {}
                payload["url"] = str(url).rstrip("/")
                refreshed.append(self.store.upsert_worker(payload))
            except (requests.RequestException, ValueError) as exc:  # pragma: no cover
                refreshed.append(
                    self.store.upsert_worker(
                        {
                            "worker_id": _short_digest(str(url)),
                            "url": str(url).rstrip("/"),
                            "status": "error",
                            "health_score": 0.0,
                            "queue_depth": 0,
                            "free_vram_gb": 0.0,
                            "model_inventory": [],
                            "metadata": {},
                            "last_error": str(exc),
                            "last_heartbeat": _now(),
                        }
                    )
                )
        return refreshed

    def maybe_select_worker(
        self,
        *,
        scope: RequestScope,
        request_payload: dict[str, Any],
    ) -> WorkerSelection | None:
        model_name = str(request_payload.get("model", "") or "").strip()
        if not model_name or not self._worker_urls():
            return None
        self.refresh_workers()
        return self.store.choose_worker(
            scope=scope,
            model_name=model_name,
            compression_profile=_compression_profile(request_payload),
        )

    def record_intent(self, *, scope: RequestScope, route_key: str) -> None:
        key = (scope.tenant, scope.session)
        route_key = str(route_key or "").strip()
        if not route_key:
            return
        with self._lock:
            previous = self._last_intents.get(key)
            self._last_intents[key] = route_key
        if previous and previous != route_key:
            self.store.record_intent_edge(
                tenant=scope.tenant,
                source_intent=previous,
                target_intent=route_key,
            )

    def _memory_service_url(self) -> str:
        return str(self.feature_flags.get("memory_service_url", "") or "").strip().rstrip("/")

    def resolve_memory(
        self,
        *,
        scope: RequestScope,
        request_payload: dict[str, Any],
        provider_mode: str,
    ) -> dict[str, Any]:
        base_url = self._memory_service_url()
        if not base_url:
            return {}
        try:
            response = requests.post(
                f"{base_url}/internal/v1/resolve",
                headers=self._internal_headers(),
                json={
                    "scope": scope.to_dict(),
                    "request": request_payload,
                    "provider_mode": provider_mode,
                },
                timeout=self.request_timeout_s,
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except (requests.RequestException, ValueError):  # pragma: no cover
            return {}

    def remember_memory(
        self,
        *,
        scope: RequestScope,
        request_payload: dict[str, Any],
        response_payload: Any,
        provider_mode: str,
        worker_id: str = "",
    ) -> None:
        base_url = self._memory_service_url()
        if not base_url:
            return
        try:
            requests.post(
                f"{base_url}/internal/v1/remember",
                headers=self._internal_headers(),
                json={
                    "scope": scope.to_dict(),
                    "request": request_payload,
                    "response": response_payload,
                    "provider_mode": provider_mode,
                    "worker_id": worker_id,
                },
                timeout=self.request_timeout_s,
            )
        except requests.RequestException:  # pragma: no cover
            return

    def dispatch_to_worker(
        self,
        *,
        worker: WorkerSelection,
        scope: RequestScope,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        response = requests.post(
            f"{worker.url.rstrip('/')}/internal/v1/generate",
            headers=self._internal_headers(),
            json={
                "scope": scope.to_dict(),
                "request": request_payload,
                "selection_source": worker.source,
            },
            timeout=self.request_timeout_s,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        self._record_worker_response(
            scope=scope,
            request_payload=request_payload,
            worker=worker,
            response_payload=payload,
        )
        return payload

    def _record_worker_response(
        self,
        *,
        scope: RequestScope,
        request_payload: dict[str, Any],
        worker: WorkerSelection,
        response_payload: dict[str, Any],
    ) -> None:
        model_name = str(request_payload.get("model", "") or "")
        prompt_digest = _short_digest(request_text(request_payload))
        self.store.record_worker_lease(
            tenant=scope.tenant,
            session_id=scope.session,
            model_family=_model_family(model_name),
            worker_id=worker.worker_id,
        )
        self.store.record_kv_residency(
            tenant=scope.tenant,
            session_id=scope.session,
            model_name=model_name,
            compression_profile=_compression_profile(request_payload),
            worker_id=worker.worker_id,
            prompt_digest=prompt_digest,
        )

    def maybe_schedule_replay(
        self,
        *,
        scope: RequestScope,
        request_payload: dict[str, Any],
        response_payload: Any,
        feature: str = "shadow",
    ) -> dict[str, Any] | None:
        if not bool(self.feature_flags.get("replay_enabled", False)):
            return None
        sample_rate = self._feature_float("replay_sample_rate", 0.0)
        if sample_rate <= 0 or random.random() > sample_rate:
            return None
        intent = extract_request_intent(request_payload)
        request_digest = _short_digest(request_text(request_payload))
        projected_savings = self._projected_savings(request_payload, response_payload)
        job = self.store.create_replay_job(
            scope=scope,
            route_key=intent.route_key,
            feature=feature,
            request_digest=request_digest,
            projected_savings=projected_savings,
        )
        outcome = self._evaluate_replay_candidate(
            request_payload,
            response_payload,
            projected_savings=projected_savings,
        )
        completed = self.store.complete_replay_job(job_id=job["id"], **outcome)
        self._update_recommendation_from_job(
            scope=scope,
            route_key=intent.route_key,
            feature=feature,
            outcome=completed,
        )
        return {"job": job, "outcome": completed}

    def _projected_savings(
        self, request_payload: dict[str, Any], response_payload: Any
    ) -> float:
        text = request_text(request_payload)
        base = max(len(text), len(response_text(response_payload))) / 4000.0
        return round(base * 0.015, 6)

    def _evaluate_replay_candidate(
        self,
        request_payload: dict[str, Any],
        response_payload: Any,
        *,
        projected_savings: float,
    ) -> dict[str, Any]:
        contract = extract_contract(request_payload)
        answer = response_text(response_payload)
        strict = bool(contract.get("strict", False))
        deterministic_score = 0.98 if strict else 0.78
        evidence_score = 0.92 if answer else 0.1
        error_rate = 0.0 if strict and answer else (0.02 if answer else 0.35)
        error_rate_upper = min(1.0, error_rate + 0.031)
        return {
            "deterministic_score": deterministic_score,
            "evidence_score": evidence_score,
            "error_rate": error_rate,
            "error_rate_upper": error_rate_upper,
            "judge_required": not strict,
            "projected_savings": projected_savings,
            "metadata": {
                "contract": contract,
                "answer_digest": _short_digest(answer),
                "request_digest": _short_digest(request_text(request_payload)),
            },
        }

    def _update_recommendation_from_job(
        self,
        *,
        scope: RequestScope,
        route_key: str,
        feature: str,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        relevant = [
            item
            for item in self.store.list_replays(limit=500)
            if item.get("tenant") == scope.tenant
            and item.get("route_key") == route_key
            and item.get("feature") == feature
        ]
        sample_count = len(relevant)
        avg_savings = (
            sum(
                float(item.get("projected_savings", 0.0) or 0.0)
                for item in relevant
            )
            / sample_count
            if sample_count
            else float(outcome.get("projected_savings", 0.0) or 0.0)
        )
        error_rate = (
            sum(float(item.get("error_rate", 0.0) or 0.0) for item in relevant)
            / sample_count
            if sample_count
            else float(outcome.get("error_rate", 0.0) or 0.0)
        )
        error_upper = min(1.0, error_rate + (1.96 / max(sample_count, 1) ** 0.5))
        suggested = 0.88 if error_upper <= 0.03 else 0.8 if error_upper <= 0.08 else 0.72
        auto_apply = bool(sample_count >= 1000 and avg_savings > 0 and error_upper <= 0.01)
        return self.store.upsert_recommendation(
            tenant=scope.tenant,
            route_key=route_key,
            feature=feature,
            suggested_threshold=suggested,
            sample_count=sample_count,
            projected_savings=avg_savings,
            error_rate=error_rate,
            error_rate_upper=error_upper,
            auto_apply_eligible=auto_apply,
            details={
                "latest_job_id": int(outcome.get("job_id", 0) or 0),
                "latest_error_rate_upper": float(
                    outcome.get("error_rate_upper", 0.0) or 0.0
                ),
            },
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "db_path": str(self.store.db_path),
            "feature_flags": self.feature_flags,
            "internal_auth_configured": bool(self._internal_auth_token()),
            "workers": self.store.list_workers(),
            "recommendations": self.store.list_recommendations(),
            "roi": self.store.roi_snapshot(),
        }


def apply_memory_resolution(
    request_payload: dict[str, Any], resolution: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge memory-service context fields into a routed request payload."""

    compiled = dict(request_payload or {})
    context = dict((resolution or {}).get("context", {}) or {})
    for field_name in (
        "byte_tool_result_context",
        "byte_repo_summary",
        "byte_retrieval_context",
    ):
        value = context.get(field_name)
        if value in (None, "", [], {}):
            continue
        existing = compiled.get(field_name)
        if existing in (None, "", [], {}):
            compiled[field_name] = value
        elif isinstance(existing, list):
            merged = list(existing)
            if isinstance(value, list):
                merged.extend(value)
            else:
                merged.append(value)
            compiled[field_name] = merged
        else:
            compiled[field_name] = f"{existing}\n{value}".strip()
    return compiled


def provider_mode_for_request(
    request_payload: dict[str, Any], worker_selected: bool = False
) -> str:
    """Classify whether a request should be treated as local or hosted."""

    model_name = str((request_payload or {}).get("model", "") or "").strip().lower()
    if worker_selected or model_name.startswith("huggingface/"):
        return "local"
    return "hosted"
