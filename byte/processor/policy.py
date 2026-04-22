import time
from collections import Counter, defaultdict
from threading import Lock
from typing import Any


class GlobalPolicyStore:
    """Content-free shared learning for routing and workflow policy."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._routes = defaultdict(
            lambda: {
                "category": "",
                "events": Counter(),
                "context_savings_chars": 0,
                "latency_ms_total": 0.0,
                "latency_samples": 0,
                "updated_at": 0.0,
            }
        )

    def record(
        self,
        route_key: str,
        *,
        category: str = "",
        event: str,
        latency_ms: float = 0.0,
        context_savings_chars: int = 0,
    ) -> None:
        if not route_key or not event:
            return
        with self._lock:
            entry = self._routes[route_key]
            if category:
                entry["category"] = category
            entry["events"][event] += 1
            if latency_ms > 0:
                entry["latency_ms_total"] += float(latency_ms)
                entry["latency_samples"] += 1
            if context_savings_chars > 0:
                entry["context_savings_chars"] += int(context_savings_chars)
            entry["updated_at"] = time.time()

    def hint(self, route_key: str) -> dict[str, Any]:
        with self._lock:
            entry = dict(self._routes.get(route_key, {}))
            events = Counter(entry.get("events", {}))
            latency_samples = int(entry.get("latency_samples", 0) or 0)
            avg_latency = (
                float(entry.get("latency_ms_total", 0.0) or 0.0) / latency_samples
                if latency_samples
                else 0.0
            )
        return {
            "route_key": route_key,
            "category": entry.get("category", ""),
            "events": dict(events),
            "avg_latency_ms": round(avg_latency, 2),
            "context_savings_chars": int(entry.get("context_savings_chars", 0) or 0),
            "clarify_first": events.get("clarify", 0) >= 1,
            "prefer_tool_context": events.get("tool_first", 0) >= 1,
            "prefer_verified_patch": events.get("verified_patch_reuse", 0) >= 1,
            "prefer_expensive": events.get("cheap_failure", 0) > events.get("cheap_success", 0),
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            top_routes = sorted(
                (
                    {
                        "route_key": route_key,
                        "category": entry.get("category", ""),
                        "events": dict(entry.get("events", {})),
                        "context_savings_chars": int(entry.get("context_savings_chars", 0) or 0),
                    }
                    for route_key, entry in self._routes.items()
                ),
                key=lambda item: sum(item["events"].values()),
                reverse=True,
            )[:20]
        return {
            "total_routes": len(self._routes),
            "top_routes": top_routes,
        }

    def clear(self) -> None:
        with self._lock:
            self._routes.clear()


_GLOBAL_POLICY = GlobalPolicyStore()


def record_policy_event(
    route_key: str,
    *,
    category: str = "",
    event: str,
    latency_ms: float = 0.0,
    context_savings_chars: int = 0,
) -> None:
    _GLOBAL_POLICY.record(
        route_key,
        category=category,
        event=event,
        latency_ms=latency_ms,
        context_savings_chars=context_savings_chars,
    )


def policy_hint(route_key: str) -> dict[str, Any]:
    return _GLOBAL_POLICY.hint(route_key)


def policy_stats() -> dict[str, Any]:
    return _GLOBAL_POLICY.stats()


def clear_global_policy() -> None:
    _GLOBAL_POLICY.clear()
