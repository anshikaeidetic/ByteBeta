from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from typing import Any

from byte.utils.log import byte_log

try:  # pragma: no cover - optional dependency import path
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
except ImportError:  # pragma: no cover - exercised through config guard
    metrics = None
    trace = None
    Status = None
    StatusCode = None


class OpenTelemetryConfigError(RuntimeError):
    """Raised when OpenTelemetry is requested without the optional dependency."""


# ─── Lightweight process-wide research counters ──────────────────────────
# These run independently of OpenTelemetry so they always provide data to the
# server's Prometheus payload, even when the observability extras are not
# installed. Subsystems (pipeline, intent filter, vcache) call the bump_*
# helpers; the server exposes the snapshot via `/metrics` and `/stats`.

_RESEARCH_LOCK = threading.Lock()
_RESEARCH_STATS: dict[str, float] = {
    "dual_threshold_reference_hits": 0,
    "intent_context_tokens_saved":   0,
    "llm_equivalence_calls":         0,
    "vcache_threshold_updates":      0,
    "recomp_augmentation_hits":      0,
    "prompt_distillation_calls":     0,
    "prompt_distillation_tokens_saved": 0,
    # RouteLLM (arXiv 2406.18665)
    "route_llm_total":               0,
    "route_llm_cheap":               0,
    "route_llm_strong":              0,
    "route_llm_skipped_cross_provider": 0,
    # Cost-Aware Eviction (arXiv 2508.07675)
    "eviction_cost_aware_evictions": 0,
    "eviction_cost_aware_savings":   0,
    # LSH Prefilter (arXiv 2503.05530)
    "lsh_prefilter_lookups":         0,
    "lsh_prefilter_tier0_hits":      0,
    "lsh_prefilter_skipped_searches": 0,
}


def bump_research_counter(name: str, delta: float = 1) -> None:
    """Increment a process-wide research counter by `delta` (default 1)."""
    with _RESEARCH_LOCK:
        _RESEARCH_STATS[name] = _RESEARCH_STATS.get(name, 0) + delta


def research_metrics_snapshot() -> dict[str, float]:
    """Return a thread-safe snapshot of all research counters."""
    with _RESEARCH_LOCK:
        return dict(_RESEARCH_STATS)


def reset_research_counters() -> None:
    """Zero every research counter (useful for tests)."""
    with _RESEARCH_LOCK:
        for key in _RESEARCH_STATS:
            _RESEARCH_STATS[key] = 0


def _normalize_attribute_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        normalized = [_normalize_attribute_value(item) for item in value]
        normalized = [item for item in normalized if item is not None]
        return normalized or None
    return str(value)


def normalize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        normalized = _normalize_attribute_value(value)
        if normalized is not None:
            result[str(key)] = normalized
    return result


def resolve_cache_owner(candidate: Any) -> Any:
    if candidate is None:
        return None
    owner_getter = getattr(candidate, "__byte_cache_owner__", None)
    if callable(owner_getter):
        try:
            resolved = owner_getter()
        except Exception:  # pylint: disable=W0703
            resolved = None
        if resolved is not None and resolved is not candidate:
            return resolve_cache_owner(resolved)
    report_owner = getattr(candidate, "owner_cache", None)
    if report_owner is not None:
        return resolve_cache_owner(report_owner)
    return candidate


def resolve_cache_from_reporter(report_func: Any) -> Any:
    return resolve_cache_owner(getattr(report_func, "__self__", None))


def resolve_runtime_cache(
    *, chat_cache: Any | None = None, report_func: Any | None = None, fallback_cache: Any | None = None
) -> Any:
    resolved = resolve_cache_owner(chat_cache)
    if resolved is None and report_func is not None:
        resolved = resolve_cache_from_reporter(report_func)
    if resolved is None:
        resolved = resolve_cache_owner(fallback_cache)
    return resolved


def get_log_time_func(
    *, chat_cache: Any | None = None, report_func: Any | None = None, fallback_cache: Any | None = None
) -> Any:
    cache_obj = resolve_runtime_cache(
        chat_cache=chat_cache,
        report_func=report_func,
        fallback_cache=fallback_cache,
    )
    cache_config = getattr(cache_obj, "config", None)
    return getattr(cache_config, "log_time_func", None)


class LibraryCacheObserver:
    def __init__(self, meter, base_attributes: dict[str, Any]) -> None:
        self._base_attributes = dict(base_attributes)
        self._operation_counter = meter.create_counter(
            "byteai.cache.operations",
            description="Total cache pipeline operations emitted by the ByteAI library runtime.",
        )
        self._operation_duration = meter.create_histogram(
            "byteai.cache.operation.duration",
            unit="s",
            description="Cache pipeline operation duration in seconds for the ByteAI library runtime.",
        )
        self._cache_hit_counter = meter.create_counter(
            "byteai.cache.hits",
            description="Total cache hits observed by the ByteAI library runtime.",
        )
        # New research-paper metrics ─────────────────────────────────────────
        # arXiv 2601.11687 — Dual-threshold reference lane
        self._dual_threshold_reference_hits = meter.create_counter(
            "byteai.dual_threshold.reference_hits",
            description="Stage 2 reference-lane activations (arXiv 2601.11687).",
        )
        # arXiv 2601.11687 — Intent-driven context filter
        self._intent_context_tokens_saved = meter.create_counter(
            "byteai.intent_context.tokens_saved",
            description="Cumulative tokens removed by IntentDrivenContextFilter (arXiv 2601.11687).",
        )
        # arXiv 2601.11687 — LLM equivalence checker
        self._llm_equivalence_calls = meter.create_counter(
            "byteai.llm_equivalence.calls",
            description="LLM equivalence checker invocations (arXiv 2601.11687).",
        )
        # arXiv 2502.03771 — vCache
        self._vcache_threshold_updates = meter.create_counter(
            "byteai.vcache.threshold_updates",
            description="Online sigmoid parameter updates received by vCache (arXiv 2502.03771).",
        )
        self._vcache_error_rate = meter.create_gauge(
            "byteai.vcache.error_rate",
            description="Rolling empirical error rate for vCache (arXiv 2502.03771).",
        )
        self._vcache_cold_embeddings = meter.create_gauge(
            "byteai.vcache.cold_embeddings",
            description="Count of embeddings below min_observations in vCache (arXiv 2502.03771).",
        )

    def record_operation(self, operation: str, delta_time: float) -> None:
        attributes = dict(self._base_attributes)
        attributes["byteai.operation"] = str(operation)
        self._operation_counter.add(1, attributes)
        self._operation_duration.record(float(delta_time or 0.0), attributes)

    def record_cache_hit(self) -> None:
        self._cache_hit_counter.add(1, self._base_attributes)

    def record_dual_threshold_reference_hit(self) -> None:
        self._dual_threshold_reference_hits.add(1, self._base_attributes)

    def record_intent_tokens_saved(self, tokens: int) -> None:
        self._intent_context_tokens_saved.add(int(tokens), self._base_attributes)

    def record_llm_equivalence_call(self) -> None:
        self._llm_equivalence_calls.add(1, self._base_attributes)

    def record_vcache_threshold_update(self) -> None:
        self._vcache_threshold_updates.add(1, self._base_attributes)

    def record_vcache_gauges(self, error_rate: float, cold_count: int) -> None:
        self._vcache_error_rate.set(float(error_rate), self._base_attributes)
        self._vcache_cold_embeddings.set(int(cold_count), self._base_attributes)


class LibraryTelemetryRuntime:
    def __init__(self, cache_obj: Any, config: Any) -> None:
        self.cache_obj = resolve_cache_owner(cache_obj)
        self.config = config
        self.enabled = bool(getattr(config, "telemetry_enabled", False))
        self._base_attributes = normalize_attributes(self._build_base_attributes())
        self._tracer = None
        self._meter = None
        self._observer = None
        self._bound_report = None
        if not self.enabled:
            return
        if trace is None or metrics is None:
            raise OpenTelemetryConfigError(
                "Library telemetry requires the optional OpenTelemetry dependencies. "
                "Install ByteAI Cache with the observability extra."
            )
        self._tracer = trace.get_tracer(str(config.telemetry_tracer_name))
        self._meter = metrics.get_meter(str(config.telemetry_meter_name))
        self._observer = LibraryCacheObserver(self._meter, self._base_attributes)

    @property
    def observer(self) -> Any:
        return self._observer

    def _build_base_attributes(self) -> dict[str, Any]:
        data_manager = getattr(self.cache_obj, "data_manager", None)
        similarity = getattr(self.cache_obj, "similarity_evaluation", None)
        payload: dict[str, Any] = {
            "byteai.component": "library",
            "byteai.cache.memory_scope": getattr(self.cache_obj, "memory_scope", "") or "",
            "byteai.cache.data_manager": type(data_manager).__name__
            if data_manager is not None
            else "",
            "byteai.cache.similarity_evaluation": type(similarity).__name__
            if similarity is not None
            else "",
            "byteai.cache.has_next_cache": bool(getattr(self.cache_obj, "next_cache", None)),
        }
        payload.update(getattr(self.config, "telemetry_attributes", None) or {})
        return payload

    def bind_report(self, report) -> None:
        if report is None or self._observer is None:
            return
        report.add_observer(self._observer)
        self._bound_report = report

    def shutdown(self) -> None:
        if self._bound_report is not None and self._observer is not None:
            try:
                self._bound_report.remove_observer(self._observer)
            except Exception:  # pylint: disable=W0703
                byte_log.debug("Failed removing library telemetry observer.", exc_info=True)
        self._bound_report = None

    @contextmanager
    def stage_span(self, stage_name: str, attributes: dict[str, Any] | None = None) -> Any:
        if not self.enabled or self._tracer is None:
            yield None
            return
        span_attributes = dict(self._base_attributes)
        span_attributes["byteai.stage"] = str(stage_name)
        span_attributes.update(normalize_attributes(attributes))
        with self._tracer.start_as_current_span(
            f"byteai.cache.{stage_name}",
            attributes=span_attributes,
        ) as span:
            try:
                yield span
            except Exception as exc:  # pylint: disable=W0703
                if span is not None:
                    span.record_exception(exc)
                    if Status is not None and StatusCode is not None:
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise


def build_library_telemetry(cache_obj: Any, config: Any) -> LibraryTelemetryRuntime:
    runtime = LibraryTelemetryRuntime(cache_obj, config)
    report = getattr(resolve_cache_owner(cache_obj), "report", None)
    runtime.bind_report(report)
    return runtime


def telemetry_stage_span(
    stage_name: str,
    *,
    chat_cache: Any | None = None,
    report_func: Any | None = None,
    fallback_cache: Any | None = None,
    attributes: dict[str, Any] | None = None,
) -> Any:
    cache_obj = resolve_runtime_cache(
        chat_cache=chat_cache,
        report_func=report_func,
        fallback_cache=fallback_cache,
    )
    runtime = getattr(cache_obj, "telemetry", None)
    if runtime is None:
        return nullcontext()
    return runtime.stage_span(stage_name, attributes=attributes)
