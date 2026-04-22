"""OpenTelemetry runtime wiring for Byte server requests and cache activity."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from byte.utils.error import ByteErrorCode
from byte.utils.log import log_byte_error

LOGGER = logging.getLogger(__name__)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _first_value(*values: str | None) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def _parse_kv_csv(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in str(raw or "").split(","):
        key, separator, value = item.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result


def _http_signal_endpoint(endpoint: str, signal: str) -> str:
    if not endpoint:
        return ""
    normalized = endpoint.rstrip("/")
    suffix = f"/v1/{signal}"
    if normalized.endswith(suffix):
        return normalized
    return f"{normalized}{suffix}"


@dataclass
class TelemetrySettings:
    enabled: bool = False
    protocol: str = "http/protobuf"
    endpoint: str = ""
    traces_endpoint: str = ""
    metrics_endpoint: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    insecure: bool = False
    traces_enabled: bool = True
    metrics_enabled: bool = True
    export_interval_ms: int = 5000
    service_name: str = "byteai-cache"
    service_namespace: str = "byteai"
    service_version: str = ""
    deployment_environment: str = ""
    resource_attributes: dict[str, str] = field(default_factory=dict)
    datadog_enabled: bool = False
    datadog_agent_host: str = "localhost"
    datadog_service: str = ""
    datadog_env: str = ""
    datadog_version: str = ""

    @classmethod
    def from_sources(
        cls,
        *,
        service_name: str,
        service_version: str,
        enabled: bool = False,
        endpoint: str | None = None,
        protocol: str | None = None,
        headers: str | None = None,
        insecure: bool = False,
        disable_traces: bool = False,
        disable_metrics: bool = False,
        export_interval_ms: int | None = None,
        service_namespace: str | None = None,
        environment: str | None = None,
        resource_attributes: str | None = None,
        datadog_enabled: bool = False,
        datadog_agent_host: str | None = None,
        datadog_service: str | None = None,
        datadog_env: str | None = None,
        datadog_version: str | None = None,
    ) -> TelemetrySettings:
        env_otel_enabled = _truthy(os.getenv("BYTE_OTEL_ENABLED")) or (
            (
                bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
                or _truthy(os.getenv("BYTE_OTEL_AUTO_ENABLE"))
            )
            and not _truthy(os.getenv("OTEL_SDK_DISABLED"))
        )
        env_datadog_enabled = _truthy(os.getenv("BYTE_DATADOG_ENABLED"))
        datadog_mode = bool(datadog_enabled or env_datadog_enabled)
        telemetry_enabled = bool(enabled or datadog_mode or env_otel_enabled)
        resolved_protocol = (
            _first_value(
                protocol, os.getenv("BYTE_OTEL_PROTOCOL"), os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL")
            )
            or "http/protobuf"
        )
        resolved_headers = {}
        resolved_headers.update(_parse_kv_csv(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")))
        resolved_headers.update(_parse_kv_csv(os.getenv("BYTE_OTEL_HEADERS", "")))
        resolved_headers.update(_parse_kv_csv(headers or ""))
        resolved_datadog_host = (
            _first_value(
                datadog_agent_host,
                os.getenv("BYTE_DATADOG_AGENT_HOST"),
                os.getenv("DD_AGENT_HOST"),
            )
            or "localhost"
        )
        resolved_endpoint = _first_value(
            endpoint,
            os.getenv("BYTE_OTEL_ENDPOINT"),
            os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        )
        if datadog_mode and not resolved_endpoint:
            resolved_endpoint = f"http://{resolved_datadog_host}:4318"

        combined_resource_attributes = {}
        combined_resource_attributes.update(
            _parse_kv_csv(os.getenv("OTEL_RESOURCE_ATTRIBUTES", ""))
        )
        combined_resource_attributes.update(
            _parse_kv_csv(os.getenv("BYTE_OTEL_RESOURCE_ATTRIBUTES", ""))
        )
        combined_resource_attributes.update(_parse_kv_csv(resource_attributes or ""))
        resolved_environment = _first_value(
            environment,
            os.getenv("BYTE_OTEL_ENVIRONMENT"),
            os.getenv("DD_ENV"),
        )
        resolved_service_name = _first_value(
            datadog_service,
            os.getenv("DD_SERVICE"),
            os.getenv("OTEL_SERVICE_NAME"),
            service_name,
        )
        resolved_service_version = _first_value(
            datadog_version,
            os.getenv("DD_VERSION"),
            service_version,
        )
        resolved_service_namespace = (
            _first_value(
                service_namespace,
                os.getenv("BYTE_OTEL_SERVICE_NAMESPACE"),
            )
            or "byteai"
        )

        return cls(
            enabled=telemetry_enabled,
            protocol=resolved_protocol,
            endpoint=resolved_endpoint or "http://localhost:4318",
            traces_endpoint="",
            metrics_endpoint="",
            headers=resolved_headers,
            insecure=bool(insecure or _truthy(os.getenv("BYTE_OTEL_INSECURE"))),
            traces_enabled=not bool(
                disable_traces or _truthy(os.getenv("BYTE_OTEL_DISABLE_TRACES"))
            ),
            metrics_enabled=not bool(
                disable_metrics or _truthy(os.getenv("BYTE_OTEL_DISABLE_METRICS"))
            ),
            export_interval_ms=int(
                export_interval_ms
                or os.getenv("BYTE_OTEL_EXPORT_INTERVAL_MS")
                or os.getenv("OTEL_METRIC_EXPORT_INTERVAL")
                or 5000
            ),
            service_name=resolved_service_name,
            service_namespace=resolved_service_namespace,
            service_version=resolved_service_version,
            deployment_environment=resolved_environment,
            resource_attributes=combined_resource_attributes,
            datadog_enabled=datadog_mode,
            datadog_agent_host=resolved_datadog_host,
            datadog_service=_first_value(
                datadog_service, os.getenv("DD_SERVICE"), resolved_service_name
            ),
            datadog_env=_first_value(datadog_env, os.getenv("DD_ENV"), resolved_environment),
            datadog_version=_first_value(
                datadog_version, os.getenv("DD_VERSION"), resolved_service_version
            ),
        )

    def span_exporter_endpoint(self) -> str:
        if self.protocol == "grpc":
            return self.endpoint
        return self.traces_endpoint or _http_signal_endpoint(self.endpoint, "traces")

    def metric_exporter_endpoint(self) -> str:
        if self.protocol == "grpc":
            return self.endpoint
        return self.metrics_endpoint or _http_signal_endpoint(self.endpoint, "metrics")

    def resource_payload(self) -> dict[str, str]:
        payload = dict(self.resource_attributes)
        payload.setdefault("service.name", self.service_name)
        payload.setdefault("service.namespace", self.service_namespace)
        if self.service_version:
            payload.setdefault("service.version", self.service_version)
        if self.deployment_environment:
            payload.setdefault("deployment.environment", self.deployment_environment)
            payload.setdefault("deployment.environment.name", self.deployment_environment)
        if self.datadog_enabled and self.datadog_env:
            payload.setdefault("env", self.datadog_env)
        return payload


class OpenTelemetryConfigError(RuntimeError):
    """Raised when observability is enabled without the required optional deps."""


class CacheReportObserver:
    def __init__(self, meter: Any, cache_role: str) -> None:
        self._base_attributes = {"cache.role": cache_role}
        self._operation_counter = meter.create_counter(
            "byteai.cache.operations",
            description="Total cache pipeline operations emitted by ByteAI Cache.",
        )
        self._operation_duration = meter.create_histogram(
            "byteai.cache.operation.duration",
            unit="s",
            description="Cache pipeline operation duration in seconds.",
        )
        self._cache_hit_counter = meter.create_counter(
            "byteai.cache.hits",
            description="Total cache hits observed by ByteAI Cache.",
        )

    def record_operation(self, operation: str, delta_time: float) -> None:
        attributes = dict(self._base_attributes)
        attributes["operation"] = operation
        self._operation_counter.add(1, attributes)
        self._operation_duration.record(float(delta_time or 0.0), attributes)

    def record_cache_hit(self) -> None:
        self._cache_hit_counter.add(1, self._base_attributes)


class TelemetryRuntime:
    def __init__(
        self,
        settings: TelemetrySettings,
        *,
        get_active_cache: Callable[[], Any],
        get_readiness: Callable[[], Any],
        get_router_summary: Callable[[], dict[str, Any]],
    ) -> None:
        self.settings = settings
        self._get_active_cache = get_active_cache
        self._get_readiness = get_readiness
        self._get_router_summary = get_router_summary
        self._tracer_provider: Any = None
        self._meter_provider: Any = None
        self._tracer: Any = None
        self._meter: Any = None
        self._request_counter: Any = None
        self._request_duration: Any = None
        self._chat_counter: Any = None
        self._report_bindings: list[tuple[Any, Any]] = []
        self._started = False

    @property
    def enabled(self) -> bool:
        return self._started and self.settings.enabled

    def start(self) -> TelemetryRuntime:
        if not self.settings.enabled or self._started:
            return self
        try:
            from opentelemetry.metrics import Observation
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:  # pragma: no cover
            raise OpenTelemetryConfigError(
                "OpenTelemetry support requires the optional server dependencies. "
                "Install the repository from source with .[server] before enabling OTEL."
            ) from exc

        trace_exporter = self._build_trace_exporter()
        metric_exporter = self._build_metric_exporter()
        resource = Resource.create(self.settings.resource_payload())

        if self.settings.traces_enabled and trace_exporter is not None:
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
            self._tracer_provider = tracer_provider
            self._tracer = tracer_provider.get_tracer("byteai.server")

        if self.settings.metrics_enabled and metric_exporter is not None:
            metric_reader = PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=self.settings.export_interval_ms,
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            self._meter_provider = meter_provider
            self._meter = meter_provider.get_meter("byteai.server")
            self._request_counter = self._meter.create_counter(
                "byteai.http.server.requests",
                description="Total HTTP requests handled by the ByteAI Cache server.",
            )
            self._request_duration = self._meter.create_histogram(
                "byteai.http.server.duration",
                unit="s",
                description="HTTP request duration for the ByteAI Cache server.",
            )
            self._chat_counter = self._meter.create_counter(
                "byteai.proxy.chat.requests",
                description="Total chat proxy requests served by the ByteAI Cache server.",
            )
            self._meter.create_observable_gauge(
                "byteai.server.ready",
                callbacks=[self._observe_readiness(Observation)],
                description="Server readiness state as a gauge.",
            )
            self._meter.create_observable_gauge(
                "byteai.cache.request_totals",
                callbacks=[self._observe_cache_totals(Observation)],
                description="Current cache request, hit, and miss totals.",
            )
            self._meter.create_observable_gauge(
                "byteai.router.health",
                callbacks=[self._observe_router_health(Observation)],
                description="Current router health score by target.",
            )

        self._started = True
        return self

    def bind_cache(self, cache_obj: Any, cache_role: str) -> None:
        if self._meter is None or cache_obj is None:
            return
        report = getattr(cache_obj, "report", None)
        if report is None or not hasattr(report, "add_observer"):
            return
        observer = CacheReportObserver(self._meter, cache_role=cache_role)
        report.add_observer(observer)
        self._report_bindings.append((report, observer))

    def shutdown(self) -> None:
        for report, observer in self._report_bindings:
            try:
                report.remove_observer(observer)
            except (AttributeError, TypeError, ValueError):  # pragma: no cover
                continue
        self._report_bindings.clear()
        if self._meter_provider is not None:
            try:
                self._meter_provider.shutdown()
            except Exception as exc:  # noqa: BLE001, RUF100 - telemetry shutdown boundary
                log_byte_error(
                    LOGGER,
                    logging.DEBUG,
                    "Failed shutting down OTel meter provider.",
                    error=exc,
                    code=ByteErrorCode.TELEMETRY_SHUTDOWN,
                    boundary="telemetry.shutdown",
                    stage="meter_provider",
                    exc_info=True,
                )
        if self._tracer_provider is not None:
            try:
                self._tracer_provider.shutdown()
            except Exception as exc:  # noqa: BLE001, RUF100 - telemetry shutdown boundary
                log_byte_error(
                    LOGGER,
                    logging.DEBUG,
                    "Failed shutting down OTel tracer provider.",
                    error=exc,
                    code=ByteErrorCode.TELEMETRY_SHUTDOWN,
                    boundary="telemetry.shutdown",
                    stage="tracer_provider",
                    exc_info=True,
                )
        self._meter_provider = None
        self._tracer_provider = None
        self._tracer = None
        self._meter = None
        self._request_counter = None
        self._request_duration = None
        self._chat_counter = None
        self._started = False

    def start_request_span(self, request: Any) -> Any:
        if self._tracer is None:
            return None
        try:
            from opentelemetry.trace import SpanKind
        except ImportError as exc:  # pragma: no cover
            raise OpenTelemetryConfigError(
                "OpenTelemetry tracing is enabled but opentelemetry-api is unavailable."
            ) from exc
        route = getattr(request.scope.get("route"), "path", request.url.path)
        attributes = {
            "http.request.method": request.method,
            "http.route": route,
            "url.path": request.url.path,
        }
        return self._tracer.start_span(
            f"{request.method} {route}", kind=SpanKind.SERVER, attributes=attributes
        )

    def finish_request_span(
        self,
        span: Any,
        request: Any,
        *,
        status_code: int,
        duration_s: float,
        error: BaseException | None = None,
    ) -> None:
        route = getattr(request.scope.get("route"), "path", request.url.path)
        attributes = {
            "http.request.method": request.method,
            "http.route": route,
            "http.response.status_code": int(status_code),
        }
        cache_hit = getattr(request.state, "byte_cache_hit", None)
        if cache_hit is not None:
            attributes["byteai.cache.hit"] = bool(cache_hit)
        if self._request_counter is not None:
            self._request_counter.add(1, attributes)
        if self._request_duration is not None:
            self._request_duration.record(float(duration_s or 0.0), attributes)
        if span is None:
            return
        try:
            from opentelemetry.trace import Status, StatusCode

            for key, value in attributes.items():
                span.set_attribute(key, value)
            if getattr(request.state, "byte_proxy_mode", ""):
                span.set_attribute("byteai.proxy.mode", request.state.byte_proxy_mode)
            if getattr(request.state, "byte_model_name", ""):
                span.set_attribute("byteai.model.name", request.state.byte_model_name)
            if error is not None:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
            elif int(status_code) >= 500:
                span.set_status(Status(StatusCode.ERROR))
            else:
                span.set_status(Status(StatusCode.OK))
        finally:
            span.end()

    def record_chat_result(self, *, mode: str, cache_hit: bool | None, model_name: str) -> None:
        if self._chat_counter is None:
            return
        attributes = {
            "proxy.mode": str(mode or "openai"),
            "model.name": str(model_name or ""),
            "cache.outcome": "unknown" if cache_hit is None else ("hit" if cache_hit else "miss"),
        }
        self._chat_counter.add(1, attributes)

    def _build_trace_exporter(self) -> Any:
        if not self.settings.traces_enabled:
            return None
        if self.settings.protocol == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter as GrpcOTLPSpanExporter,
                )
            except ImportError as exc:  # pragma: no cover
                raise OpenTelemetryConfigError(
                    "OTLP gRPC trace export requires opentelemetry-exporter-otlp-proto-grpc."
                ) from exc
            return GrpcOTLPSpanExporter(
                endpoint=self.settings.span_exporter_endpoint(),
                headers=self.settings.headers,
                insecure=self.settings.insecure,
            )
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HttpOTLPSpanExporter,
            )
        except ImportError as exc:  # pragma: no cover
            raise OpenTelemetryConfigError(
                "OTLP HTTP trace export requires opentelemetry-exporter-otlp-proto-http."
            ) from exc
        return HttpOTLPSpanExporter(
            endpoint=self.settings.span_exporter_endpoint(),
            headers=self.settings.headers,
        )

    def _build_metric_exporter(self) -> Any:
        if not self.settings.metrics_enabled:
            return None
        if self.settings.protocol == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter as GrpcOTLPMetricExporter,
                )
            except ImportError as exc:  # pragma: no cover
                raise OpenTelemetryConfigError(
                    "OTLP gRPC metric export requires opentelemetry-exporter-otlp-proto-grpc."
                ) from exc
            return GrpcOTLPMetricExporter(
                endpoint=self.settings.metric_exporter_endpoint(),
                headers=self.settings.headers,
                insecure=self.settings.insecure,
            )
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter as HttpOTLPMetricExporter,
            )
        except ImportError as exc:  # pragma: no cover
            raise OpenTelemetryConfigError(
                "OTLP HTTP metric export requires opentelemetry-exporter-otlp-proto-http."
            ) from exc
        return HttpOTLPMetricExporter(
            endpoint=self.settings.metric_exporter_endpoint(),
            headers=self.settings.headers,
        )

    def _observe_readiness(self, observation_type: Any) -> Callable[[Any], Any]:
        def callback(_options: Any) -> Any:
            ready, payload = self._get_readiness()
            yield observation_type(1 if ready else 0, {"state": "ready"})
            yield observation_type(
                1 if payload.get("proxy_enabled") else 0, {"state": "proxy_enabled"}
            )

        return callback

    def _observe_cache_totals(self, observation_type: Any) -> Callable[[Any], Any]:
        def callback(_options: Any) -> Any:
            active = self._get_active_cache()
            report = getattr(active, "report", None)
            total_requests = int(getattr(getattr(report, "op_pre", None), "count", 0) or 0)
            cache_hits = int(getattr(report, "hint_cache_count", 0) or 0)
            cache_misses = max(0, total_requests - cache_hits)
            yield observation_type(total_requests, {"metric": "requests"})
            yield observation_type(cache_hits, {"metric": "hits"})
            yield observation_type(cache_misses, {"metric": "misses"})

        return callback

    def _observe_router_health(self, observation_type: Any) -> Callable[[Any], Any]:
        def callback(_options: Any) -> Any:
            router = self._get_router_summary() or {}
            aliases = router.get("aliases", {})
            yield observation_type(len(aliases), {"target": "__aliases__"})
            for target, stats in router.get("targets", {}).items():
                yield observation_type(
                    float(stats.get("health_score", 0.0) or 0.0), {"target": str(target)}
                )

        return callback
