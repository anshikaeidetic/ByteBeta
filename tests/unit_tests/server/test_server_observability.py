from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import byte_server.server as server
from byte import Cache, Config, __version__
from byte.embedding.string import to_embeddings as string_embedding
from byte.manager import manager_factory
from byte.processor.pre import last_content
from byte.similarity_evaluation import ExactMatchEvaluation
from byte_server.telemetry import TelemetrySettings

CHAT_ROUTE = f"{server.BYTE_GATEWAY_ROOT}/chat"


def _build_cache(tmp_path, config=None) -> object:
    cache_obj = Cache()
    cache_obj.init(
        pre_embedding_func=last_content,
        embedding_func=string_embedding,
        data_manager=manager_factory("map", data_dir=str(tmp_path)),
        similarity_evaluation=ExactMatchEvaluation(),
        config=config or Config(enable_token_counter=False),
    )
    return cache_obj


@contextmanager
def _client_with_cache(tmp_path, monkeypatch, config=None) -> object:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BYTE_BACKEND_API_KEY", raising=False)
    monkeypatch.delenv("BYTE_OTEL_ENABLED", raising=False)
    monkeypatch.delenv("BYTE_DATADOG_ENABLED", raising=False)
    server.gateway_cache = _build_cache(tmp_path, config=config)
    server.gateway_mode = "backend"
    server.gateway_routes = {}
    server.cache_dir = ""
    server.cache_file_key = ""
    server.telemetry_runtime = None
    server.request_guard_runtime.reset()
    try:
        with TestClient(server.app) as test_client:
            yield test_client
    finally:
        server.gateway_cache = None
        server.gateway_mode = "backend"
        server.gateway_routes = {}
        server.cache_dir = ""
        server.cache_file_key = ""
        server.telemetry_runtime = None
        server.request_guard_runtime.reset()


def test_telemetry_settings_enable_datadog_defaults() -> None:
    settings = TelemetrySettings.from_sources(
        service_name="byteai-cache-server",
        service_version=__version__,
        datadog_enabled=True,
        datadog_agent_host="dd-agent.monitoring",
        environment="production",
    )

    assert settings.enabled is True
    assert settings.endpoint == "http://dd-agent.monitoring:4318"
    assert settings.resource_payload()["service.name"] == "byteai-cache-server"
    assert settings.resource_payload()["deployment.environment.name"] == "production"
    assert settings.resource_payload()["env"] == "production"


def test_configure_server_telemetry_binds_base_and_gateway_cache(tmp_path, monkeypatch) -> object:
    class FakeRuntime:
        def __init__(self, settings, **kwargs) -> None:
            self.settings = settings
            self.kwargs = kwargs
            self.bound = []

        def start(self) -> object:
            return self

        def bind_cache(self, cache_obj, cache_role) -> None:
            self.bound.append((cache_obj, cache_role))

        def shutdown(self) -> None:
            pass

    server.gateway_cache = _build_cache(tmp_path)
    monkeypatch.setattr(server, "TelemetryRuntime", FakeRuntime)

    args = SimpleNamespace(
        otel_enabled=True,
        otel_endpoint="http://otel-collector:4318",
        otel_protocol="http/protobuf",
        otel_headers="",
        otel_insecure=False,
        otel_disable_traces=False,
        otel_disable_metrics=False,
        otel_export_interval_ms=7000,
        otel_service_namespace="byteai",
        otel_environment="staging",
        otel_resource_attributes="team=platform",
        datadog_enabled=True,
        datadog_agent_host="dd-agent",
        datadog_service="byteai-cache",
        datadog_env="staging",
        datadog_version=__version__,
    )

    server._configure_server_telemetry(args)

    assert server.telemetry_runtime.settings.enabled is True
    assert server.telemetry_runtime.settings.datadog_enabled is True
    assert server.telemetry_runtime.bound[0][1] == "base"
    assert server.telemetry_runtime.bound[1][1] == "gateway"


def test_chat_request_records_cache_hit_for_telemetry(tmp_path, monkeypatch) -> object:
    class DummyTelemetry:
        def __init__(self) -> None:
            self.finished = []
            self.chat_results = []

        def start_request_span(self, request) -> object:
            return {"path": request.url.path}

        def finish_request_span(self, span, request, *, status_code, duration_s, error=None) -> None:
            self.finished.append(
                {
                    "span": span,
                    "path": request.url.path,
                    "status_code": status_code,
                    "cache_hit": getattr(request.state, "byte_cache_hit", None),
                    "error": error,
                    "duration_s": duration_s,
                }
            )

        def record_chat_result(self, *, mode, cache_hit, model_name) -> None:
            self.chat_results.append((mode, cache_hit, model_name))

        def shutdown(self) -> None:
            pass

    dummy = DummyTelemetry()
    with _client_with_cache(tmp_path, monkeypatch) as client:
        server.telemetry_runtime = dummy
        with patch.object(
            server.default_backend.ChatCompletion, "create", return_value={"ok": True, "byte": True}
        ):
            response = client.post(
                CHAT_ROUTE,
                headers={"Authorization": "Bearer byo-key"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "say hi"}],
                },
            )

    assert response.status_code == 200
    assert dummy.chat_results == [("backend", True, "gpt-4o-mini")]
    assert dummy.finished[0]["cache_hit"] is True


def test_metrics_expose_h2o_runtime_counters(tmp_path, monkeypatch) -> None:
    with _client_with_cache(tmp_path, monkeypatch) as client, patch.object(
        server,
        "h2o_runtime_stats",
        return_value={
            "requests": 7,
            "applied": 5,
            "fallbacks": 2,
            "evictions": 41,
            "avg_retained_fraction": 0.375,
        },
    ):
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "byteai_h2o_requests_total 7" in response.text
    assert "byteai_h2o_applied_total 5" in response.text
    assert "byteai_h2o_fallbacks_total 2" in response.text
    assert "byteai_h2o_evictions_total 41" in response.text
    assert "byteai_h2o_avg_retained_fraction 0.375000" in response.text
