from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import byte_server.server as server
from byte import Cache, Config
from byte.embedding.string import to_embeddings as string_embedding
from byte.manager import manager_factory
from byte.processor.pre import last_content
from byte.similarity_evaluation import ExactMatchEvaluation

CHAT_ROUTE = f"{server.BYTE_GATEWAY_ROOT}/chat"
OPENAI_CHAT_ROUTE = "/v1/chat/completions"
IMAGE_ROUTE = f"{server.BYTE_GATEWAY_ROOT}/images"
MODERATION_ROUTE = f"{server.BYTE_GATEWAY_ROOT}/moderations"
SPEECH_ROUTE = f"{server.BYTE_GATEWAY_ROOT}/audio/speech"
TRANSCRIPTIONS_ROUTE = f"{server.BYTE_GATEWAY_ROOT}/audio/transcriptions"
TRANSLATIONS_ROUTE = f"{server.BYTE_GATEWAY_ROOT}/audio/translations"
MCP_TOOLS_ROUTE = f"{server.BYTE_MCP_ROOT}/tools"
MCP_REGISTER_ROUTE = f"{server.BYTE_MCP_ROOT}/register"
MCP_CALL_ROUTE = f"{server.BYTE_MCP_ROOT}/call"


class _HTTPResponse:
    def __init__(self, payload) -> None:
        self._payload = payload
        self.headers = {"content-type": "application/json"}
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


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


def _set_gateway_routes(routes) -> None:
    server.gateway_routes = dict(routes or {})
    server.clear_router_alias_registry()
    for alias_name, targets in server.gateway_routes.items():
        server.register_router_alias(alias_name, list(targets or []))


@contextmanager
def _client_with_cache(tmp_path, monkeypatch, config=None) -> object:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BYTE_BACKEND_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("BYTE_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("BYTE_AUDIT_LOG_PATH", raising=False)
    monkeypatch.delenv("BYTE_SECURITY_EXPORT_ROOT", raising=False)
    monkeypatch.delenv("BYTE_REQUIRE_HTTPS", raising=False)
    server.gateway_cache = _build_cache(tmp_path, config=config)
    server.gateway_mode = "backend"
    _set_gateway_routes({})
    server.cache_dir = ""
    server.cache_file_key = ""
    server.telemetry_runtime = None
    server.mcp_gateway = server.MCPGateway()
    server.request_guard_runtime.reset()
    try:
        with TestClient(server.app) as test_client:
            yield test_client
    finally:
        server.gateway_cache = None
        server.gateway_mode = "backend"
        _set_gateway_routes({})
        server.cache_dir = ""
        server.cache_file_key = ""
        server.telemetry_runtime = None
        server.mcp_gateway = server.MCPGateway()
        server.request_guard_runtime.reset()


@pytest.fixture()
def client(tmp_path, monkeypatch) -> object:
    with _client_with_cache(tmp_path, monkeypatch) as test_client:
        yield test_client


def test_chat_requires_client_or_server_key(client) -> None:
    response = client.post(
        CHAT_ROUTE,
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "say hi"}],
        },
    )

    assert response.status_code == 401
    assert "Missing Byte gateway credentials" in response.json()["detail"]


def test_chat_rejects_malformed_auth_header(client) -> None:
    response = client.post(
        CHAT_ROUTE,
        headers={"Authorization": "Bearer"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "say hi"}],
        },
    )

    assert response.status_code == 401
    assert "Invalid Authorization header" in response.json()["detail"]


def test_chat_rate_limit_enforced_under_security_mode(tmp_path, monkeypatch) -> None:
    config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_rate_limit_public_per_minute=1,
    )
    with _client_with_cache(tmp_path, monkeypatch, config=config) as client, patch.object(
        server.default_backend.ChatCompletion, "create", return_value={"ok": True}
    ):
        first = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )
        second = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert first.status_code == 200
    assert second.status_code == 429
    assert "rate limit" in second.json()["detail"].lower()


def test_chat_internal_errors_are_redacted(tmp_path, monkeypatch) -> None:
    config = Config(enable_token_counter=False, security_mode=True)
    with _client_with_cache(tmp_path, monkeypatch, config=config) as client, patch.object(
        server.default_backend.ChatCompletion,
        "create",
        side_effect=RuntimeError("secret token should not leak"),
    ):
        response = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Byte gateway chat request failed."


def test_chat_uses_byo_key_when_present(client) -> None:
    with patch.object(
        server.default_backend.ChatCompletion, "create", return_value={"ok": True}
    ) as mock_create:
        response = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json().get("ok") is True
    assert mock_create.call_args.kwargs["api_key"] == "byo-key"


def test_openai_chat_completions_alias_uses_gateway_flow(client) -> None:
    with patch.object(
        server.default_backend.ChatCompletion, "create", return_value={"ok": True}
    ) as mock_create:
        response = client.post(
            OPENAI_CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json().get("ok") is True
    assert mock_create.call_args.kwargs["api_key"] == "byo-key"


def test_chat_non_stream_offloads_to_threadpool(client) -> None:
    threadpool_result = {"ok": True}
    with patch.object(
        server, "run_in_threadpool", new=AsyncMock(return_value=threadpool_result)
    ) as mock_threadpool:
        response = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 200
    resp_body = response.json()
    for k, v in threadpool_result.items():
        assert resp_body.get(k) == v
    assert mock_threadpool.await_count == 1
    assert (
        mock_threadpool.await_args.args[0].__qualname__
        == server.default_backend.ChatCompletion.create.__qualname__
    )
    assert mock_threadpool.await_args.kwargs["api_key"] == "byo-key"


def test_chat_falls_back_to_server_key_env(client, monkeypatch) -> None:
    monkeypatch.setenv("BYTE_BACKEND_API_KEY", "server-key")

    with patch.object(
        server.default_backend.ChatCompletion, "create", return_value={"ok": True}
    ) as mock_create:
        response = client.post(
            CHAT_ROUTE,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json().get("ok") is True
    assert mock_create.call_args.kwargs["api_key"] == "server-key"


def test_stats_uses_gateway_cache(client) -> None:
    server.cache.report.op_pre.count = 99
    server.cache.report.hint_cache_count = 98
    server.gateway_cache.report.op_pre.count = 2
    server.gateway_cache.report.hint_cache_count = 1

    response = client.get("/stats")

    assert response.status_code == 200
    stats = response.json()
    assert stats["total_requests"] == 2
    assert stats["cache_hits"] == 1


def test_clear_targets_gateway_cache(client) -> None:
    server.gateway_cache.clear = MagicMock()

    response = client.post("/clear")

    assert response.status_code == 200
    server.gateway_cache.clear.assert_called_once_with()


@pytest.mark.requires_feature("transformers", "onnx", "sqlalchemy", "faiss")
def test_init_gateway_cache_hybrid_builds_chain(tmp_path) -> None:
    cache_obj = server._init_gateway_cache("hybrid", str(tmp_path))

    assert cache_obj.next_cache is not None
    assert cache_obj.next_cache.next_cache is not None


def test_init_gateway_cache_normalized_uses_normalized_prompt(tmp_path) -> None:
    cache_obj = server._init_gateway_cache("normalized", str(tmp_path))

    normalized_value = cache_obj.pre_embedding_func(
        {"messages": [{"role": "user", "content": "  Hello,   WORLD!!  "}]}
    )

    assert normalized_value == "hello world"


def test_memory_recent_uses_gateway_cache(client) -> None:
    server.gateway_cache.remember_interaction(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Translate to French: Hello"}],
        },
        answer="Bonjour",
        reasoning="translation",
    )

    response = client.get("/memory/recent")

    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["total_entries"] == 1
    assert body["entries"][0]["answer"] == "Bonjour"


def test_healthz_and_readyz_expose_deployment_probes(client) -> None:
    health = client.get("/healthz")
    ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["ready"] is True
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_metrics_exposes_prometheus_style_counters(client) -> None:
    server.gateway_cache.report.op_pre.count = 3
    server.gateway_cache.report.hint_cache_count = 2
    server.gateway_cache.report.op_llm.count = 1
    server.gateway_cache.report.op_llm.total_time = 0.25

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "byteai_cache_requests_total 3" in response.text
    assert "byteai_cache_hits_total 2" in response.text
    assert 'byteai_cache_operation_total{operation="llm"} 1' in response.text


def test_init_gateway_cache_accepts_routing_config(tmp_path) -> None:
    cache_obj = server._init_gateway_cache(
        "normalized",
        str(tmp_path),
        config=Config(
            enable_token_counter=False,
            model_routing=True,
            routing_cheap_model="cheap-model",
        ),
    )

    assert cache_obj.config.model_routing is True
    assert cache_obj.config.routing_cheap_model == "cheap-model"


def test_chat_uses_routed_gateway_for_byte_route_name(client) -> None:
    server.gateway_mode = "adaptive"
    _set_gateway_routes({"byte-chat": ["openai/gpt-4o-mini"]})

    with patch.object(
        server.byte_adapter.ChatCompletion,
        "create",
        return_value={"ok": True, "byte_router": {"selected_provider": "openai"}},
    ) as mock_create:
        response = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "byte-chat",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert mock_create.call_args.kwargs["api_key"] == "byo-key"
    assert mock_create.call_args.kwargs["byte_provider_keys"] == {}


def test_chat_allows_credentialless_local_huggingface_route(client) -> None:
    server.gateway_mode = "adaptive"
    _set_gateway_routes({"local-chat": ["huggingface/meta-llama/Llama-3.2-1B-Instruct"]})

    with patch.object(
        server.byte_adapter.ChatCompletion,
        "create",
        return_value={"ok": True, "byte_runtime": {"provider": "huggingface"}},
    ) as mock_create:
        response = client.post(
            CHAT_ROUTE,
            json={
                "model": "local-chat",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "api_key" not in mock_create.call_args.kwargs


def test_image_gateway_uses_byo_key(client) -> None:
    with patch.object(
        server.default_backend.Image,
        "create",
        return_value={"data": [{"url": "https://example.com/image.png"}]},
    ) as mock_create:
        response = client.post(
            IMAGE_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-image-1",
                "prompt": "Draw a cache diagram.",
            },
        )

    assert response.status_code == 200
    assert response.json()["data"][0]["url"] == "https://example.com/image.png"
    assert mock_create.call_args.kwargs["api_key"] == "byo-key"


def test_moderation_gateway_uses_routed_byte_route(client) -> None:
    server.gateway_mode = "adaptive"
    _set_gateway_routes({"moderation-default": ["openai/omni-moderation-latest"]})

    with patch.object(
        server.byte_adapter.Moderation, "create", return_value={"results": [{"flagged": False}]}
    ) as mock_create:
        response = client.post(
            MODERATION_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "moderation-default",
                "input": "hello world",
            },
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["flagged"] is False
    assert mock_create.call_args.kwargs["api_key"] == "byo-key"


def test_speech_gateway_returns_audio_bytes(client) -> None:
    with patch.object(
        server.default_backend.Speech, "create", return_value={"audio": b"abc", "format": "mp3"}
    ) as mock_create:
        response = client.post(
            SPEECH_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini-tts",
                "voice": "alloy",
                "input": "Byte speeds up repeated prompts.",
                "response_format": "mp3",
            },
        )

    assert response.status_code == 200
    assert response.content == b"abc"
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert mock_create.call_args.kwargs["api_key"] == "byo-key"


def test_audio_transcription_gateway_accepts_file_upload(client) -> None:
    with patch.object(
        server.default_backend.Audio, "transcribe", return_value={"text": "hello"}
    ) as mock_transcribe:
        response = client.post(
            TRANSCRIPTIONS_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            data={"model": "whisper-1"},
            files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "hello"
    assert mock_transcribe.call_args.kwargs["api_key"] == "byo-key"
    assert mock_transcribe.call_args.kwargs["file"]["name"] == "sample.wav"


def test_mcp_gateway_endpoints_register_and_cache_read_only_tools(client) -> None:
    register = client.post(
        MCP_REGISTER_ROUTE,
        json={
            "server_name": "docs",
            "tool_name": "search",
            "endpoint": "https://example.com/search",
            "cache_policy": "read_only",
        },
    )

    assert register.status_code == 200

    with patch(
        "byte.mcp_gateway.requests.request", return_value=_HTTPResponse({"items": [1]})
    ) as mock_request:
        first = client.post(
            MCP_CALL_ROUTE,
            json={
                "server_name": "docs",
                "tool_name": "search",
                "arguments": {"q": "byte"},
            },
        )
        second = client.post(
            MCP_CALL_ROUTE,
            json={
                "server_name": "docs",
                "tool_name": "search",
                "arguments": {"q": "byte"},
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert mock_request.call_count == 1
    assert second.json()["cached"] is True


def test_chat_uses_authorization_for_routed_gateway(client) -> None:
    server.gateway_mode = "adaptive"
    _set_gateway_routes({"flash-chat": ["gemini/gemini-2.0-flash"]})

    with patch.object(
        server.byte_adapter.ChatCompletion, "create", return_value={"ok": True}
    ) as mock_create:
        response = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer gem-key"},
            json={
                "model": "flash-chat",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert mock_create.call_args.kwargs["api_key"] == "gem-key"


def test_chat_uses_server_env_for_routed_gateway(client, monkeypatch) -> None:
    server.gateway_mode = "adaptive"
    _set_gateway_routes({"reasoning-chat": ["deepseek/deepseek-chat"]})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-key")

    with patch.object(
        server.byte_adapter.ChatCompletion, "create", return_value={"ok": True}
    ) as mock_create:
        response = client.post(
            CHAT_ROUTE,
            json={
                "model": "reasoning-chat",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert mock_create.call_args.kwargs["byte_provider_keys"]["deepseek"] == "deep-key"


def test_chat_rejects_client_supplied_api_base_override(client) -> None:
    response = client.post(
        CHAT_ROUTE,
        headers={"Authorization": "Bearer byo-key"},
        json={
            "model": "gpt-4o-mini",
            "api_base": "http://localhost:1234/v1",
            "messages": [{"role": "user", "content": "say hi"}],
        },
    )

    assert response.status_code == 403
    assert "host overrides are disabled" in response.json()["detail"].lower()


def test_https_enforcement_does_not_trust_forwarded_proto_by_default(tmp_path, monkeypatch) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_require_https=True,
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BYTE_BACKEND_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("BYTE_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("BYTE_AUDIT_LOG_PATH", raising=False)
    monkeypatch.delenv("BYTE_SECURITY_EXPORT_ROOT", raising=False)
    monkeypatch.delenv("BYTE_REQUIRE_HTTPS", raising=False)
    server.gateway_cache = _build_cache(tmp_path, config=secure_config)
    server.gateway_mode = "backend"
    _set_gateway_routes({})
    server.cache_dir = ""
    server.cache_file_key = ""
    server.telemetry_runtime = None
    try:
        with TestClient(server.app, raise_server_exceptions=False) as test_client:
            response = test_client.get(
                "/healthz",
                headers={"Host": "api.example.com", "X-Forwarded-Proto": "https"},
            )
    finally:
        server.gateway_cache = None
        server.gateway_mode = "backend"
        _set_gateway_routes({})
        server.cache_dir = ""
        server.cache_file_key = ""
        server.telemetry_runtime = None

    assert response.status_code == 403
    assert "https is required" in response.json()["detail"].lower()


def test_https_enforcement_can_trust_forwarded_proto_when_enabled(tmp_path, monkeypatch) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_require_https=True,
        security_trust_proxy_headers=True,
    )

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        response = test_client.get(
            "/healthz",
            headers={"Host": "api.example.com", "X-Forwarded-Proto": "https"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_rejects_non_object_json_payload(client) -> None:
    response = client.post(
        CHAT_ROUTE,
        headers={"Authorization": "Bearer byo-key"},
        json=["not-an-object"],
    )

    assert response.status_code == 400
    assert "json object" in response.json()["detail"].lower()


def test_chat_handles_non_string_message_content_without_cache_skip_crash(client) -> None:
    with patch.object(
        server.default_backend.ChatCompletion, "create", return_value={"ok": True}
    ) as mock_create:
        response = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "say hi"}],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json().get("ok") is True
    assert mock_create.call_count == 1


def test_chat_rejects_client_supplied_host_override(client) -> None:
    response = client.post(
        CHAT_ROUTE,
        headers={"Authorization": "Bearer byo-key"},
        json={
            "model": "gpt-4o-mini",
            "host": "http://localhost:11434",
            "messages": [{"role": "user", "content": "say hi"}],
        },
    )

    assert response.status_code == 403
    assert "host overrides are disabled" in response.json()["detail"].lower()


def test_admin_endpoint_requires_token_in_security_mode(tmp_path, monkeypatch) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_require_admin_auth=True,
        security_admin_token="admin-secret",
    )

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        response = test_client.post("/clear")

    assert response.status_code == 403
    assert "admin token" in response.json()["detail"].lower()


def test_admin_endpoint_accepts_token_and_sets_secure_headers(tmp_path, monkeypatch) -> None:
    audit_log = tmp_path / "audit.jsonl"
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_require_admin_auth=True,
        security_admin_token="admin-secret",
        security_audit_log_path=str(audit_log),
    )

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        response = test_client.get("/stats", headers={"X-Byte-Admin-Token": "admin-secret"})

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"
    assert audit_log.exists()
    assert '"action": "cache.stats"' in audit_log.read_text(encoding="utf-8")


def test_memory_export_artifact_rejects_paths_outside_export_root(tmp_path, monkeypatch) -> None:
    export_root = tmp_path / "exports"
    export_root.mkdir()
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_require_admin_auth=True,
        security_admin_token="admin-secret",
        security_export_root=str(export_root),
    )

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        response = test_client.post(
            "/memory/export_artifact",
            headers={"X-Byte-Admin-Token": "admin-secret"},
            json={"path": str(tmp_path / "outside.json"), "format": "json"},
        )

    assert response.status_code == 403
    assert "export root" in response.json()["detail"].lower()


def test_memory_export_artifact_requires_export_root_in_security_mode(tmp_path, monkeypatch) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_admin_token="admin-secret",
    )

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        response = test_client.post(
            "/memory/export_artifact",
            headers={"X-Byte-Admin-Token": "admin-secret"},
            json={"path": str(tmp_path / "memory.json"), "format": "json"},
        )

    assert response.status_code == 503
    assert "export root" in response.json()["detail"].lower()


def test_cache_file_endpoint_disabled_in_security_mode(tmp_path, monkeypatch) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_require_admin_auth=True,
        security_admin_token="admin-secret",
        security_disable_cache_file_endpoint=True,
    )
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "entry.txt").write_text("value", encoding="utf-8")

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        server.cache_dir = str(tmp_path / "cache")
        server.cache_file_key = "download-key"
        response = test_client.get(
            "/cache_file",
            params={"key": "download-key"},
            headers={"X-Byte-Admin-Token": "admin-secret"},
        )

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"].lower()


def test_chat_with_server_key_requires_admin_token_in_security_mode(tmp_path, monkeypatch) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_require_admin_auth=True,
        security_admin_token="admin-secret",
    )

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        monkeypatch.setenv("BYTE_BACKEND_API_KEY", "server-key")
        denied = test_client.post(
            CHAT_ROUTE,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )
        with patch.object(
            server.default_backend.ChatCompletion, "create", return_value={"ok": True}
        ) as mock_create:
            allowed = test_client.post(
                CHAT_ROUTE,
                headers={"X-Byte-Admin-Token": "admin-secret"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "say hi"}],
                },
            )

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["ok"] is True
    assert mock_create.call_args.kwargs["api_key"] == "server-key"


def test_chat_rejects_request_body_over_limit(tmp_path, monkeypatch) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_max_request_bytes=1024,
    )

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        response = test_client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "x" * 4096}],
            },
        )

    assert response.status_code == 413
    assert "size limit" in response.json()["detail"].lower()


def test_audio_transcription_rejects_upload_over_limit(tmp_path, monkeypatch) -> None:
    secure_config = Config(
        enable_token_counter=False,
        security_mode=True,
        security_max_request_bytes=1024,
        security_max_upload_bytes=1024,
    )

    with _client_with_cache(tmp_path, monkeypatch, config=secure_config) as test_client:
        response = test_client.post(
            TRANSCRIPTIONS_ROUTE,
            headers={"Authorization": "Bearer byo-key"},
            data={"model": "whisper-1"},
            files={"file": ("sample.wav", b"R" * 4096, "audio/wav")},
        )

    assert response.status_code == 413
    assert "size limit" in response.json()["detail"].lower()
