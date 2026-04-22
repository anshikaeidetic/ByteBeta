from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import byte_server.server as server
from byte import Cache, Config
from byte.embedding.string import to_embeddings as string_embedding
from byte.manager import manager_factory
from byte.processor.pre import last_content
from byte.similarity_evaluation import ExactMatchEvaluation
from byte_server._control_plane import ControlPlaneRuntime, RequestScope, WorkerSelection

CHAT_ROUTE = f"{server.BYTE_GATEWAY_ROOT}/chat"


def _build_cache(tmp_path: Path, config=None) -> object:
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
def _client_with_control_plane(tmp_path, monkeypatch) -> object:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BYTE_BACKEND_API_KEY", raising=False)
    server.gateway_cache = _build_cache(tmp_path)
    server.gateway_mode = "backend"
    server.gateway_routes = {}
    server.cache_dir = ""
    server.cache_file_key = ""
    server.telemetry_runtime = None
    server.request_guard_runtime.reset()
    server.control_plane_runtime = ControlPlaneRuntime(db_path=str(tmp_path / "control-plane.db"))
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
        server.control_plane_runtime = None
        server.request_guard_runtime.reset()


def test_control_plane_settings_and_inspect(tmp_path, monkeypatch) -> None:
    with _client_with_control_plane(tmp_path, monkeypatch) as client:
        response = client.post(
            "/byte/control/settings/features",
            json={
                "worker_urls": ["http://worker.example:8090"],
                "replay_enabled": True,
                "replay_sample_rate": 1.0,
            },
        )
        assert response.status_code == 200
        inspect_response = client.get("/byte/control/cache/inspect")

    assert inspect_response.status_code == 200
    payload = inspect_response.json()
    assert payload["runtime"]["feature_flags"]["worker_urls"] == ["http://worker.example:8090"]
    assert payload["runtime"]["feature_flags"]["replay_enabled"] is True
    assert payload["runtime"]["internal_auth_configured"] is False


def test_control_plane_sends_internal_token_to_private_services(tmp_path) -> object:
    runtime = ControlPlaneRuntime(
        db_path=str(tmp_path / "control-plane.db"),
        worker_urls=["http://worker.example:8090"],
        memory_service_url="http://memory.example:8091",
        internal_auth_token="secret-token",
    )
    headers_seen: list[dict[str, str]] = []

    class _Response:
        def __init__(self, payload) -> None:
            self._payload = payload
            self.content = b"{}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self._payload

    def _fake_get(url, headers=None, timeout=None) -> object:
        headers_seen.append(dict(headers or {}))
        return _Response(
            {
                "worker_id": "worker-1",
                "url": "http://worker.example:8090",
                "status": "ready",
                "health_score": 1.0,
                "queue_depth": 0,
                "free_vram_gb": 24.0,
                "model_inventory": ["gpt-4o-mini"],
            }
        )

    def _fake_post(url, headers=None, json=None, timeout=None) -> object:
        headers_seen.append(dict(headers or {}))
        if url.endswith("/resolve"):
            return _Response({"context": {}})
        return _Response({"choices": [{"message": {"content": "worker"}}]})

    with (
        patch("byte_server._control_plane.requests.get", side_effect=_fake_get),
        patch("byte_server._control_plane.requests.post", side_effect=_fake_post),
    ):
        runtime.refresh_workers()
        runtime.resolve_memory(
            scope=RequestScope(tenant="acme", session="s1", workflow="chat"),
            request_payload={"messages": [{"role": "user", "content": "hi"}]},
            provider_mode="hosted",
        )
        runtime.dispatch_to_worker(
            worker=WorkerSelection(
                worker_id="worker-1",
                url="http://worker.example:8090",
                source="health",
                model_name="gpt-4o-mini",
                score=1.0,
            ),
            scope=RequestScope(tenant="acme", session="s1", workflow="chat"),
            request_payload={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert headers_seen
    assert all(item["X-Byte-Internal-Token"] == "secret-token" for item in headers_seen)


def test_chat_route_records_control_plane_event(tmp_path, monkeypatch) -> None:
    with _client_with_control_plane(tmp_path, monkeypatch) as client:
        with patch.object(
            server.default_backend.ChatCompletion,
            "create",
            return_value={"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        ):
            response = client.post(
                CHAT_ROUTE,
                headers={"Authorization": "Bearer test-key", "X-Byte-Session": "s1"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "say hi"}],
                },
            )
        inspect_response = client.get("/byte/control/cache/inspect")

    assert response.status_code == 200
    events = inspect_response.json()["events"]
    assert events
    assert events[0]["event_type"] == "chat_completion"
    assert events[0]["workflow_id"]


def test_chat_route_uses_worker_selection_when_available(tmp_path, monkeypatch) -> object:
    with _client_with_control_plane(tmp_path, monkeypatch) as client:
        server.control_plane_runtime.update_feature_flags(
            {"worker_urls": ["http://worker.example:8090"]}
        )

        def _fake_refresh() -> object:
            server.control_plane_runtime.store.upsert_worker(
                {
                    "worker_id": "worker-1",
                    "url": "http://worker.example:8090",
                    "status": "ready",
                    "health_score": 0.9,
                    "queue_depth": 0,
                    "free_vram_gb": 48.0,
                    "model_inventory": ["gpt-4o-mini"],
                }
            )
            return server.control_plane_runtime.store.list_workers()

        monkeypatch.setattr(server.control_plane_runtime, "refresh_workers", _fake_refresh)
        monkeypatch.setattr(
            server.control_plane_runtime,
            "dispatch_to_worker",
            lambda **kwargs: {
                "choices": [{"message": {"role": "assistant", "content": "worker"}}],
                "byte_worker": {"worker_id": "worker-1"},
            },
        )

        response = client.post(
            CHAT_ROUTE,
            headers={"Authorization": "Bearer test-key", "X-Byte-Session": "s2"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "route me"}],
            },
        )
        roi_response = client.get("/byte/control/roi")

    assert response.status_code == 200
    assert roi_response.status_code == 200
    assert roi_response.json()["worker_routed_requests"] >= 1


def test_chat_streaming_preserves_gateway_path(tmp_path, monkeypatch) -> object:
    with _client_with_control_plane(tmp_path, monkeypatch) as client:

        def _streaming_handler(**kwargs) -> object:
            yield {"choices": [{"delta": {"content": "hel"}}]}
            yield {"choices": [{"delta": {"content": "lo"}}]}
            yield "[DONE]"

        monkeypatch.setattr(
            server.control_plane_runtime,
            "dispatch_to_worker",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("worker dispatch should not run for streaming")
            ),
        )

        with patch.object(
            server.default_backend.ChatCompletion,
            "create",
            side_effect=_streaming_handler,
        ), client.stream(
            "POST",
            CHAT_ROUTE,
            headers={"Authorization": "Bearer test-key", "X-Byte-Session": "stream-s1"},
            json={
                "model": "gpt-4o-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "stream please"}],
            },
        ) as response:
            payload = "".join(chunk.decode() for chunk in response.iter_raw())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"delta": {"content": "hel"}' in payload
    assert "data: [DONE]" in payload
