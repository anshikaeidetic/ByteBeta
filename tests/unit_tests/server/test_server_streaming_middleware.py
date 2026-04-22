from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import byte_server.server as server
from byte import Cache, Config
from byte.embedding.string import to_embeddings as string_embedding
from byte.manager import manager_factory
from byte.processor.pre import last_content
from byte.similarity_evaluation import ExactMatchEvaluation
from byte_server import _server_security

OPENAI_CHAT_ROUTE = "/v1/chat/completions"


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


class _LeaseProbe:
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class _GuardProbe:
    def __init__(self, lease) -> None:
        self.lease = lease
        self.enter_calls = 0

    def enter(self, **kwargs) -> object:
        self.enter_calls += 1
        return self.lease

    def reset(self) -> None:
        return None


class _TelemetryProbe:
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
                "duration_s": duration_s,
                "error": error,
            }
        )

    def record_chat_result(self, *, mode, cache_hit, model_name) -> None:
        self.chat_results.append((mode, cache_hit, model_name))

    def shutdown(self) -> None:
        return None


def _streaming_backend(*args, **kwargs) -> object:
    assert kwargs["stream"] is True
    yield {"id": "chunk-1", "choices": [{"delta": {"content": "hi"}}]}
    yield "[DONE]"


@pytest.mark.asyncio
async def test_wrap_streaming_response_releases_lease_once_after_iterator_finishes() -> object:
    lease = _LeaseProbe()
    finalized = []

    async def body_iterator() -> object:
        yield b"chunk-1"
        yield b"chunk-2"

    response = StreamingResponse(body_iterator(), media_type="text/event-stream")
    _server_security._wrap_streaming_response(
        response,
        lease,
        finalize_request=lambda *, error=None: finalized.append(error),
    )

    assert lease.release_calls == 0
    assert finalized == []

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    assert chunks == [b"chunk-1", b"chunk-2"]
    assert lease.release_calls == 1
    assert finalized == [None]


def test_streaming_middleware_releases_lease_once_and_finishes_span_after_stream_route(
    tmp_path, monkeypatch
) -> None:
    lease = _LeaseProbe()
    guard = _GuardProbe(lease)
    telemetry = _TelemetryProbe()
    original_guard = server.request_guard_runtime
    original_telemetry = server.telemetry_runtime
    try:
        with _client_with_cache(tmp_path, monkeypatch) as client, patch.object(
            server.default_backend.ChatCompletion,
            "create",
            side_effect=_streaming_backend,
        ):
            server.request_guard_runtime = guard
            server.telemetry_runtime = telemetry
            with client.stream(
                "POST",
                OPENAI_CHAT_ROUTE,
                headers={"Authorization": "Bearer byo-key"},
                json={
                    "model": "gpt-4o-mini",
                    "stream": True,
                    "messages": [{"role": "user", "content": "say hi"}],
                },
            ) as response:
                assert response.status_code == 200
                body = list(response.iter_text())

        assert any("data: [DONE]" in chunk for chunk in body)
        assert lease.release_calls == 1
        assert len(telemetry.finished) == 1
        assert telemetry.finished[0]["status_code"] == 200
        assert telemetry.finished[0]["error"] is None
    finally:
        server.request_guard_runtime = original_guard
        server.telemetry_runtime = original_telemetry
