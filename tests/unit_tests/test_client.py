from __future__ import annotations

from unittest.mock import patch

import pytest

from byte.utils import import_httpx

import_httpx()
from byte import ByteClient
from byte.client import Client


class _SyncResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _AsyncResponse(_SyncResponse):
    pass


class _ModelResponse:
    def __init__(self, payload) -> None:
        self._payload = payload

    def model_dump(self) -> object:
        return dict(self._payload)


def _chat_payload(*, content: str, model: str = "openai/gpt-4o-mini") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1712345678,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        },
    }


def test_client_sync_put_and_get() -> None:
    client = Client()
    with patch("httpx.Client.post") as mock_response:
        mock_response.return_value = _SyncResponse({}, status_code=200)
        status_code = client.put("Hi", "Hi back")
        assert status_code == 200

    with patch("httpx.Client.post") as mock_response:
        mock_response.return_value = _SyncResponse({"answer": "Hi back"})
        answer = client.get("Hi")
        assert answer == "Hi back"


def test_client_chat_uses_public_openai_route() -> None:
    client = Client()
    payload = _chat_payload(content="Hi back", model="gpt-4o-mini")

    with patch("httpx.Client.post", return_value=_SyncResponse(payload)) as mock_post:
        response = client.chat(
            "gpt-4o-mini",
            [{"role": "user", "content": "Say hi"}],
            temperature=0.2,
        )

    assert response == payload
    assert mock_post.call_args.args[0] == "http://localhost:8000/v1/chat/completions"
    assert mock_post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": "Say hi"}
    ]
    assert mock_post.call_args.kwargs["json"]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_client_async_methods() -> None:
    client = Client()

    with patch("httpx.AsyncClient.post") as mock_response:
        mock_response.return_value = _AsyncResponse({}, status_code=201)
        assert await client.aput("Hi", "Hi back") == 201

    with patch("httpx.AsyncClient.post") as mock_response:
        mock_response.return_value = _AsyncResponse({"answer": "Hi back"})
        assert await client.aget("Hi") == "Hi back"

    payload = _chat_payload(content="Async hello", model="gpt-4o-mini")
    with patch("httpx.AsyncClient.post", return_value=_AsyncResponse(payload)) as mock_post:
        response = await client.achat(
            "gpt-4o-mini",
            [{"role": "user", "content": "Say hi"}],
        )

    assert response == payload
    assert mock_post.call_args.args[0] == "http://localhost:8000/v1/chat/completions"


@pytest.mark.asyncio
async def test_client_sync_methods_work_with_running_event_loop() -> None:
    client = Client()
    with patch("httpx.Client.post", return_value=_SyncResponse({"answer": "Hi back"})):
        assert client.get("Hi") == "Hi back"


def test_byte_client_safe_mode_initializes_safe_cache_and_normalizes_response() -> None:
    client = ByteClient(mode="safe", model="openai/gpt-4o-mini")
    response_payload = _ModelResponse(_chat_payload(content="hello"))

    with patch("byte.client.init_safe_semantic_cache") as mock_init, patch(
        "byte.client.ChatCompletion.create", return_value=response_payload
    ) as mock_create:
        response = client.chat("Say hello")

    assert response["choices"][0]["message"]["content"] == "hello"
    assert response["object"] == "chat.completion"
    assert response["model"] == "openai/gpt-4o-mini"
    assert mock_init.call_count == 1
    assert mock_create.call_args.kwargs["messages"] == [{"role": "user", "content": "Say hello"}]


def test_byte_client_exact_mode_initializes_exact_cache_and_adds_missing_defaults() -> None:
    client = ByteClient(mode="exact", model="openai/gpt-4o-mini")

    with patch("byte.client.init_exact_cache") as mock_init, patch(
        "byte.client.ChatCompletion.create",
        return_value={"choices": [{"message": {"content": "Use exact cache"}}]},
    ):
        response = client.chat([{"role": "user", "content": "Use exact cache"}])

    assert response["choices"][0]["message"]["content"] == "Use exact cache"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["object"] == "chat.completion"
    assert response["model"] == "openai/gpt-4o-mini"
    assert isinstance(response["created"], int)
    assert mock_init.call_count == 1


def test_byte_client_proxy_mode_uses_openai_route_and_matches_shape() -> None:
    client = ByteClient(
        mode="proxy",
        model="gpt-4o-mini",
        base_url="http://localhost:8000/",
        api_key="proxy-key",
    )
    payload = _chat_payload(content="Proxy this", model="gpt-4o-mini")

    with patch("httpx.Client.post", return_value=_SyncResponse(payload)) as mock_post:
        response = client.chat("Proxy this", temperature=0.2)

    assert response == payload
    assert mock_post.call_args.args[0] == "http://localhost:8000/v1/chat/completions"
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer proxy-key"
    assert mock_post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": "Proxy this"}
    ]
    assert mock_post.call_args.kwargs["json"]["temperature"] == 0.2


def test_byte_client_modes_return_consistent_shape() -> None:
    local_payload = _ModelResponse(_chat_payload(content="hello"))
    proxy_payload = _chat_payload(content="hello", model="gpt-4o-mini")

    safe_client = ByteClient(mode="safe", model="gpt-4o-mini")
    exact_client = ByteClient(mode="exact", model="gpt-4o-mini")
    proxy_client = ByteClient(mode="proxy", model="gpt-4o-mini")

    with patch("byte.client.init_safe_semantic_cache"), patch(
        "byte.client.init_exact_cache"
    ), patch("byte.client.ChatCompletion.create", return_value=local_payload), patch(
        "httpx.Client.post", return_value=_SyncResponse(proxy_payload)
    ):
        safe = safe_client.chat("hello")
        exact = exact_client.chat("hello")
        proxy = proxy_client.chat("hello")

    for response in (safe, exact, proxy):
        assert sorted(response) == ["choices", "created", "id", "model", "object", "usage"]
        assert response["choices"][0]["message"]["content"] == "hello"
        assert response["choices"][0]["message"]["role"] == "assistant"
