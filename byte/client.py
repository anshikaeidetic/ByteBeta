"""High-level and compatibility clients for the Byte gateway."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from byte.utils import import_httpx

import_httpx()

import httpx  # pylint: disable=C0413

from byte.adapter import ChatCompletion  # pylint: disable=C0413
from byte.adapter.api import init_exact_cache, init_safe_semantic_cache  # pylint: disable=C0413
from byte.config import Config  # pylint: disable=C0413
from byte.core import Cache  # pylint: disable=C0413

_CLIENT_HEADER = {"Content-Type": "application/json", "Accept": "application/json"}


def _coerce_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)
    if hasattr(payload, "dict"):
        dumped = payload.dict()
        if isinstance(dumped, dict):
            return dict(dumped)
    if hasattr(payload, "__dict__"):
        return dict(vars(payload))
    raise TypeError("Byte chat responses must be JSON-like mappings or model objects.")


def _normalize_chat_payload(payload: Any, *, model: str) -> dict[str, Any]:
    raw = _coerce_payload(payload)
    created = raw.get("created")
    try:
        created_at = int(created)
    except (TypeError, ValueError):
        created_at = int(time.time())

    normalized_choices: list[dict[str, Any]] = []
    for index, choice in enumerate(raw.get("choices", []) or []):
        choice_payload = dict(choice or {})
        message_payload = choice_payload.get("message")
        if isinstance(message_payload, dict):
            message = dict(message_payload)
        else:
            message = {"content": str(message_payload or "")}
        message.setdefault("role", "assistant")
        message.setdefault("content", "")
        choice_payload["message"] = message
        choice_payload.setdefault("index", index)
        choice_payload.setdefault("finish_reason", "stop")
        normalized_choices.append(choice_payload)

    if not normalized_choices:
        normalized_choices.append(
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        )

    usage = raw.get("usage")
    usage_payload = dict(usage) if isinstance(usage, dict) else {}

    return {
        "id": str(raw.get("id") or f"byte-{uuid4().hex}"),
        "object": str(raw.get("object") or "chat.completion"),
        "created": created_at,
        "model": str(raw.get("model") or model),
        "choices": normalized_choices,
        "usage": usage_payload,
    }


class Client:
    """Compatibility client for the standalone Byte server."""

    def __init__(self, uri: str = "http://localhost:8000") -> None:
        self._uri = str(uri or "http://localhost:8000").rstrip("/")

    def _post_json(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        with httpx.Client() as client:
            response = client.post(
                f"{self._uri}{path}",
                headers=_CLIENT_HEADER,
                json=payload,
            )
        response.raise_for_status()
        return response

    async def _apost_json(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._uri}{path}",
                headers=_CLIENT_HEADER,
                json=payload,
            )
        response.raise_for_status()
        return response

    def put(self, question: str, answer: str) -> int:
        response = self._post_json("/put", {"prompt": question, "answer": answer})
        return int(response.status_code)

    def get(self, question: str) -> Any:
        response = self._post_json("/get", {"prompt": question})
        return response.json().get("answer")

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        payload = {"model": model, "messages": list(messages)}
        payload.update(kwargs)
        response = self._post_json("/v1/chat/completions", payload)
        return response.json()

    async def aput(self, question: str, answer: str) -> int:
        response = await self._apost_json("/put", {"prompt": question, "answer": answer})
        return int(response.status_code)

    async def aget(self, question: str) -> Any:
        response = await self._apost_json("/get", {"prompt": question})
        return response.json().get("answer")

    async def achat(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        payload = {"model": model, "messages": list(messages)}
        payload.update(kwargs)
        response = await self._apost_json("/v1/chat/completions", payload)
        return response.json()


class ByteClient:
    """Preferred high-level Byte client with stable response shape across modes."""

    def __init__(
        self,
        *,
        mode: str = "safe",
        model: str | None = None,
        data_dir: str = "byte_data",
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        config: Config | None = None,
    ) -> None:
        normalized_mode = str(mode or "safe").strip().lower()
        if normalized_mode not in {"safe", "exact", "proxy"}:
            raise ValueError("mode must be one of: safe, exact, proxy")
        self._mode = normalized_mode
        self._model = str(model or "").strip() or None
        self._data_dir = str(data_dir or "byte_data")
        self._base_url = str(base_url or "http://localhost:8000").rstrip("/")
        self._api_key = str(api_key or "").strip() or None
        self._config = config or Config(enable_token_counter=False)
        self._cache_obj: Cache | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def chat(self, input: str | list[dict[str, Any]], model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        messages = self._normalize_messages(input)
        resolved_model = self._resolve_model(model)
        if self._mode == "proxy":
            timeout = kwargs.pop("request_timeout", 30.0)
            payload = {"model": resolved_model, "messages": messages}
            payload.update(kwargs)
            headers = dict(_CLIENT_HEADER)
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self._base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            return _normalize_chat_payload(response.json(), model=resolved_model)

        cache_obj = self._ensure_local_cache()
        response = ChatCompletion.create(
            model=resolved_model,
            messages=messages,
            cache_obj=cache_obj,
            **kwargs,
        )
        return _normalize_chat_payload(response, model=resolved_model)

    def _ensure_local_cache(self) -> Cache:
        if self._mode == "proxy":
            raise RuntimeError("proxy mode does not initialize a local cache")
        if self._cache_obj is not None:
            return self._cache_obj
        cache_obj = Cache()
        if self._mode == "safe":
            init_safe_semantic_cache(
                data_dir=self._data_dir,
                cache_obj=cache_obj,
                config=self._config,
            )
        else:
            init_exact_cache(
                data_dir=self._data_dir,
                cache_obj=cache_obj,
                config=self._config,
            )
        self._cache_obj = cache_obj
        return cache_obj

    def _resolve_model(self, model: str | None) -> str:
        resolved_model = str(model or self._model or "").strip()
        if not resolved_model:
            raise ValueError(
                "A model must be provided either when constructing ByteClient or when calling chat()."
            )
        return resolved_model

    @staticmethod
    def _normalize_messages(input: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        if isinstance(input, list):
            return [dict(message or {}) for message in input]
        raise TypeError("input must be either a string or an OpenAI-style messages list")


__all__ = ["ByteClient", "Client"]
