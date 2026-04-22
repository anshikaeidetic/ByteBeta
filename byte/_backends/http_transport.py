import asyncio
import json
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import Any

import requests

from byte.utils.error import ByteErrorCode, FeatureNotSupportedError, ProviderRequestError

_STREAM_END = object()


def request_json(
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 60.0,
    stream: bool = False,
    max_retries: int = 1,
    retry_backoff_s: float = 0.5,
) -> requests.Response:
    # Streaming responses must not be retried: the connection is already
    # partially consumed and rewinding would yield duplicate or garbled output.
    effective_retries = 0 if stream else max(0, int(max_retries))
    attempts = effective_retries + 1
    # Give streaming a longer wall-clock budget; non-streaming should fail fast.
    effective_timeout = (120.0 if stream else timeout)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=effective_timeout,
                stream=bool(stream),
            )
            raise_for_status(response, provider=provider)
            return response
        except (requests.RequestException, ProviderRequestError) as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            retryable = isinstance(exc, requests.RequestException) or bool(
                status_code in {408, 409, 425, 429, 500, 502, 503, 504}
            )
            if attempt >= attempts - 1 or not retryable:
                break
            time.sleep(float(retry_backoff_s) * (attempt + 1))
    if last_error is None:
        raise ProviderRequestError(
            provider=provider,
            message="request failed",
            code=ByteErrorCode.PROVIDER_TRANSPORT,
        )
    raise last_error


def raise_for_status(response: requests.Response, *, provider: str) -> None:
    if response.status_code < 400:
        return
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = response.text
    raise ProviderRequestError(
        provider=provider,
        message=f"API error {response.status_code}: {payload}",
        status_code=int(response.status_code),
        code=ByteErrorCode.PROVIDER_RESPONSE,
    )


def iter_sse_events(response: requests.Response) -> Iterator[tuple[str, str]]:
    event_name = ""
    data_lines: list[str] = []
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            line = str(raw_line or "")
            if not line:
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                    event_name = ""
                    data_lines = []
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            if stripped.startswith("event:"):
                event_name = stripped[6:].strip()
                continue
            if stripped.startswith("data:"):
                data_lines.append(stripped[5:].strip())
        if data_lines:
            yield event_name, "\n".join(data_lines)
    finally:
        response.close()


def iter_json_sse_events(response: requests.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    for event_name, payload_text in iter_sse_events(response):
        if payload_text == "[DONE]":
            break
        if not payload_text:
            continue
        yield event_name, json.loads(payload_text)


def iter_openai_sse_chunks(response: requests.Response, *, provider: str) -> Iterator[dict[str, Any]]:
    for _, payload in iter_json_sse_events(response):
        payload.setdefault("byte_provider", provider)
        yield payload


async def async_wrap_sync_iterator(iterator: Iterator[Any]) -> AsyncGenerator[Any, None]:
    while True:
        item = await asyncio.to_thread(_next_or_end, iterator)
        if item is _STREAM_END:
            break
        yield item


def require_streaming_supported(value: bool, *, provider: str) -> None:
    if not value:
        raise FeatureNotSupportedError(f"{provider} streaming is not available in this adapter.")


def stream_from_event_iterator(
    events: Iterator[Any],
    *,
    mapper: Callable[[Any], dict[str, Any] | None],
) -> Iterator[dict[str, Any]]:
    try:
        for event in events:
            chunk = mapper(event)
            if chunk is not None:
                yield chunk
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()


def _next_or_end(iterator: Iterator[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_END
