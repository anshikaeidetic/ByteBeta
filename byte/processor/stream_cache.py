"""Streaming response cache helper.

Provides a wrapper that buffers streamed LLM chunks, reconstructs the
full response, and caches it for future cache hits.  On a cache hit the
full response is replayed as a stream of synthetic chunks so callers
always get a consistent streaming interface.
"""

import time
from collections.abc import Callable, Generator
from typing import Any


class StreamCacheWrapper:
    """Wraps a streaming LLM response to capture and cache it.

    Usage:
        .. code-block:: python

            from byte.processor.stream_cache import StreamCacheWrapper

            wrapper = StreamCacheWrapper(
                stream=openai_stream,
                save_callback=lambda content: cache.put(key, content),
            )
            for chunk in wrapper:
                yield chunk  # forward to caller

    :param stream: the original LLM streaming iterator/generator
    :param save_callback: called with the fully reassembled content
        string after the stream is exhausted, so the result can be cached
    :param chunk_extractor: optional function to extract text from each
        chunk.  Defaults to OpenAI-style ``chunk.choices[0].delta.content``
    """

    def __init__(
        self,
        stream,
        save_callback: Callable[[str], None],
        chunk_extractor: Callable | None = None,
    ) -> None:
        self._stream = stream
        self._save_callback = save_callback
        self._chunk_extractor = chunk_extractor or self._default_extractor
        self._buffer: list = []
        self._finished = False

    def __iter__(self) -> Any:
        return self._iterate()

    def _iterate(self) -> Any:
        try:
            for chunk in self._stream:
                text = self._chunk_extractor(chunk)
                if text:
                    self._buffer.append(text)
                yield chunk
        finally:
            self._finished = True
            full_content = "".join(self._buffer)
            if full_content:
                try:
                    self._save_callback(full_content)
                except Exception:
                    pass  # don't break streaming on cache-save errors

    @staticmethod
    def _default_extractor(chunk) -> str:
        """Extract text from an OpenAI-style streaming chunk."""
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                return getattr(delta, "content", "") or ""
            elif isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                if choices:
                    return choices[0].get("delta", {}).get("content", "") or ""
        except (IndexError, KeyError, AttributeError):
            pass
        return ""


def replay_as_stream(
    content: str,
    model: str = "cache",
    chunk_size: int = 50,
) -> Generator:
    """Replay cached content as synthetic streaming chunks.

    This allows cache hits to be served through the same streaming
    interface as real LLM responses, keeping caller code identical
    regardless of cache hit/miss.

    :param content: the full cached response text
    :param model: model name to include in synthetic chunks
    :param chunk_size: approximate characters per synthetic chunk
    :return: generator yielding dicts matching OpenAI streaming format

    Example:
        .. code-block:: python

            from byte.processor.stream_cache import replay_as_stream

            for chunk in replay_as_stream("Hello world", model="gpt-4o"):
                print(chunk)
    """
    chunk_id = f"cache-{int(time.time())}"

    for i in range(0, len(content), chunk_size):
        piece = content[i : i + chunk_size]
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": piece},
                    "finish_reason": None,
                }
            ],
        }

    # Final chunk with finish_reason
    yield {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
