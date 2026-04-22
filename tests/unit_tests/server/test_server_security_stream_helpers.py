from __future__ import annotations

import pytest
from fastapi.responses import StreamingResponse

from byte_server import _server_security


class _LeaseProbe:
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


def test_wrap_streaming_response_finalizes_immediately_when_iterator_is_missing() -> None:
    lease = _LeaseProbe()
    finalized = []
    response = StreamingResponse(iter(()), media_type="text/event-stream")
    response.body_iterator = None

    wrapped = _server_security._wrap_streaming_response(
        response,
        lease,
        finalize_request=lambda *, error=None: finalized.append(error),
    )

    assert wrapped is response
    assert lease.release_calls == 1
    assert finalized == [None]


@pytest.mark.asyncio
async def test_wrap_streaming_response_reports_stream_errors_to_finalize() -> object:
    lease = _LeaseProbe()
    finalized = []

    async def body_iterator() -> object:
        yield b"chunk-1"
        raise ValueError("stream failed")

    response = StreamingResponse(body_iterator(), media_type="text/event-stream")
    _server_security._wrap_streaming_response(
        response,
        lease,
        finalize_request=lambda *, error=None: finalized.append(error),
    )

    chunks = []
    with pytest.raises(ValueError, match="stream failed"):
        async for chunk in response.body_iterator:
            chunks.append(chunk)

    assert chunks == [b"chunk-1"]
    assert lease.release_calls == 1
    assert len(finalized) == 1
    assert isinstance(finalized[0], ValueError)
