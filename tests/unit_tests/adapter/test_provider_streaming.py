from byte._backends import bedrock, cohere
from byte.utils.response import get_stream_message_from_openai_answer


class _StreamingHTTPResponse:
    def __init__(self, lines) -> None:
        self._lines = list(lines)
        self.closed = False

    def iter_lines(self, decode_unicode=True) -> object:
        del decode_unicode
        for line in self._lines:
            yield line

    def close(self) -> None:
        self.closed = True


def test_cohere_streaming_returns_openai_compatible_chunks(monkeypatch) -> None:
    response = _StreamingHTTPResponse(
        [
            "event: message-start",
            'data: {"type":"message-start"}',
            "",
            "event: content-delta",
            'data: {"type":"content-delta","delta":{"message":{"content":{"text":"Byte "}}}}',
            "",
            "event: content-delta",
            'data: {"type":"content-delta","delta":{"message":{"content":{"text":"stream"}}}}',
            "",
            "event: message-end",
            'data: {"type":"message-end","finish_reason":"COMPLETE"}',
            "",
        ]
    )

    monkeypatch.setattr(
        cohere,
        "request_json",
        lambda **kwargs: response,
    )

    chunks = list(
        cohere.ChatCompletion._llm_handler(
            model="command-r-plus",
            api_key="test-key",
            stream=True,
            messages=[{"role": "user", "content": "Say hello"}],
        )
    )

    assert "".join(get_stream_message_from_openai_answer(chunk) for chunk in chunks) == "Byte stream"
    assert chunks[-1]["choices"][0]["finish_reason"] == "COMPLETE"
    assert response.closed is True


def test_bedrock_streaming_returns_openai_compatible_chunks(monkeypatch) -> object:
    class _Client:
        def converse_stream(self, **kwargs) -> object:
            assert kwargs["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
            return {
                "stream": [
                    {"messageStart": {"role": "assistant"}},
                    {"contentBlockDelta": {"delta": {"text": "Byte "}}},
                    {"contentBlockDelta": {"delta": {"text": "stream"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                ]
            }

    monkeypatch.setattr(bedrock, "_get_client", lambda **kwargs: _Client())

    chunks = list(
        bedrock.ChatCompletion._llm_handler(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            region_name="us-east-1",
            stream=True,
            messages=[{"role": "user", "content": "Say hello"}],
        )
    )

    assert "".join(get_stream_message_from_openai_answer(chunk) for chunk in chunks) == "Byte stream"
    assert chunks[-1]["choices"][0]["finish_reason"] == "end_turn"
