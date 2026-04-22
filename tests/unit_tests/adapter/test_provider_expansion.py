from unittest.mock import patch

from byte._backends import bedrock, cohere, deepseek, huggingface, mistral
from byte.adapter.api import provider_capabilities, provider_capability_matrix


class _MockResponse:
    def __init__(self, payload, status_code=200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> object:
        return self._payload


def test_provider_capability_matrix_includes_new_providers_and_runtime_flags() -> None:
    matrix = provider_capability_matrix()

    assert matrix["deepseek"]["chat_completion"]
    assert matrix["deepseek"]["coding_tasks"]
    assert matrix["deepseek"]["strict_cache_revalidation"]
    assert matrix["mistral"]["chat_completion"]
    assert matrix["cohere"]["chat_completion"]
    assert matrix["bedrock"]["chat_completion"]
    assert matrix["huggingface"]["chat_completion"]
    assert matrix["huggingface"]["text_completion"]
    assert matrix["huggingface"]["h2o_generation"]
    assert matrix["mistral"]["streaming_memory_recording"]
    assert matrix["cohere"]["distributed_vector_search"]
    assert matrix["bedrock"]["memory_export_sqlite_dump"]
    assert matrix["huggingface"]["provider"] == "huggingface"

    assert provider_capabilities("deep-seek")["provider"] == "deepseek"
    assert provider_capabilities("mistral")["provider"] == "mistral"
    assert provider_capabilities("hugging-face")["provider"] == "huggingface"


@patch("byte.adapter.mistral.request_json")
def test_mistral_adapter_returns_openai_compatible_payload(mock_request) -> None:
    mock_request.return_value = _MockResponse(
        {
            "id": "mistral-1",
            "model": "mistral-small-latest",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Byte via Mistral"},
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        }
    )

    response = mistral.ChatCompletion._llm_handler(
        model="mistral-small-latest",
        api_key="test-key",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert response["byte_provider"] == "mistral"
    assert response["choices"][0]["message"]["content"] == "Byte via Mistral"
    assert response["usage"]["total_tokens"] == 15


@patch("byte.adapter.cohere.request_json")
def test_cohere_adapter_returns_openai_compatible_payload(mock_request) -> None:
    mock_request.return_value = _MockResponse(
        {
            "id": "cohere-1",
            "message": {
                "content": [
                    {"type": "text", "text": "Byte via Cohere"},
                ]
            },
            "usage": {"tokens": {"input_tokens": 7, "output_tokens": 3}},
        }
    )

    response = cohere.ChatCompletion._llm_handler(
        model="command-r-plus",
        api_key="test-key",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert response["byte_provider"] == "cohere"
    assert response["choices"][0]["message"]["content"] == "Byte via Cohere"
    assert response["usage"]["total_tokens"] == 10


def test_bedrock_adapter_returns_openai_compatible_payload() -> object:
    class _Client:
        def converse(self, **kwargs) -> object:
            assert kwargs["modelId"] == "anthropic.claude-3-5-sonnet"
            return {
                "output": {
                    "message": {
                        "content": [{"text": "Byte via Bedrock"}],
                    }
                },
                "usage": {"inputTokens": 9, "outputTokens": 4},
                "stopReason": "end_turn",
                "ResponseMetadata": {"RequestId": "req-123"},
            }

    with patch("byte.adapter.bedrock._get_client", return_value=_Client()):
        response = bedrock.ChatCompletion._llm_handler(
            model="anthropic.claude-3-5-sonnet",
            region_name="us-east-1",
            messages=[{"role": "user", "content": "Say hello"}],
        )

    assert response["byte_provider"] == "bedrock"
    assert response["choices"][0]["message"]["content"] == "Byte via Bedrock"
    assert response["usage"]["total_tokens"] == 13


@patch("byte.adapter.deepseek.request_json")
def test_deepseek_adapter_returns_openai_compatible_payload(mock_request) -> None:
    mock_request.return_value = _MockResponse(
        {
            "id": "deepseek-1",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Byte via DeepSeek"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )

    response = deepseek.ChatCompletion._llm_handler(
        model="deepseek-chat",
        api_key="test-key",
        messages=[{"role": "user", "content": "Say hello"}],
    )

    assert response["byte_provider"] == "deepseek"
    assert response["choices"][0]["message"]["content"] == "Byte via DeepSeek"
    assert response["usage"]["total_tokens"] == 14


def test_huggingface_adapter_delegates_to_local_runtime() -> object:
    class _Runtime:
        def generate_chat(self, **kwargs) -> object:
            assert kwargs["byte_h2o_enabled"] is True
            return {
                "byte_provider": "huggingface",
                "byte_runtime": {"provider": "huggingface", "h2o_applied": True},
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Byte via Hugging Face"},
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }

    with patch("byte.adapter.huggingface.get_huggingface_runtime", return_value=_Runtime()):
        response = huggingface.ChatCompletion._llm_handler(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            byte_h2o_enabled=True,
        )

    assert response["byte_provider"] == "huggingface"
    assert response["choices"][0]["message"]["content"] == "Byte via Hugging Face"
    assert response["usage"]["total_tokens"] == 12
