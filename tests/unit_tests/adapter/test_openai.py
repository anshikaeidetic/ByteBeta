"""OpenAI adapter tests covering chat, media, and cache compatibility flows."""

import base64
import concurrent.futures as cf
import os
import random
import threading
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from byte import Cache, cache
from byte._backends import openai
from byte.adapter.api import (
    init_exact_cache,
    init_hybrid_cache,
    init_normalized_cache,
    init_similar_cache,
)
from byte.config import Config
from byte.manager import get_data_manager
from byte.processor.pre import (
    get_file_bytes,
    get_openai_moderation_input,
    get_prompt,
    last_content,
    normalized_last_content,
)
from byte.similarity_evaluation import ExactMatchEvaluation
from byte.utils.error import CacheError
from byte.utils.response import (
    get_image_from_openai_b64,
    get_image_from_openai_url,
    get_image_from_path,
    get_message_from_openai_answer,
    get_stream_message_from_openai_answer,
    get_text_from_openai_answer,
)

try:
    from PIL import Image
except ModuleNotFoundError:
    from byte.utils.dependency_control import prompt_install

    prompt_install("pillow")
    from PIL import Image


@pytest.fixture(autouse=True)
def reset_global_state() -> object:
    """Reset the global byte cache and adapter mock state before each test."""
    cache.__init__()
    openai.ChatCompletion.llm = None
    openai.Image.llm = None
    openai.Moderation.llm = None
    openai.Audio.llm = None
    import os

    if os.path.exists("data_map.txt"):
        os.remove("data_map.txt")
    yield


def _make_mock_client(mock_response, method="chat.completions.create") -> object:
    """Helper to create a mock OpenAI client with the given response."""
    mock_client = MagicMock()
    parts = method.split(".")
    target = mock_client
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], MagicMock(return_value=mock_response))
    return mock_client


def _png_bytes(size=(16, 16), color=(12, 34, 56)) -> object:
    image = Image.new("RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("enable_token_counter", (True, False))
def test_normal_openai(enable_token_counter) -> None:
    cache.init(config=Config(enable_token_counter=enable_token_counter))
    question = "calculate 1+3"
    expect_answer = "the result is 4"

    datas = {
        "choices": [
            {
                "message": {"content": expect_answer, "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": 1677825464,
        "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion.chunk",
    }

    mock_client = _make_mock_client(datas)
    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )

        assert get_message_from_openai_answer(response) == expect_answer, response

    # Second call should hit cache
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    answer_text = get_message_from_openai_answer(response)
    assert answer_text == expect_answer, answer_text


def test_chat_completion_coalesces_concurrent_duplicate_requests(tmp_path) -> object:
    init_exact_cache(
        data_dir=str(tmp_path / "exact"),
        pre_func=last_content,
        config=Config(enable_token_counter=False),
    )
    expected = "the result is 4"
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    call_count = 0

    def fake_llm(*args, **kwargs) -> object:
        nonlocal call_count
        time.sleep(0.2)
        with lock:
            call_count += 1
        return {
            "choices": [
                {
                    "message": {"content": expected, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-coalesce",
            "model": kwargs.get("model", "gpt-4o-mini"),
            "object": "chat.completion",
        }

    openai.ChatCompletion.llm = fake_llm

    def one_call() -> object:
        barrier.wait()
        return openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "calculate 2+2"}],
        )

    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: one_call(), range(2)))

    answers = [get_message_from_openai_answer(response) for response in responses]
    assert answers == [expected, expected]
    assert call_count == 1
    assert sum(1 for response in responses if response.get("byte")) == 1


def test_chat_completion_coalesces_concurrent_normalized_variants(tmp_path) -> object:
    init_normalized_cache(
        data_dir=str(tmp_path / "normalized"),
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )
    expected = "POSITIVE"
    prompts = [
        (
            "Classify the sentiment.\n"
            "Labels: POSITIVE, NEGATIVE, NEUTRAL\n"
            'Review: "I absolutely loved this movie and would watch it again."\n'
            "Answer with exactly one label."
        ),
        (
            'Review: "I absolutely loved this movie and would watch it again."\n'
            "Labels: POSITIVE, NEGATIVE, NEUTRAL\n"
            "Classify the sentiment and answer with exactly one label."
        ),
    ]
    barrier = threading.Barrier(len(prompts))
    lock = threading.Lock()
    call_count = 0

    def fake_llm(*args, **kwargs) -> object:
        nonlocal call_count
        time.sleep(0.2)
        with lock:
            call_count += 1
        return {
            "choices": [
                {
                    "message": {"content": expected, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-normalized-coalesce",
            "model": kwargs.get("model", "gpt-4o-mini"),
            "object": "chat.completion",
        }

    openai.ChatCompletion.llm = fake_llm

    def one_call(prompt) -> object:
        barrier.wait()
        return openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        responses = list(pool.map(one_call, prompts))

    answers = [get_message_from_openai_answer(response) for response in responses]
    assert answers == [expected, expected]
    assert call_count == 1
    assert sum(1 for response in responses if response.get("byte")) == 1


@pytest.mark.requires_feature("onnx", "sqlalchemy", "faiss")
def test_chat_completion_hybrid_coalesces_concurrent_normalized_variants(tmp_path) -> object:
    init_hybrid_cache(
        data_dir=str(tmp_path / "hybrid"),
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )
    expected = "POSITIVE"
    prompts = [
        (
            "Classify the sentiment.\n"
            "Labels: POSITIVE, NEGATIVE, NEUTRAL\n"
            'Review: "I absolutely loved this movie and would watch it again."\n'
            "Answer with exactly one label."
        ),
        (
            'Review: "I absolutely loved this movie and would watch it again."\n'
            "Labels: POSITIVE, NEGATIVE, NEUTRAL\n"
            "Classify the sentiment and answer with exactly one label."
        ),
    ]
    barrier = threading.Barrier(len(prompts))
    lock = threading.Lock()
    call_count = 0

    def fake_llm(*args, **kwargs) -> object:
        nonlocal call_count
        time.sleep(0.2)
        with lock:
            call_count += 1
        return {
            "choices": [
                {
                    "message": {"content": expected, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-hybrid-normalized-coalesce",
            "model": kwargs.get("model", "gpt-4o-mini"),
            "object": "chat.completion",
        }

    openai.ChatCompletion.llm = fake_llm

    def one_call(prompt) -> object:
        barrier.wait()
        return openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        responses = list(pool.map(one_call, prompts))

    answers = [get_message_from_openai_answer(response) for response in responses]
    assert answers == [expected, expected]
    assert call_count == 1
    assert sum(1 for response in responses if response.get("byte")) == 1


def test_chat_completion_routes_structured_requests_to_cheap_model_and_records_memory(tmp_path) -> object:
    cache_obj = Cache()
    init_normalized_cache(
        data_dir=str(tmp_path / "routed"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            model_routing=True,
            routing_cheap_model="cheap-model",
            routing_expensive_model="expensive-model",
            routing_default_model="expensive-model",
        ),
    )
    prompt = (
        "Classify the sentiment.\n"
        "Labels: POSITIVE, NEGATIVE, NEUTRAL\n"
        'Review: "I absolutely loved this movie."\n'
        "Answer with exactly one label."
    )
    seen_models = []

    def fake_llm(*args, **kwargs) -> object:
        seen_models.append(kwargs["model"])
        return {
            "choices": [
                {
                    "message": {"content": "POSITIVE", "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-routed-cheap",
            "model": kwargs["model"],
            "object": "chat.completion",
            "byte_provider": "openai",
        }

    openai.ChatCompletion.llm = fake_llm

    response = openai.ChatCompletion.create(
        model="expensive-model",
        messages=[{"role": "user", "content": prompt}],
        cache_obj=cache_obj,
    )

    assert seen_models == ["cheap-model"]
    assert get_message_from_openai_answer(response) == "POSITIVE"
    interaction = cache_obj.recent_interactions(1)[0]
    assert interaction["model"] == "cheap-model"
    assert interaction["metadata"]["model_route"]["tier"] == "cheap"


def test_openai_security_mode_blocks_client_api_base_override() -> None:
    cache.init(config=Config(enable_token_counter=False, security_mode=True))
    openai.ChatCompletion.llm = lambda **kwargs: {
        "choices": [
            {
                "message": {"content": "ok", "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }
        ]
    }

    with pytest.raises(CacheError, match="host overrides are disabled"):
        openai.ChatCompletion.create(
            model="gpt-4o-mini",
            api_base="http://localhost:9999/v1",
            messages=[{"role": "user", "content": "hello"}],
        )


def test_chat_completion_routes_complex_requests_to_expensive_model(tmp_path) -> object:
    cache_obj = Cache()
    init_normalized_cache(
        data_dir=str(tmp_path / "complex-routed"),
        cache_obj=cache_obj,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(
            enable_token_counter=False,
            model_routing=True,
            routing_cheap_model="cheap-model",
            routing_expensive_model="expensive-model",
            routing_default_model="cheap-model",
            routing_long_prompt_chars=120,
        ),
    )
    seen_models = []

    def fake_llm(*args, **kwargs) -> object:
        seen_models.append(kwargs["model"])
        return {
            "choices": [
                {
                    "message": {"content": "Detailed plan", "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-routed-expensive",
            "model": kwargs["model"],
            "object": "chat.completion",
        }

    openai.ChatCompletion.llm = fake_llm

    openai.ChatCompletion.create(
        model="cheap-model",
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze this architecture, compare the tradeoffs, debug the bottlenecks, "
                    "and reason step by step about the safest migration strategy."
                ),
            }
        ],
        cache_obj=cache_obj,
    )

    assert seen_models == ["expensive-model"]


def test_cache_miss_coalesces_before_embedding_work(tmp_path) -> object:
    cache_obj = Cache()
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    llm_call_count = 0
    embedding_call_count = 0

    def slow_embedding(data, **kwargs) -> object:  # pylint: disable=unused-argument
        nonlocal embedding_call_count
        time.sleep(0.2)
        with lock:
            embedding_call_count += 1
        return data

    cache_obj.init(
        pre_embedding_func=last_content,
        embedding_func=slow_embedding,
        data_manager=get_data_manager(data_path=str(tmp_path / "embed_map.pkl")),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(enable_token_counter=False),
    )

    def fake_llm(*args, **kwargs) -> object:
        nonlocal llm_call_count
        time.sleep(0.2)
        with lock:
            llm_call_count += 1
        return {
            "choices": [
                {
                    "message": {"content": "the result is 9", "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-embed-coalesce",
            "model": kwargs.get("model", "gpt-4o-mini"),
            "object": "chat.completion",
        }

    openai.ChatCompletion.llm = fake_llm

    def one_call() -> object:
        barrier.wait()
        return openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "calculate 3+6"}],
            cache_obj=cache_obj,
        )

    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: one_call(), range(2)))

    answers = [get_message_from_openai_answer(response) for response in responses]
    assert answers == ["the result is 9", "the result is 9"]
    assert llm_call_count == 1
    assert embedding_call_count == 1


def test_stream_openai() -> None:
    cache.init()
    question = "calculate 1+1"
    expect_answer = "the result is 2"

    datas = [
        {
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None, "index": 0}],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [
                {
                    "delta": {"content": "the result"},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [{"delta": {"content": " is 2"}, "finish_reason": None, "index": 0}],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
        {
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "created": 1677825464,
            "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion.chunk",
        },
    ]

    mock_client = _make_mock_client(iter(datas))
    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            stream=True,
        )

        all_text = ""
        for res in response:
            all_text += get_stream_message_from_openai_answer(res)
        assert all_text == expect_answer, all_text

    # Cache hit â€” non-stream call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    answer_text = get_message_from_openai_answer(response)
    assert answer_text == expect_answer, answer_text


def test_completion() -> None:
    cache.init(pre_embedding_func=get_prompt)
    question = "what is your name?"
    expect_answer = "byte"

    datas = {
        "choices": [{"text": expect_answer, "finish_reason": None, "index": 0}],
        "created": 1677825464,
        "id": "cmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
        "model": "text-davinci-003",
        "object": "text_completion",
    }
    mock_client = MagicMock()
    mock_client.completions.create = MagicMock(return_value=datas)

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.Completion.create(model="text-davinci-003", prompt=question)
        answer_text = get_text_from_openai_answer(response)
        assert answer_text == expect_answer

    # Cache hit
    response = openai.Completion.create(model="text-davinci-003", prompt=question)
    answer_text = get_text_from_openai_answer(response)
    assert answer_text == expect_answer


def test_completion_error_wrapping() -> None:
    cache.init(pre_embedding_func=get_prompt)
    import openai as real_openai

    # Sync error wrapping
    mock_client = MagicMock()
    mock_client.completions.create = MagicMock(side_effect=real_openai.OpenAIError("test"))

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        with pytest.raises(real_openai.OpenAIError) as e:
            openai.Completion.create(model="text-davinci-003", prompt="boom")
        assert isinstance(e.value, CacheError)


def test_image_create() -> None:
    cache.init(pre_embedding_func=get_prompt)
    prompt1 = "test url"  # bytes
    test_url = "https://example.com/fake-image.png"
    test_response = {"created": 1677825464, "data": [{"url": test_url}]}
    prompt2 = "test base64"
    source_png = _png_bytes()
    mock_response = MagicMock()
    mock_response.content = source_png
    mock_response.raise_for_status.return_value = None
    with patch("byte.utils.response.requests.get", return_value=mock_response):
        img_bytes = base64.b64decode(get_image_from_openai_url(test_response))
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    img = Image.open(img_file)
    img = img.resize((256, 256))
    img = img.convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    expected_img_data = base64.b64encode(buffered.getvalue()).decode("ascii")

    ###### Return base64 ######
    mock_client = MagicMock()
    mock_client.images.generate = MagicMock(
        return_value={
            "created": 1677825464,
            "data": [{"b64_json": expected_img_data}],
        }
    )

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.Image.create(prompt=prompt1, size="256x256", response_format="b64_json")
        img_returned = get_image_from_openai_b64(response)
        assert img_returned == expected_img_data

    response = openai.Image.create(prompt=prompt1, size="256x256", response_format="b64_json")
    img_returned = get_image_from_openai_b64(response)
    assert img_returned == expected_img_data

    ###### Return url ######
    mock_client2 = MagicMock()
    mock_client2.images.generate = MagicMock(
        return_value={
            "created": 1677825464,
            "data": [{"url": test_url}],
        }
    )

    with patch("byte.adapter.openai._get_client", return_value=mock_client2), patch(
        "byte.utils.response.requests.get", return_value=mock_response
    ):
        response = openai.Image.create(prompt=prompt2, size="256x256", response_format="url")
        answer_url = response["data"][0]["url"]
        assert test_url == answer_url

    response = openai.Image.create(prompt=prompt2, size="256x256", response_format="url")
    img_returned = get_image_from_path(response).decode("ascii")
    assert img_returned == expected_img_data
    os.remove(response["data"][0]["url"])


def test_speech_create() -> None:
    cache.init(pre_embedding_func=get_openai_moderation_input)
    expected_audio = b"fake-mp3-audio"

    mock_client = MagicMock()
    mock_client.audio.speech.create = MagicMock(return_value=expected_audio)

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.Speech.create(
            model="gpt-4o-mini-tts",
            input="Hello from Byte.",
            voice="alloy",
            response_format="mp3",
        )
        assert response["audio"] == expected_audio
        assert response["format"] == "mp3"
        assert not response.get("byte", False)

    response = openai.Speech.create(
        model="gpt-4o-mini-tts",
        input="Hello from Byte.",
        voice="alloy",
        response_format="mp3",
    )
    assert response["audio"] == expected_audio
    assert response["format"] == "mp3"
    assert response["byte"]


def test_audio_transcribe_caches_file_payloads(tmp_path) -> None:
    cache.init(pre_embedding_func=get_file_bytes)
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake-audio-bytes")

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create = MagicMock(
        return_value={"text": "Byte speeds up repeated AI requests."}
    )

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        with open(audio_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                "gpt-4o-mini-transcribe",
                audio_file,
            )
        assert response["text"] == "Byte speeds up repeated AI requests."

    with open(audio_path, "rb") as audio_file:
        response = openai.Audio.transcribe(
            "gpt-4o-mini-transcribe",
            audio_file,
        )
    assert response["text"] == "Byte speeds up repeated AI requests."
    assert response["byte"]


def test_image_create_b64_json_does_not_forward_response_format() -> None:
    cache.init(pre_embedding_func=get_prompt)
    expected_img_data = base64.b64encode(b"fake-image-bytes").decode("ascii")

    mock_client = MagicMock()
    mock_client.images.generate = MagicMock(
        return_value={
            "created": 1677825464,
            "data": [{"b64_json": expected_img_data}],
        }
    )

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.Image.create(
            model="gpt-image-1-mini",
            prompt="blue whale icon",
            size="256x256",
            response_format="b64_json",
        )

    _, call_kwargs = mock_client.images.generate.call_args
    assert "response_format" not in call_kwargs
    assert get_image_from_openai_b64(response) == expected_img_data


@pytest.mark.requires_feature("sqlalchemy", "faiss")
def test_moderation() -> object:
    class MockEmbedding:
        def to_embeddings(self, data, **kwargs) -> object:
            return np.array([float(hash(str(data)) % 100) / 100.0] * 768, dtype=np.float32)

        @property
        def dimension(self) -> object:
            return 768

    init_similar_cache(
        data_dir=str(random.random()),
        pre_func=get_openai_moderation_input,
        embedding=MockEmbedding(),
    )
    expect_violence = 0.8864422

    mock_client = MagicMock()
    mock_client.moderations.create = MagicMock(
        return_value={
            "id": "modr-7IxkwrKvfnNJJIBsXAc0mfcpGaQJF",
            "model": "text-moderation-004",
            "results": [
                {
                    "categories": {
                        "hate": False,
                        "hate/threatening": False,
                        "self-harm": False,
                        "sexual": False,
                        "sexual/minors": False,
                        "violence": True,
                        "violence/graphic": False,
                    },
                    "category_scores": {
                        "hate": 0.18067425,
                        "hate/threatening": 0.0032884814,
                        "self-harm": 1.8089558e-09,
                        "sexual": 9.759996e-07,
                        "sexual/minors": 1.3364182e-08,
                        "violence": 0.8864422,
                        "violence/graphic": 3.2011528e-08,
                    },
                    "flagged": True,
                }
            ],
        }
    )

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.Moderation.create(
            input=["I want to kill them."],
        )
        assert response.get("results")[0].get("category_scores").get("violence") == expect_violence

    response = openai.Moderation.create(
        input=["I want to kill them."],
    )
    assert response.get("results")[0].get("category_scores").get("violence") == expect_violence

    expect_violence = 0.88708615
    mock_client2 = MagicMock()
    mock_client2.moderations.create = MagicMock(
        return_value={
            "id": "modr-7Ixe5Bvq4wqzZb1xtOxGxewg0G87F",
            "model": "text-moderation-004",
            "results": [
                {
                    "flagged": False,
                    "categories": {
                        "sexual": False,
                        "hate": False,
                        "violence": False,
                        "self-harm": False,
                        "sexual/minors": False,
                        "hate/threatening": False,
                        "violence/graphic": False,
                    },
                    "category_scores": {
                        "sexual": 1.5214279e-06,
                        "hate": 2.0188916e-06,
                        "violence": 1.8034231e-09,
                        "self-harm": 1.0547879e-10,
                        "sexual/minors": 2.6696927e-09,
                        "hate/threatening": 8.445262e-12,
                        "violence/graphic": 5.324232e-10,
                    },
                },
                {
                    "flagged": True,
                    "categories": {
                        "sexual": False,
                        "hate": False,
                        "violence": True,
                        "self-harm": False,
                        "sexual/minors": False,
                        "hate/threatening": False,
                        "violence/graphic": False,
                    },
                    "category_scores": {
                        "sexual": 9.5307604e-07,
                        "hate": 0.18386655,
                        "violence": 0.88708615,
                        "self-harm": 1.7594172e-09,
                        "sexual/minors": 1.3112497e-08,
                        "hate/threatening": 0.0032587533,
                        "violence/graphic": 3.1731048e-08,
                    },
                },
            ],
        }
    )
    with patch("byte.adapter.openai._get_client", return_value=mock_client2):
        response = openai.Moderation.create(
            input=["hello, world", "I want to kill them."],
        )
        assert not response.get("results")[0].get("flagged")
        assert response.get("results")[1].get("category_scores").get("violence") == expect_violence


@pytest.mark.requires_feature("sqlalchemy", "faiss")
def test_base_llm_cache() -> object:
    class MockEmbedding:
        def to_embeddings(self, data, **kwargs) -> object:
            return np.array([float(hash(str(data)) % 100) / 100.0] * 768, dtype=np.float32)

        @property
        def dimension(self) -> object:
            return 768

    cache_obj = Cache()
    init_similar_cache(
        data_dir=str(random.random()),
        pre_func=last_content,
        cache_obj=cache_obj,
        embedding=MockEmbedding(),
    )
    question = "What's Github"
    expect_answer = "Github is a great place to start"

    datas = {
        "choices": [
            {
                "message": {"content": expect_answer, "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": 1677825464,
        "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion.chunk",
    }

    mock_client = _make_mock_client(datas)

    import openai as real_openai

    def proxy_openai_chat_complete_exception(*args, **kwargs) -> None:
        raise real_openai.APIConnectionError(request=MagicMock())

    openai.ChatCompletion.llm = proxy_openai_chat_complete_exception

    is_openai_exception = False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            cache_obj=cache_obj,
        )
    except real_openai.APIConnectionError:
        is_openai_exception = True

    assert is_openai_exception

    is_proxy = False

    def proxy_openai_chat_complete(*args, **kwargs) -> object:
        nonlocal is_proxy
        is_proxy = True
        # Return the mock data directly
        return datas

    openai.ChatCompletion.llm = proxy_openai_chat_complete

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            cache_obj=cache_obj,
        )
        assert is_proxy
        assert get_message_from_openai_answer(response) == expect_answer, response

    is_exception = False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
    except Exception:
        is_exception = True
    assert is_exception

    openai.ChatCompletion.cache_args = {"cache_obj": cache_obj}

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )

    openai.ChatCompletion.llm = None
    openai.ChatCompletion.cache_args = {}
    assert get_message_from_openai_answer(response) == expect_answer, response
