import random
from unittest.mock import patch

import pytest

from byte import Cache
from byte._backends import openai
from byte.adapter.api import init_similar_cache
from byte.embedding import Onnx
from byte.manager import manager_factory
from byte.processor.pre import last_content
from byte.utils.response import get_message_from_openai_answer

pytestmark = pytest.mark.integration_live


def test_redis_sqlite(live_service_stack) -> None:
    encoder = Onnx()

    try:
        redis_data_managers = [
            manager_factory(
                "sqlite,redis",
                data_dir=str(random.random()),
                vector_params={"dimension": encoder.dimension},
            ),
            manager_factory(
                "redis,redis",
                data_dir=str(random.random()),
                scalar_params={"global_key_prefix": "byte_scalar"},
                vector_params={
                    "dimension": encoder.dimension,
                    "namespace": "byte_vector",
                    "collection_name": "cache_vector",
                },
            ),
        ]
    except Exception as exc:  # pylint: disable=W0703
        pytest.skip(f"Redis integration requires a local Redis/RediSearch instance: {exc}")

    for redis_data_manager in redis_data_managers:
        redis_cache = Cache()
        init_similar_cache(
            cache_obj=redis_cache,
            pre_func=last_content,
            embedding=encoder,
            data_manager=redis_data_manager,
        )
        question = "what's github"
        expect_answer = "GitHub is an online platform used primarily for version control and coding collaborations."
        with patch("openai.ChatCompletion.create") as mock_create:
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
            mock_create.return_value = datas

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                cache_obj=redis_cache,
            )

            assert get_message_from_openai_answer(response) == expect_answer, response

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "can you explain what GitHub is"},
            ],
            cache_obj=redis_cache,
        )
        answer_text = get_message_from_openai_answer(response)
        assert answer_text == expect_answer, answer_text
