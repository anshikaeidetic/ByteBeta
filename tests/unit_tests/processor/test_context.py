from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

from byte import cache
from byte._backends import openai
from byte.manager import manager_factory
from byte.processor import ContextProcess
from byte.processor.pre import all_content
from byte.utils.response import get_message_from_openai_answer


class CITestContextProcess(ContextProcess):
    def __init__(self) -> None:
        self.content = ""

    def format_all_content(self, data: dict[str, Any], **params: dict[str, Any]) -> None:
        self.content = all_content(data)

    def process_all_content(self) -> (Any, Any):
        save_content = self.content.upper()
        embedding_content = self.content
        return save_content, embedding_content


def test_context_process() -> None:
    with TemporaryDirectory() as root:
        map_manager = manager_factory(data_dir=root)
        context_process = CITestContextProcess()
        cache.init(pre_embedding_func=context_process.pre_process, data_manager=map_manager)

        question = "test calculate 1+3"
        expect_answer = "the result is 4"
        with patch("byte.adapter.openai._get_client") as mock_get_client:
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
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = datas
            mock_get_client.return_value = mock_client

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
            )

            assert get_message_from_openai_answer(response) == expect_answer, response

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )
        answer_text = get_message_from_openai_answer(response)
        assert answer_text == expect_answer, answer_text
        cache.flush()

        map_manager = manager_factory(data_dir=root)
        content = f"You are a helpful assistant.\n{question}"
        cache_answer = map_manager.search(content)[0]
        assert cache_answer[0] == content.upper()
        assert cache_answer[1].answer == expect_answer
        assert cache_answer[2] == content
