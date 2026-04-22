import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from byte import Cache
from byte.embedding import Onnx
from byte.manager.factory import manager_factory
from byte.processor.pre import get_prompt

question = "test_llama_cpp"
expect_answer = "hello world"
onnx = Onnx()


class MockLlama:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def create_completion(*args, **kwargs) -> object:
        data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "text": expect_answer,
                }
            ],
            "created": 1677825456,
            "id": "chatcmpl-6ptKqrhgRoVchm58Bby0UvJzq2ZuQ",
            "model": "llam_cpp",
            "object": "chat.completion",
            "usage": {"completion_tokens": 301, "prompt_tokens": 36, "total_tokens": 337},
        }
        if not kwargs.get("stream", False):
            return data
        else:
            return iter([data])


mock_module = MagicMock()
sys.modules["llama_cpp"] = mock_module


class TestLlama(unittest.TestCase):
    def test_llama_cpp(self) -> None:
        mock_module.Llama = MockLlama
        with TemporaryDirectory() as root:
            m = manager_factory(
                "sqlite,faiss,local", data_dir=root, vector_params={"dimension": onnx.dimension}
            )
            llm_cache = Cache()
            llm_cache.init(
                pre_embedding_func=get_prompt, data_manager=m, embedding_func=onnx.to_embeddings
            )

            with patch("byte.utils.import_llama_cpp_python"):
                from byte.adapter.llama_cpp import Llama

                llm = Llama("model.bin")
                answer = llm(prompt=question, cache_obj=llm_cache)
                assert expect_answer == answer["choices"][0]["text"]

                answer2 = llm(prompt=question, cache_obj=llm_cache)
                assert answer2["byte"] is True
                assert expect_answer == answer2["choices"][0]["text"]

                llm(prompt=question, cache_obj=llm_cache, stream=True, stop=["\n"])

                answer = llm(prompt=question, cache_obj=llm_cache, stream=True)
                for item in answer:
                    self.assertEqual(item["choices"][0]["text"], expect_answer)

    def test_llama_cpp_stream(self) -> None:
        with TemporaryDirectory() as root:
            m = manager_factory(
                "sqlite,faiss,local", data_dir=root, vector_params={"dimension": onnx.dimension}
            )
            llm_cache = Cache()
            llm_cache.init(
                pre_embedding_func=get_prompt, data_manager=m, embedding_func=onnx.to_embeddings
            )

            with patch("byte.utils.import_llama_cpp_python"):
                from byte.adapter.llama_cpp import Llama

                llm = Llama("model.bin")
                answer = llm(prompt=question, cache_obj=llm_cache, stream=True)
                for item in answer:
                    assert expect_answer == item["choices"][0]["text"]

                answer2 = llm(prompt=question, cache_obj=llm_cache)
                assert answer2["byte"] is True
                assert expect_answer == answer2["choices"][0]["text"]

                answer = llm(prompt=question, cache_obj=llm_cache, stream=True)
                for item in answer:
                    self.assertEqual(item["choices"][0]["text"], expect_answer)
