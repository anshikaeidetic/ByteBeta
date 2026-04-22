import time
from collections.abc import Iterator
from typing import Any

from byte.adapter.adapter import adapt
from byte.manager.scalar_data.base import Answer, DataType
from byte.utils import load_optional_module


def _load_llama_cpp() -> Any:
    return load_optional_module("llama_cpp", package="llama-cpp-python")


class Llama:
    """llama.cpp wrapper

        You should have the llama-cpp-python library installed.
        https://github.com/abetlen/llama-cpp-python

    Example:
        .. code-block:: python

            onnx = Onnx()
            m = manager_factory('sqlite,faiss,local', data_dir=root, vector_params={"dimension": onnx.dimension})
            llm_cache = Cache()
            llm_cache.init(
                pre_embedding_func=get_prompt,
                data_manager=m,
                embedding_func=onnx.to_embeddings
            )
            llm = Llama('./models/7B/ggml-model.bin')
            answer = llm(prompt=question, cache_obj=llm_cache)
    """

    def __init__(self, *args, **kwargs) -> None:
        self._delegate = _load_llama_cpp().Llama(*args, **kwargs)

    def __getattr__(self, item) -> Any:
        return getattr(self._delegate, item)

    def __call__(self, prompt: str, **kwargs) -> Any:
        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            if not isinstance(llm_data, Iterator):
                update_cache_func(Answer(llm_data["choices"][0]["text"], DataType.STR))
                return llm_data
            else:

                def stream_answer(it) -> Any:
                    total_answer = ""
                    for item in it:
                        total_answer += item["choices"][0]["text"]
                        yield item
                    update_cache_func(Answer(total_answer, DataType.STR))

                return stream_answer(llm_data)

        def cache_data_convert(cache_data) -> Any:
            if kwargs.get("stream", False):
                return _construct_stream_resp_from_cache(cache_data)
            return _construct_resp_from_cache(cache_data)

        return adapt(
            self._delegate.create_completion,
            cache_data_convert,
            update_cache_callback,
            prompt=prompt,
            **kwargs,
        )


def _construct_resp_from_cache(return_message) -> dict[str, Any]:
    return {
        "byte": True,
        "choices": [
            {
                "text": return_message,
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": int(time.time()),
        "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        "object": "chat.completion",
    }


def _construct_stream_resp_from_cache(return_message) -> list[Any]:
    return [
        {
            "byte": True,
            "choices": [
                {
                    "text": return_message,
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "created": int(time.time()),
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
            "object": "chat.completion",
        }
    ]
