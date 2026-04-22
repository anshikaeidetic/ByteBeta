import os
import time

from byte import Config, cache
from byte._backends import openai
from byte.embedding import Onnx
from byte.manager import CacheBase, VectorBase, get_data_manager
from byte.similarity_evaluation.distance import SearchDistanceEvaluation


def test_sqlite_faiss_onnx() -> None:
    onnx = Onnx()

    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    if os.path.isfile(sqlite_file):
        os.remove(sqlite_file)
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base, max_size=2000)

    def log_time_func(func_name, delta_time) -> None:
        print(f"func `{func_name}` consume time: {delta_time:.2f}s")

    cache.init(
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(log_time_func=log_time_func, similarity_threshold=0.9),
    )

    question = "what do you think about assistant runtimes"
    answer = "assistant runtimes are useful applications"
    cache.import_data([question], [answer])

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what do you think about assistant runtimes"},
    ]

    start_time = time.time()
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    end_time = time.time()
    print(f"cache hint time consuming: {end_time - start_time:.2f}s")
    print(answer)

    is_exception = False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=mock_messages,
            cache_factor=100,
        )
    except Exception:
        is_exception = True

    assert is_exception

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what do you think about assistant runtimes"},
    ]
    is_exception = False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=mock_messages,
        )
    except Exception:
        is_exception = True

    assert is_exception

    is_exception = False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=mock_messages,
            cache_factor=0.5,
        )
    except Exception:
        is_exception = True

    assert not is_exception
