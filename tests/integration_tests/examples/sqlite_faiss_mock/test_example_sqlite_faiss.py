import os

import numpy as np

from byte import Config, cache
from byte._backends import openai
from byte.manager import VectorBase, get_data_manager
from byte.similarity_evaluation.distance import SearchDistanceEvaluation

d = 8


def mock_embeddings(data, **kwargs) -> object:  # pylint: disable=W0613
    seed = sum(ord(char) for char in str(data))
    rng = np.random.default_rng(seed)
    return rng.random(d, dtype=np.float32)


def test_sqlite_faiss() -> None:
    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"

    if os.path.isfile(sqlite_file):
        os.remove(sqlite_file)
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)

    vector_base = VectorBase("faiss", dimension=d, top_k=3)
    data_manager = get_data_manager("sqlite", vector_base, max_size=8, clean_size=2)
    cache.init(
        embedding_func=mock_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(
            similarity_threshold=0.8,
        ),
    )

    mock_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "foo"},
    ]
    cache.import_data(["foo"], ["receiver the foo"])

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    assert answer["choices"][0]["message"]["content"] == "receiver the foo"

    cache.flush()
    vector_base = VectorBase("faiss", dimension=d, top_k=3)
    data_manager = get_data_manager("sqlite", vector_base, max_size=8, clean_size=2)
    cache.init(
        embedding_func=mock_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(
            similarity_threshold=0.8,
        ),
    )
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mock_messages,
    )
    assert answer["choices"][0]["message"]["content"] == "receiver the foo"
