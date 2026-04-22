from tempfile import TemporaryDirectory

import pytest

from byte import Config, cache
from byte._backends import openai
from byte._testing.integration_support import IntegrationTestBase, log_time_func
from byte.embedding import SBERT
from byte.manager import manager_factory
from byte.similarity_evaluation.distance import SearchDistanceEvaluation

pytestmark = pytest.mark.integration_live


def get_text_response(response) -> object:
    if response is None:
        return ""
    collated_response = [chunk["choices"][0]["delta"].get("content", "") for chunk in response]
    return "".join(collated_response)


class TestSqliteMilvus(IntegrationTestBase):
    """
    ******************************************************************
    #  The followings are general cases
    ******************************************************************
    """

    @pytest.mark.tags("L1")
    def test_cache_health_check(self, live_service_stack) -> None:
        """
        target: test hit the cache function
        method: keep default similarity_threshold
        expected: cache health detection & correction
        """
        with TemporaryDirectory() as root:
            try:
                onnx = SBERT()
            except ModuleNotFoundError as exc:
                pytest.skip(f"SBERT integration requires optional dependencies: {exc}")

            data_manager_factories = [
                lambda: manager_factory(
                    "sqlite,milvus",
                    data_dir=str(root),
                    vector_params={
                        "dimension": onnx.dimension,
                        "local_mode": True,
                        "port": "10086",
                    },
                ),
                lambda: manager_factory("sqlite,chromadb", data_dir=str(root)),
            ]

            ran_any = False
            for build_data_manager in data_manager_factories:
                try:
                    data_manager = build_data_manager()
                except ModuleNotFoundError:
                    continue

                ran_any = True
                cache.init(
                    embedding_func=onnx.to_embeddings,
                    data_manager=data_manager,
                    similarity_evaluation=SearchDistanceEvaluation(),
                    config=Config(
                        log_time_func=log_time_func,
                        enable_token_counter=False,
                    ),
                )

                question = [
                    "what is apple?",
                    "what is intel?",
                    "what is openai?",
                ]
                answer = [
                    "apple",
                    "intel",
                    "openai",
                ]
                for q, a in zip(question, answer):
                    cache.data_manager.save(q, a, cache.embedding_func(q))

                # Simulate cache out-of-sync by corrupting the vector entry.
                trouble_query = "what is google?"
                cache.data_manager.v.update_embeddings(1, cache.embedding_func(trouble_query))

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": trouble_query},
                    ],
                    search_only=True,
                    stream=True,
                )
                assert get_text_response(response) == answer[0]

                cache.config.data_check = True
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": trouble_query},
                    ],
                    search_only=True,
                    stream=True,
                )
                assert response is None

                cache.config.data_check = False
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": trouble_query},
                    ],
                    search_only=True,
                    stream=True,
                )
                assert response is None

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question[0]},
                    ],
                    search_only=True,
                    stream=True,
                )
                assert get_text_response(response) == answer[0]

            if not ran_any:
                pytest.skip(
                    "Milvus/Chroma integration backends are not available in this environment."
                )
