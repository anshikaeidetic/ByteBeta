import pytest

from byte import Config, cache
from byte._backends import openai
from byte._testing.integration_support import (
    IntegrationTestBase,
    disable_cache,
    log_time_func,
    test_log,
)
from byte.embedding import Onnx
from byte.manager import VectorBase, get_data_manager
from byte.similarity_evaluation.distance import SearchDistanceEvaluation


class TestSqliteInvalid(IntegrationTestBase):
    """
    ******************************************************************
    #  The followings are the exception cases
    ******************************************************************
    """

    @pytest.mark.parametrize("threshold", [-1, 2, 2.0, 1000, "0.5"])
    @pytest.mark.tags("L1")
    def test_invalid_similarity_threshold(self, threshold) -> None:
        """
        target: test init: invalid similarity threshold
        method: input non-num and num which is out of range [0, 1]
        expected: raise exception and report the error
        """
        onnx = Onnx()
        vector_base = VectorBase("faiss", dimension=onnx.dimension)
        data_manager = get_data_manager("sqlite", vector_base, max_size=2000)
        is_exception = False
        try:
            cache.init(
                embedding_func=onnx.to_embeddings,
                data_manager=data_manager,
                similarity_evaluation=SearchDistanceEvaluation,
                config=Config(
                    log_time_func=log_time_func,
                    similarity_threshold=threshold,
                ),
            )
        except Exception as e:
            test_log.info(e)
            is_exception = True

        assert is_exception

    @pytest.mark.tags("L2")
    def test_no_openai_key(self) -> None:
        """
        target: test no openai key when could not hit in cache
        method: set similarity_threshold as 1 and no openai key
        expected: raise exception and report the error
        """
        onnx = Onnx()
        vector_base = VectorBase("faiss", dimension=onnx.dimension)
        data_manager = get_data_manager("sqlite", vector_base, max_size=2000)
        cache.init(
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation,
            config=Config(
                log_time_func=log_time_func,
                similarity_threshold=1,
            ),
        )

        is_exception = False
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "what do you think about assistant runtimes"},
                ],
            )
        except Exception as e:
            test_log.info(e)
            is_exception = True

        assert is_exception


class TestSqliteFaiss(IntegrationTestBase):
    """
    ******************************************************************
    #  The followings are general cases
    ******************************************************************
    """

    @pytest.mark.tags("L1")
    def test_hit_default(self) -> None:
        """
        target: test hit the cache function
        method: keep default similarity_threshold
        expected: hit successfully
        """

        onnx = Onnx()
        vector_base = VectorBase("faiss", dimension=onnx.dimension)
        data_manager = get_data_manager("sqlite", vector_base, max_size=2000)
        cache.init(
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            config=Config(
                log_time_func=log_time_func,
            ),
        )

        question = "what do you think about assistant runtimes"
        answer = "assistant runtimes are useful applications"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "what do you think about assistant runtimes"},
            ],
        )

    @pytest.mark.tags("L1")
    def test_hit(self) -> None:
        """
        target: test hit the cache function
        method: set similarity_threshold as 1
        expected: hit successfully
        """

        onnx = Onnx()
        vector_base = VectorBase("faiss", dimension=onnx.dimension)
        data_manager = get_data_manager("sqlite", vector_base, max_size=2000)
        cache.init(
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            config=Config(
                log_time_func=log_time_func,
                similarity_threshold=0.8,
            ),
        )

        question = "what do you think about assistant runtimes"
        answer = "assistant runtimes are useful applications"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "what do you think about assistant runtimes"},
            ],
        )

    @pytest.mark.tags("L1")
    def test_miss(self) -> None:
        """
        target: test miss the cache function
        method: set similarity_threshold as 0
        expected: raise exception and report the error
        """
        onnx = Onnx()
        vector_base = VectorBase("faiss", dimension=onnx.dimension)
        data_manager = get_data_manager("sqlite", vector_base, max_size=2000)
        cache.init(
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation,
            config=Config(
                log_time_func=log_time_func,
                similarity_threshold=0,
            ),
        )

        question = "what do you think about assistant runtimes"
        answer = "assistant runtimes are useful applications"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        is_exception = False
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "what do you think about assistant runtimes"},
                ],
            )
        except Exception as e:
            test_log.info(e)
            is_exception = True

        assert is_exception

    @pytest.mark.tags("L1")
    def test_disable_cache(self) -> None:
        """
        target: test cache not enabled
        method: set cache enable as false
        expected: hit successfully
        """

        onnx = Onnx()
        vector_base = VectorBase("faiss", dimension=onnx.dimension)
        data_manager = get_data_manager("sqlite", vector_base, max_size=2000)
        cache.init(
            cache_enable_func=disable_cache,
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=SearchDistanceEvaluation(),
            config=Config(
                log_time_func=log_time_func,
            ),
        )

        question = "what do you think about assistant runtimes"
        answer = "assistant runtimes are useful applications"
        cache.data_manager.save(question, answer, cache.embedding_func(question))

        is_exception = False
        try:
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "what do you think about assistant runtimes"},
                ],
            )
        except Exception as e:
            test_log.info(e)
            is_exception = True

        assert is_exception
