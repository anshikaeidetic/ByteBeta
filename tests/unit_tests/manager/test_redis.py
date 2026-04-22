import numpy as np
import pytest

redis_exceptions = pytest.importorskip("redis.exceptions")
RedisConnectionError = redis_exceptions.ConnectionError
ResponseError = redis_exceptions.ResponseError

from byte.embedding import Onnx
from byte.manager import VectorBase
from byte.manager.vector_data.base import VectorData


def test_redis_vector_store() -> None:
    encoder = Onnx()
    dim = encoder.dimension

    try:
        vector_base = VectorBase("redis", dimension=dim)
        vector_base.mul_add([VectorData(id=i, data=np.random.rand(dim)) for i in range(10)])

        search_res = vector_base.search(np.random.rand(dim))
        print(search_res)
        assert len(search_res) == 1

        search_res = vector_base.search(np.random.rand(dim), top_k=10)
        print(search_res)
        assert len(search_res) == 10

        vector_base.delete([i for i in range(5)])

        search_res = vector_base.search(np.random.rand(dim), top_k=10)
        print(search_res)
        assert len(search_res) == 5
    except (RedisConnectionError, ResponseError) as exc:
        pytest.skip(f"Redis vector search is not available for this test suite: {exc}")
