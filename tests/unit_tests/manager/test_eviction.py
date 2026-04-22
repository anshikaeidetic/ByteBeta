import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from byte.manager import CacheBase, VectorBase, get_data_manager

DIM = 8
pytestmark = pytest.mark.requires_feature("sqlalchemy", "faiss")


def mock_embeddings() -> object:
    return np.random.random((DIM,)).astype("float32")


class TestEviction(unittest.TestCase):
    """Test data eviction"""

    def test_eviction_lru(self) -> None:
        with TemporaryDirectory() as root:
            db_path = Path(root) / "sqlite.db"
            cache_base = CacheBase("sqlite", sql_url="sqlite:///" + str(db_path))
            vector_base = VectorBase("faiss", dimension=DIM)
            data_manager = get_data_manager(
                cache_base, vector_base, max_size=10, clean_size=2, eviction="LRU"
            )
            for i in range(19):
                question = f"foo{i}"
                answer = f"receiver the foo {i}"
                data_manager.save(question, answer, mock_embeddings())
            cache_count = data_manager.s.count()
            self.assertEqual(cache_count, 9)
            ids = data_manager.s.get_ids(deleted=True)
            self.assertEqual(len(ids), 0)

    def test_eviction_fifo(self) -> None:
        with TemporaryDirectory() as root:
            db_path = Path(root) / "sqlite.db"
            cache_base = CacheBase("sqlite", sql_url="sqlite:///" + str(db_path))
            vector_base = VectorBase("faiss", dimension=DIM)
            data_manager = get_data_manager(
                cache_base, vector_base, max_size=10, clean_size=2, eviction="FIFO"
            )
            for i in range(18):
                question = f"foo{i}"
                answer = f"receiver the foo {i}"
                data_manager.save(question, answer, mock_embeddings())

            cache_count = data_manager.s.count()
            self.assertEqual(cache_count, 10)

    # def test_eviction_milvus(self):
    #     cache_base = CacheBase('sqlite', sql_url='sqlite:///./byte2.db')
    #     vector_base = VectorBase('milvus', dimension=DIM, host='172.16.70.4', collection_name='byte2')
    #     data_manager = get_data_manager(cache_base, vector_base, max_size=10, clean_size=2, eviction='LRU')
    #     for i in range(10):
    #         question = f'foo{i}'
    #         answer = f'receiver the foo {i}'
    #         data_manager.save(question, answer, mock_embeddings())
    #
    #     cache_count = data_manager.s.count(is_all=True)
    #     self.assertEqual(cache_count, 10)
