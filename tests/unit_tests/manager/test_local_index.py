import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from byte.manager.vector_data import VectorBase
from byte.manager.vector_data.base import VectorData
from byte.manager.vector_data.hnswlib_store import Hnswlib, _FallbackIndex

DIM = 512
MAX_ELEMENTS = 10000
SIZE = 1000
TOP_K = 10


class TestLocalIndex(unittest.TestCase):
    @pytest.mark.requires_feature("faiss")
    def test_faiss(self) -> None:
        from byte.manager.vector_data.faiss import Faiss

        cls = partial(Faiss, dimension=DIM)
        self._internal_test_normal(cls)
        self._internal_test_with_rebuild(cls)
        self._internal_test_reload(cls)
        self._internal_test_delete(cls)

        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            self._internal_test_create_from_vector_base(
                name="faiss", top_k=3, dimension=DIM, index_path=index_path
            )

    def test_hnswlib(self) -> None:
        cls = partial(Hnswlib, max_elements=MAX_ELEMENTS, dimension=DIM)
        self._internal_test_normal(cls)
        self._internal_test_with_rebuild(cls)
        self._internal_test_reload(cls)
        self._internal_test_delete(cls)

        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            self._internal_test_create_from_vector_base(
                name="hnswlib",
                top_k=3,
                dimension=DIM,
                index_path=index_path,
                max_elements=MAX_ELEMENTS,
            )

    def test_hnswlib_fallback_reload(self) -> None:
        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            index = _FallbackIndex()
            data = np.random.randn(5, DIM).astype(np.float32)
            index.add_items(data, np.arange(5))
            index.save_index(index_path)

            reloaded = _FallbackIndex()
            reloaded.load_index(index_path)
            self.assertEqual(len(reloaded.knn_query(data[:1], k=3)[0][0]), 3)

    def test_hnswlib_fallback_parallel_add_and_query(self) -> None:
        index = _FallbackIndex()
        data = np.random.randn(64, DIM).astype(np.float32)

        def writer(offset) -> None:
            for batch_start in range(0, 32, 4):
                batch = data[offset + batch_start : offset + batch_start + 4]
                ids = np.arange(offset + batch_start, offset + batch_start + len(batch))
                index.add_items(batch, ids)

        def reader() -> None:
            for _ in range(60):
                ids, distances = index.knn_query(data[:1], k=3)
                self.assertLessEqual(len(ids[0]), 3)
                self.assertEqual(ids.shape, distances.shape)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(writer, 0),
                pool.submit(writer, 32),
                pool.submit(reader),
                pool.submit(reader),
            ]
            for future in futures:
                future.result()

    @pytest.mark.requires_feature("docarray")
    def test_docarray(self) -> None:
        from byte.manager.vector_data.docarray_index import DocArrayIndex

        self._internal_test_normal(DocArrayIndex)
        self._internal_test_with_rebuild(DocArrayIndex)
        self._internal_test_reload(DocArrayIndex)
        self._internal_test_delete(DocArrayIndex)

        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            self._internal_test_create_from_vector_base(
                name="docarray", top_k=3, index_path=index_path
            )

    def _internal_test_normal(self, vector_class) -> None:
        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))])
            self.assertEqual(len(index.search(data[0])), TOP_K)
            index.mul_add([VectorData(id=SIZE, data=data[0])])
            ret = index.search(data[0])
            self.assertIn(ret[0][1], [0, SIZE])
            self.assertIn(ret[1][1], [0, SIZE])

    def _internal_test_with_rebuild(self, vector_class) -> None:
        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))])
            index.delete([0, 1, 2])
            index.rebuild(list(range(3, SIZE)))
            self.assertNotEqual(index.search(data[0])[0], 0)

    def _internal_test_reload(self, vector_class) -> None:
        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))])
            index.close()

            new_index = vector_class(index_file_path=index_path, top_k=TOP_K)
            self.assertEqual(len(new_index.search(data[0])), TOP_K)
            new_index.mul_add([VectorData(id=SIZE, data=data[0])])
            ret = new_index.search(data[0])
            self.assertIn(ret[0][1], [0, SIZE])
            self.assertIn(ret[1][1], [0, SIZE])

    def _internal_test_delete(self, vector_class) -> None:
        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            index = vector_class(index_file_path=index_path, top_k=TOP_K)
            data = np.random.randn(SIZE, DIM).astype(np.float32)
            index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, list(range(SIZE)))])
            self.assertEqual(len(index.search(data[0])), TOP_K)
            index.delete([0, 1, 2, 3])
            self.assertNotEqual(index.search(data[0])[0][1], 0)
            if hasattr(index, "count"):
                self.assertEqual(index.count(), 996)

    def _internal_test_create_from_vector_base(self, **kwargs) -> None:
        index = VectorBase(**kwargs)
        data = np.random.randn(100, DIM).astype(np.float32)
        index.mul_add([VectorData(id=i, data=v) for v, i in zip(data, range(100))])
        self.assertEqual(index.search(data[0])[0][1], 0)

    @pytest.mark.requires_feature("faiss")
    def test_faiss_update_embeddings_and_get_embeddings(self) -> None:
        from byte.manager.vector_data.faiss import Faiss

        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            index = Faiss(index_file_path=index_path, dimension=DIM, top_k=3)
            original = np.random.randn(DIM).astype(np.float32)
            replacement = np.random.randn(DIM).astype(np.float32)
            other = np.random.randn(DIM).astype(np.float32)
            index.mul_add([VectorData(id=1, data=original), VectorData(id=2, data=other)])

            np.testing.assert_allclose(index.get_embeddings(1), original, rtol=0, atol=1e-6)

            index.update_embeddings(1, replacement)

            np.testing.assert_allclose(index.get_embeddings(1), replacement, rtol=0, atol=1e-6)
            self.assertEqual(index.search(replacement)[0][1], 1)

    def test_hnswlib_update_embeddings_and_get_embeddings(self) -> None:
        with TemporaryDirectory() as root:
            index_path = str((Path(root) / "index.bin").absolute())
            index = Hnswlib(
                index_file_path=index_path,
                dimension=DIM,
                top_k=3,
                max_elements=MAX_ELEMENTS,
            )
            original = np.random.randn(DIM).astype(np.float32)
            replacement = np.random.randn(DIM).astype(np.float32)
            other = np.random.randn(DIM).astype(np.float32)
            index.mul_add([VectorData(id=1, data=original), VectorData(id=2, data=other)])

            np.testing.assert_allclose(index.get_embeddings(1), original, rtol=0, atol=1e-6)

            index.update_embeddings(1, replacement)

            np.testing.assert_allclose(index.get_embeddings(1), replacement, rtol=0, atol=1e-6)
            self.assertEqual(index.search(replacement)[0][1], 1)
