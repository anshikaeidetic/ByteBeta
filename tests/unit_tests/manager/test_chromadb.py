import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from byte.manager import VectorBase
from byte.manager.vector_data.base import VectorData
from byte.manager.vector_data.chroma import _FallbackCollection


class TestChromadb(unittest.TestCase):
    def test_normal(self) -> None:
        db = VectorBase("chromadb", client_settings={}, top_k=3)
        db.mul_add([VectorData(id=i, data=np.random.sample(10)) for i in range(100)])
        search_res = db.search(np.random.sample(10))
        self.assertEqual(len(search_res), 3)
        db.delete(["1", "3", "5", "7"])
        self.assertEqual(db._collection.count(), 96)

    def test_fallback_collection_parallel_add_and_query(self) -> None:
        collection = _FallbackCollection()
        data = [np.random.sample(10).astype("float32") for _ in range(64)]

        def writer(offset) -> None:
            for index in range(32):
                item_id = f"{offset}-{index}"
                collection.add([data[(offset + index) % len(data)]], [item_id])

        def reader() -> None:
            for _ in range(80):
                result = collection.query([data[0]], n_results=3)
                self.assertLessEqual(len(result["ids"][0]), 3)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(writer, 0),
                pool.submit(writer, 32),
                pool.submit(reader),
                pool.submit(reader),
            ]
            for future in futures:
                future.result()

        self.assertGreater(collection.count(), 0)

    def test_flush_and_close_satisfy_contract(self) -> None:
        db = VectorBase("chromadb", client_settings={}, top_k=3)
        db.flush()
        db.close()
