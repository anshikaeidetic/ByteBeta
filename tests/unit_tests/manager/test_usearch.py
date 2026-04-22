import unittest

import numpy as np
import pytest

from byte.manager.vector_data import VectorBase
from byte.manager.vector_data.base import VectorData

pytest.importorskip("usearch")


class TestUSearchDB(unittest.TestCase):
    def test_normal(self) -> None:
        size = 1000
        dim = 512
        top_k = 10

        db = VectorBase(
            "usearch",
            index_file_path="./index.usearch",
            dimension=dim,
            top_k=top_k,
            metric="cos",
            dtype="f32",
        )
        db.mul_add([VectorData(id=i, data=np.random.rand(dim)) for i in range(size)])
        self.assertEqual(len(db.search(np.random.rand(dim))), top_k)
        self.assertEqual(db.count(), size)
        db.close()

    def test_delete_update_and_reload(self) -> None:
        dim = 16
        db = VectorBase(
            "usearch",
            index_file_path="./index_stateful.usearch",
            dimension=dim,
            top_k=3,
            metric="cos",
            dtype="f32",
        )
        emb1 = np.random.rand(dim).astype("float32")
        emb2 = np.random.rand(dim).astype("float32")
        db.mul_add([VectorData(id=1, data=emb1), VectorData(id=2, data=emb2)])

        self.assertEqual(db.count(), 2)
        self.assertIsNotNone(db.get_embeddings(1))

        updated = np.ones(dim, dtype="float32")
        self.assertTrue(db.update_embeddings(1, updated))
        self.assertTrue(np.allclose(db.get_embeddings(1), updated))

        self.assertTrue(db.delete(2))
        self.assertEqual(db.count(), 1)
        db.flush()
        db.close()

        reloaded = VectorBase(
            "usearch",
            index_file_path="./index_stateful.usearch",
            dimension=dim,
            top_k=3,
            metric="cos",
            dtype="f32",
        )
        self.assertEqual(reloaded.count(), 1)
        self.assertTrue(np.allclose(reloaded.get_embeddings(1), updated))
        self.assertIsNone(reloaded.get_embeddings(2))
        reloaded.close()
