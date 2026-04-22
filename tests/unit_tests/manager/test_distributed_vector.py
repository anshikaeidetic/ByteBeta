import numpy as np

from byte.manager.vector_data import VectorBase
from byte.manager.vector_data.base import VectorBase as AbstractVectorBase
from byte.manager.vector_data.base import VectorData


class _FakeVectorStore(AbstractVectorBase):
    def __init__(self, name) -> None:
        self.name = name
        self.saved = []

    def mul_add(self, datas) -> None:
        self.saved.extend(datas)

    def search(self, data: np.ndarray, top_k: int) -> object:
        return [(0.9 - (index * 0.1), item.id) for index, item in enumerate(self.saved[:top_k])]

    def rebuild(self, ids=None) -> object:
        return True

    def delete(self, ids) -> object:
        self.saved = [item for item in self.saved if item.id not in set(ids)]
        return True

    def get_embeddings(self, data_id) -> object:
        for item in self.saved:
            if item.id == data_id:
                return item.data
        return None

    def update_embeddings(self, data_id, emb: np.ndarray) -> None:
        for item in self.saved:
            if item.id == data_id:
                item.data = emb


def test_distributed_vector_store_hash_shards_and_merges_results() -> None:
    shard_a = _FakeVectorStore("a")
    shard_b = _FakeVectorStore("b")
    distributed = VectorBase(
        "distributed", stores=[shard_a, shard_b], top_k=4, shard_strategy="hash"
    )

    distributed.mul_add(
        [
            VectorData(id=1, data=np.array([1.0, 0.0], dtype="float32")),
            VectorData(id=2, data=np.array([0.0, 1.0], dtype="float32")),
            VectorData(id=3, data=np.array([1.0, 1.0], dtype="float32")),
        ]
    )

    assert len(shard_a.saved) + len(shard_b.saved) == 3
    results = distributed.search(np.array([1.0, 0.0], dtype="float32"), top_k=4)
    returned_ids = {item_id for _, item_id in results}
    assert returned_ids == {1, 2, 3}


def test_distributed_vector_store_broadcast_replicates_updates() -> None:
    shard_a = _FakeVectorStore("a")
    shard_b = _FakeVectorStore("b")
    distributed = VectorBase("distributed", stores=[shard_a, shard_b], shard_strategy="broadcast")

    distributed.mul_add([VectorData(id=9, data=np.array([0.5, 0.5], dtype="float32"))])
    assert len(shard_a.saved) == 1
    assert len(shard_b.saved) == 1

    distributed.update_embeddings(9, np.array([0.2, 0.8], dtype="float32"))
    assert np.allclose(shard_a.get_embeddings(9), np.array([0.2, 0.8], dtype="float32"))
    assert np.allclose(shard_b.get_embeddings(9), np.array([0.2, 0.8], dtype="float32"))
