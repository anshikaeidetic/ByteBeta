from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from byte.manager.vector_data.base import VectorBase, VectorData


class DistributedVectorStore(VectorBase):
    """Fan out vector search across multiple shards and merge the candidates."""

    def __init__(
        self,
        *,
        stores: list[VectorBase],
        top_k: int = 1,
        shard_strategy: str = "hash",
    ) -> None:
        if not stores:
            raise ValueError("DistributedVectorStore requires at least one shard store.")
        if shard_strategy not in {"hash", "broadcast"}:
            raise ValueError("shard_strategy must be either 'hash' or 'broadcast'.")
        self.stores = list(stores)
        self.top_k = int(top_k or 1)
        self.shard_strategy = shard_strategy

    def _shard_index(self, data_id) -> int:
        return abs(hash(str(data_id))) % len(self.stores)

    def _target_indexes(self, data_id) -> list[int]:
        if self.shard_strategy == "broadcast":
            return list(range(len(self.stores)))
        return [self._shard_index(data_id)]

    def mul_add(self, datas: list[VectorData]) -> None:
        if self.shard_strategy == "broadcast":
            for store in self.stores:
                store.mul_add(list(datas))
            return
        grouped = defaultdict(list)
        for data in datas:
            grouped[self._shard_index(data.id)].append(data)
        for index, batch in grouped.items():
            self.stores[index].mul_add(batch)

    def search(self, data: np.ndarray, top_k: int = -1) -> Any:
        request_top_k = self.top_k if top_k == -1 else int(top_k or self.top_k)
        merged = []
        for store in self.stores:
            merged.extend(store.search(data, request_top_k) or [])
        deduped = {}
        for score, item_id in merged:
            current = deduped.get(item_id)
            if current is None or float(score) > float(current):
                deduped[item_id] = float(score)
        return sorted(((score, item_id) for item_id, score in deduped.items()), reverse=True)[
            :request_top_k
        ]

    def delete(self, ids) -> None:
        if self.shard_strategy == "broadcast":
            for store in self.stores:
                store.delete(ids)
            return
        grouped = defaultdict(list)
        for item_id in ids:
            grouped[self._shard_index(item_id)].append(item_id)
        for index, batch in grouped.items():
            self.stores[index].delete(batch)

    def rebuild(self, ids=None) -> bool:
        for store in self.stores:
            store.rebuild(ids=ids)
        return True

    def flush(self) -> None:
        for store in self.stores:
            if hasattr(store, "flush"):
                store.flush()

    def close(self) -> None:
        for store in self.stores:
            if hasattr(store, "close"):
                store.close()

    def get_embeddings(self, data_id) -> Any | None:
        indexes: Iterable[int]
        indexes = self._target_indexes(data_id)
        for index in indexes:
            if hasattr(self.stores[index], "get_embeddings"):
                embedding = self.stores[index].get_embeddings(data_id)
                if embedding is not None:
                    return embedding
        return None

    def update_embeddings(self, data_id, emb: np.ndarray) -> None:
        if self.shard_strategy == "broadcast":
            for store in self.stores:
                store.update_embeddings(data_id, emb)
            return
        self.stores[self._shard_index(data_id)].update_embeddings(data_id, emb)
