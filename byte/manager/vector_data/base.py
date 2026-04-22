from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from byte.utils.async_ops import run_sync


@dataclass
class VectorData:
    id: int
    data: np.ndarray


class VectorBase(ABC):
    """VectorBase: base vector store interface"""

    @abstractmethod
    def mul_add(self, datas: list[VectorData]) -> None:
        pass

    async def amul_add(self, datas: list[VectorData]) -> Any:
        return await run_sync(self.mul_add, datas)

    @abstractmethod
    def search(self, data: np.ndarray, top_k: int) -> None:
        pass

    async def asearch(self, data: np.ndarray, top_k: int) -> Any:
        return await run_sync(self.search, data, top_k)

    @abstractmethod
    def rebuild(self, ids=None) -> bool:
        pass

    async def arebuild(self, ids=None) -> bool:
        return await run_sync(self.rebuild, ids)

    @abstractmethod
    def delete(self, ids) -> bool:
        pass

    async def adelete(self, ids) -> bool:
        return await run_sync(self.delete, ids)

    def flush(self) -> None:
        return None

    async def aflush(self) -> Any:
        return await run_sync(self.flush)

    def close(self) -> Any:
        return self.flush()

    async def aclose(self) -> Any:
        return await run_sync(self.close)

    def get_embeddings(self, data_id: int | str) -> np.ndarray | None:
        raise NotImplementedError

    async def aget_embeddings(self, data_id: int | str) -> np.ndarray | None:
        return await run_sync(self.get_embeddings, data_id)

    @abstractmethod
    def update_embeddings(self, data_id: int | str, emb: np.ndarray) -> None:
        pass

    async def aupdate_embeddings(self, data_id: int | str, emb: np.ndarray) -> Any:
        return await run_sync(self.update_embeddings, data_id, emb)
