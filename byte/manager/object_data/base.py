from abc import ABC, abstractmethod
from typing import Any

from byte.utils.async_ops import run_sync


class ObjectBase(ABC):
    """
    Object storage base.
    """

    @abstractmethod
    def put(self, obj: Any) -> str:
        pass

    async def aput(self, obj: Any) -> str:
        return await run_sync(self.put, obj)

    @abstractmethod
    def get(self, obj: str) -> Any:
        pass

    async def aget(self, obj: str) -> Any:
        return await run_sync(self.get, obj)

    @abstractmethod
    def get_access_link(self, obj: str) -> str:
        pass

    async def aget_access_link(self, obj: str) -> str:
        return await run_sync(self.get_access_link, obj)

    @abstractmethod
    def delete(self, to_delete: list[str]) -> None:
        pass

    async def adelete(self, to_delete: list[str]) -> Any:
        return await run_sync(self.delete, to_delete)
