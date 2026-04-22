from abc import ABCMeta, abstractmethod
from typing import Any


class EvictionBase(metaclass=ABCMeta):
    """
    Eviction base.
    """

    @abstractmethod
    def put(self, objs: list[Any]) -> None:
        pass

    @abstractmethod
    def get(self, obj: Any) -> None:
        pass

    @property
    @abstractmethod
    def policy(self) -> str:
        pass
