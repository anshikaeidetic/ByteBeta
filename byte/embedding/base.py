from abc import ABCMeta, abstractmethod


class BaseEmbedding(metaclass=ABCMeta):
    """
    Base Embedding interface.
    """

    @abstractmethod
    def to_embeddings(self, data, **kwargs) -> None:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        return 0
