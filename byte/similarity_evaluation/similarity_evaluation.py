from abc import ABCMeta, abstractmethod
from typing import Any

from byte.utils.async_ops import run_sync


class SimilarityEvaluation(metaclass=ABCMeta):
    """Similarity Evaluation interface,
    determine the similarity between the input request and the requests from the Vector Store.
    Based on this similarity, it determines whether a request matches the cache.

    Example:
        .. code-block:: python

            from byte import cache
            from byte.similarity_evaluation import SearchDistanceEvaluation

            cache.init(
                similarity_evaluation=SearchDistanceEvaluation()
            )
    """

    @abstractmethod
    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **kwargs) -> float:
        """Evaluate the similarity score of the user and cache requests pair.

        :param src_dict: the user request params.
        :type src_dict: Dict
        :param cache_dict: the cache request params.
        :type cache_dict: Dict
        """

    async def aevaluation(
        self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **kwargs
    ) -> float:
        return await run_sync(self.evaluation, src_dict, cache_dict, **kwargs)

    @abstractmethod
    def range(self) -> tuple[float, float]:
        """Range of similarity score.

        :return: the range of similarity score, which is the min and max values
        :rtype: Tuple[float, float]
        """
