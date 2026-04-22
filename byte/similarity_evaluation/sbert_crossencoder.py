from typing import Any

from byte.similarity_evaluation import SimilarityEvaluation
from byte.utils import lazy_optional_module

sentence_transformers = lazy_optional_module(
    "sentence_transformers",
    package="sentence-transformers",
)


class SbertCrossencoderEvaluation(SimilarityEvaluation):
    """Using SBERT crossencoders to evaluate sentences pair similarity.

    This evaluator use the crossencoder model to evaluate the similarity of two sentences.

    :param model: model name of SbertCrossencoderEvaluation. Default is 'cross-encoder/quora-distilroberta-base'.
    Check more please refer to https://www.sbert.net/docs/pretrained_cross-encoders.html#quora-duplicate-questions.
    :type model: str

    Example:
        .. code-block:: python

            from byte.similarity_evaluation import SbertCrossencoderEvaluation

            evaluation = SbertCrossencoderEvaluation()
            score = evaluation.evaluation(
                {
                    'question': 'What is the color of sky?'
                },
                {
                    'question': 'hello'
                }
            )
    """

    def __init__(self, model: str = "cross-encoder/quora-distilroberta-base") -> None:
        self.model = sentence_transformers.CrossEncoder(model)

    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **_) -> float:
        """Evaluate the similarity score of pair.

        :param src_dict: the query dictionary to evaluate with cache.
        :type src_dict: Dict
        :param cache_dict: the cache dictionary.
        :type cache_dict: Dict

        :return: evaluation score.
        """
        try:
            src_question = src_dict["question"]
            cache_question = cache_dict["question"]
            if src_question.lower() == cache_question.lower():
                return 1
            return self.model.predict([(src_question, cache_question)])[0]
        except Exception:  # pylint: disable=W0703
            return 0

    def range(self) -> tuple[float, float]:
        """Range of similarity score.

        :return: minimum and maximum of similarity score.
        """
        return 0.0, 1.0
