from typing import Any

import numpy as np

from byte.similarity_evaluation import SimilarityEvaluation


class NumpyNormEvaluation(SimilarityEvaluation):
    """Using Numpy norm to evaluate sentences pair similarity."""

    def __init__(self, enable_normal: bool = True, question_embedding_function=None) -> None:
        self.enable_normal = enable_normal
        self.question_encoder = question_embedding_function

    @staticmethod
    def normalize(vec: np.ndarray) -> Any:
        magnitude = np.linalg.norm(vec)
        if magnitude == 0:
            return vec
        return vec / magnitude

    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **_) -> float:
        if "question" in src_dict and "question" in cache_dict:
            if src_dict["question"].lower() == cache_dict["question"].lower():
                return self.range()[1]
            if (
                "embedding" not in src_dict
                or "embedding" not in cache_dict
                or src_dict["embedding"] is None
                or cache_dict["embedding"] is None
            ):
                assert self.question_encoder, (
                    "You need to a valid question_embedding_function to generate "
                    "question embedding in the evaluator."
                )
                src_dict["embedding"] = self.question_encoder(src_dict["question"])
                cache_dict["embedding"] = self.question_encoder(cache_dict["question"])
        src_embedding = (
            self.normalize(src_dict["embedding"]) if self.enable_normal else src_dict["embedding"]
        )
        cache_embedding = (
            self.normalize(cache_dict["embedding"])
            if self.enable_normal
            else cache_dict["embedding"]
        )
        return max(0.0, self.range()[1] - np.linalg.norm(src_embedding - cache_embedding))

    def range(self) -> tuple[float, float]:
        return 0.0, 2.0
