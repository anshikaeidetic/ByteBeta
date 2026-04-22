from typing import Any

from byte.processor.pre import canonicalize_text, normalize_text
from byte.similarity_evaluation.similarity_evaluation import SimilarityEvaluation

_STRUCTURED_CANONICAL_PREFIXES = (
    "exact_answer::",
    "classify::",
    "translate::",
    "summarize::",
    "extract::",
)


class GuardedSimilarityEvaluation(SimilarityEvaluation):
    """Wrap another evaluator with lexical and canonical safety guards."""

    def __init__(
        self,
        base_evaluation: SimilarityEvaluation,
        *,
        min_token_overlap: float = 0.0,
        max_length_ratio: float | None = None,
        enforce_canonical_match: bool = False,
    ) -> None:
        self._base = base_evaluation
        self._min_token_overlap = min_token_overlap
        self._max_length_ratio = max_length_ratio
        self._enforce_canonical_match = enforce_canonical_match

    def evaluation(self, src_dict: dict[str, Any], cache_dict: dict[str, Any], **kwargs) -> float:
        score = self._base.evaluation(src_dict, cache_dict, **kwargs)
        min_rank, _ = self.range()

        src_question = src_dict.get("question")
        cache_question = cache_dict.get("question")
        if not isinstance(src_question, str) or not isinstance(cache_question, str):
            return score

        src_canonical = canonicalize_text(src_question)
        cache_canonical = canonicalize_text(cache_question)
        if (
            self._enforce_canonical_match
            and src_canonical != cache_canonical
            and (
                _is_structured_canonical(src_canonical) or _is_structured_canonical(cache_canonical)
            )
        ):
            return min_rank

        src_tokens = _tokenize(src_question)
        cache_tokens = _tokenize(cache_question)
        if self._max_length_ratio and src_tokens and cache_tokens:
            length_ratio = max(len(src_tokens), len(cache_tokens)) / max(
                min(len(src_tokens), len(cache_tokens)),
                1,
            )
            if length_ratio > self._max_length_ratio:
                return min_rank

        if self._min_token_overlap > 0 and src_tokens and cache_tokens:
            overlap = len(src_tokens & cache_tokens) / max(
                min(len(src_tokens), len(cache_tokens)),
                1,
            )
            if overlap < self._min_token_overlap:
                return min_rank

        return score

    def range(self) -> tuple[float, float]:
        return self._base.range()


def _tokenize(text: str) -> set:
    return {token for token in normalize_text(text).split() if token}


def _is_structured_canonical(value: str) -> bool:
    return value.startswith(_STRUCTURED_CANONICAL_PREFIXES)
