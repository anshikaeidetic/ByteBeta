import importlib
import sys
from typing import Any

from byte.similarity_evaluation.similarity_evaluation import SimilarityEvaluation
from byte.utils.lazy_import import LazyImport

__all__ = [
    "CohereRerankEvaluation",
    "ExactMatchEvaluation",
    "GuardedSimilarityEvaluation",
    "KReciprocalEvaluation",
    "LLMEquivalenceEvaluation",
    "NumpyNormEvaluation",
    "OnnxModelEvaluation",
    "SbertCrossencoderEvaluation",
    "SearchDistanceEvaluation",
    "SequenceMatchEvaluation",
    "SimilarityEvaluation",
    "TimeEvaluation",
    "VCacheEvaluation",
]

onnx = LazyImport("onnx", globals(), "byte.similarity_evaluation.onnx")
numpy_similarity = LazyImport(
    "numpy_similarity", globals(), "byte.similarity_evaluation.numpy_similarity"
)
distance = LazyImport("simple", globals(), "byte.similarity_evaluation.distance")
exact_match = LazyImport("exact_match", globals(), "byte.similarity_evaluation.exact_match")
kreciprocal = LazyImport("kreciprocal", globals(), "byte.similarity_evaluation.kreciprocal")
cohere = LazyImport("cohere", globals(), "byte.similarity_evaluation.cohere_rerank")
sequence_match = LazyImport(
    "sequence_match", globals(), "byte.similarity_evaluation.sequence_match"
)
time = LazyImport("time", globals(), "byte.similarity_evaluation.time")
sbert_crossencoder = LazyImport(
    "sbert_crossencoder", globals(), "byte.similarity_evaluation.sbert_crossencoder"
)
guarded = LazyImport("guarded", globals(), "byte.similarity_evaluation.guarded")
vcache_mod = LazyImport("vcache_mod", globals(), "byte.similarity_evaluation.vcache")
llm_equiv_mod = LazyImport("llm_equiv_mod", globals(), "byte.similarity_evaluation.llm_equivalence")


def OnnxModelEvaluation(model="GPTCache/albert-duplicate-onnx") -> Any:
    return onnx.OnnxModelEvaluation(model)


def NumpyNormEvaluation(enable_normal: bool = False, **kwargs) -> Any:
    return numpy_similarity.NumpyNormEvaluation(enable_normal, **kwargs)


def SearchDistanceEvaluation(max_distance=4.0, positive=False) -> Any:
    return distance.SearchDistanceEvaluation(max_distance, positive)


def ExactMatchEvaluation() -> Any:
    return exact_match.ExactMatchEvaluation()


def KReciprocalEvaluation(vectordb, top_k=3, max_distance=4.0, positive=False) -> Any:
    return kreciprocal.KReciprocalEvaluation(vectordb, top_k, max_distance, positive)


def CohereRerankEvaluation(model: str = "rerank-english-v2.0", api_key: str | None = None) -> Any:
    return cohere.CohereRerank(model=model, api_key=api_key)


def SequenceMatchEvaluation(weights, embedding_extractor, embedding_config: dict[str, Any] | None = None) -> Any:
    return sequence_match.SequenceMatchEvaluation(
        weights,
        embedding_extractor,
        embedding_config=embedding_config,
    )


def TimeEvaluation(evaluation: str, evaluation_config: dict[str, Any], time_range: float = 86400.0) -> Any:
    return time.TimeEvaluation(
        evaluation, evaluation_config=evaluation_config, time_range=time_range
    )


def SbertCrossencoderEvaluation(model: str = "cross-encoder/quora-distilroberta-base") -> Any:
    return sbert_crossencoder.SbertCrossencoderEvaluation(model)


def GuardedSimilarityEvaluation(
    base_evaluation: SimilarityEvaluation,
    min_token_overlap: float = 0.0,
    max_length_ratio: float | None = None,
    enforce_canonical_match: bool = False,
) -> Any:
    return guarded.GuardedSimilarityEvaluation(
        base_evaluation,
        min_token_overlap=min_token_overlap,
        max_length_ratio=max_length_ratio,
        enforce_canonical_match=enforce_canonical_match,
    )


def VCacheEvaluation(
    delta: float = 0.05,
    min_observations: int = 10,
    learning_rate: float = 0.01,
    cold_fallback_threshold: float = 0.80,
    store_path: str | None = None,
) -> Any:
    return vcache_mod.VCacheEvaluation(
        delta=delta,
        min_observations=min_observations,
        learning_rate=learning_rate,
        cold_fallback_threshold=cold_fallback_threshold,
        store_path=store_path,
    )


def LLMEquivalenceEvaluation(
    base: "SimilarityEvaluation",
    ambiguity_band_low: float = 0.70,
    ambiguity_band_high: float = 0.85,
    equivalence_model: str = "",
    provider_key: str | None = None,
) -> Any:
    return llm_equiv_mod.LLMEquivalenceEvaluation(
        base,
        ambiguity_band_low=ambiguity_band_low,
        ambiguity_band_high=ambiguity_band_high,
        equivalence_model=equivalence_model,
        provider_key=provider_key,
    )


sys.modules.setdefault(
    "byte.similarity_evaluation.np",
    importlib.import_module("byte.similarity_evaluation.numpy_similarity"),
)
