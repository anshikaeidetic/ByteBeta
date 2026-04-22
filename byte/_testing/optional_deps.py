"""Shared optional-dependency registry for Byte tests and repo validation."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from byte._optional_features import OPTIONAL_FEATURES, feature_spec, normalize_feature_name
from byte._optional_features import feature_available as _feature_available

FeatureModules = tuple[tuple[str, ...], ...]
FeatureRequirements = tuple[str, ...]

FEATURE_MODULES: dict[str, FeatureModules] = {
    name: feature_spec(name).module_groups for name in OPTIONAL_FEATURES
}

FEATURE_TEST_REQUIREMENTS: dict[str, FeatureRequirements] = {
    "tests/unit_tests/adapter/test_adapter.py::test_cache_temperature": (
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_adapter.py::test_input_summarization": ("transformers",),
    "tests/unit_tests/adapter/test_api.py::test_byte_api": ("sqlalchemy", "faiss"),
    "tests/unit_tests/adapter/test_api.py::test_none_scale_data": (
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_api.py::test_init_with_config": (
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_api.py::test_init_with_new_config": (
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_api.py::test_hybrid_cache_hits_normalized_prompt_before_upstream": (
        "onnx",
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_api.py::test_hybrid_cache_warm_seeds_normalized_and_semantic_chain": (
        "onnx",
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_api.py::test_init_cache_can_prewarm_hot_prompts": (
        "onnx",
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_api.py::test_init_safe_semantic_cache_raises_default_guard_rails": (
        "onnx",
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_openai.py": ("openai", "pillow"),
    "tests/unit_tests/adapter/test_groq_openrouter.py": ("groq",),
    "tests/unit_tests/adapter/test_langchain_models.py": ("langchain",),
    "tests/unit_tests/adapter/test_llama_cpp.py": ("llama_cpp",),
    "tests/unit_tests/adapter/test_provider_agnostic_coding_workloads.py": ("groq",),
    "tests/unit_tests/adapter/test_provider_agnostic_reasoning_reuse.py": (
        "groq",
        "onnx",
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/adapter/test_provider_native_prompt_cache_bridge.py": ("groq",),
    "tests/unit_tests/embedding/test_embedding_openai.py": ("openai",),
    "tests/unit_tests/embedding/test_cohere.py": ("cohere",),
    "tests/unit_tests/embedding/test_data2vec.py": ("transformers",),
    "tests/unit_tests/embedding/test_huggingface.py": ("transformers",),
    "tests/unit_tests/embedding/test_langchain.py": ("langchain",),
    "tests/unit_tests/embedding/test_onnx.py": ("onnx",),
    "tests/unit_tests/embedding/test_paddlenlp.py": ("paddle",),
    "tests/unit_tests/embedding/test_rwkv.py": ("transformers",),
    "tests/unit_tests/embedding/test_sbert.py": ("sbert",),
    "tests/unit_tests/embedding/test_timm.py": ("timm", "pillow"),
    "tests/unit_tests/embedding/test_uform.py": ("uform", "pillow"),
    "tests/unit_tests/embedding/test_vit.py": ("vit", "pillow"),
    "tests/unit_tests/eviction/test_distributed_cache.py": ("redis",),
    "tests/unit_tests/benchmarking/test_quickstart.py::test_run_local_comparison_runs_without_provider_key": (
        "onnx",
        "sqlalchemy",
        "faiss",
    ),
    "tests/unit_tests/manager/test_dynamo_storage.py": ("dynamo",),
    "tests/unit_tests/manager/test_eviction.py": ("sqlalchemy", "faiss"),
    "tests/unit_tests/manager/test_factory.py": ("sqlalchemy", "faiss"),
    "tests/unit_tests/manager/test_mongo.py": ("mongo",),
    "tests/unit_tests/manager/test_object_storage.py": ("dynamo",),
    "tests/unit_tests/manager/test_pgvector.py": ("pgvector",),
    "tests/unit_tests/manager/test_qdrant.py": ("qdrant",),
    "tests/unit_tests/manager/test_redis.py": ("redis",),
    "tests/unit_tests/manager/test_redis_cache_storage.py": ("redis",),
    "tests/unit_tests/manager/test_sql_scalar.py": ("sqlalchemy",),
    "tests/unit_tests/manager/test_usearch.py": ("usearch",),
    "tests/unit_tests/processor/test_summarize_context.py": ("transformers",),
    "tests/unit_tests/similarity_evaluation/test_cohere_rerank.py": ("cohere",),
    "tests/unit_tests/similarity_evaluation/test_evaluation_kreciprocal.py": ("faiss",),
    "tests/unit_tests/similarity_evaluation/test_evaluation_onnx.py": ("onnx",),
    "tests/unit_tests/similarity_evaluation/test_evaluation_sequence.py::test_sequence_match": (
        "onnx",
    ),
    "tests/unit_tests/similarity_evaluation/test_evaluation_sequence.py::test_get_eval": (
        "onnx",
    ),
    "tests/unit_tests/similarity_evaluation/test_evaluation_sbert.py": ("sbert",),
    "tests/unit_tests/test_library_telemetry.py": ("telemetry",),
    "tests/unit_tests/test_session.py": ("onnx", "openai"),
    "tests/unit_tests/utils/test_error.py": ("openai",),
    "tests/integration_tests/test_redis_onnx.py": ("onnx",),
    "tests/integration_tests/test_sqlite_faiss_onnx.py": ("onnx",),
    "tests/integration_tests/test_sqlite_milvus_sbert.py": ("milvus_sbert",),
    "tests/integration_tests/examples/sqlite_faiss_onnx/test_example_sqlite_faiss_onnx.py": (
        "onnx",
    ),
}

# Compatibility alias retained for older callers that expect a single feature per path.
FEATURE_TEST_PATHS: dict[str, str] = {
    path: requirements[0]
    for path, requirements in FEATURE_TEST_REQUIREMENTS.items()
    if len(requirements) == 1
}


def _normalize_features(features: Iterable[str]) -> tuple[str, ...]:
    return tuple(normalize_feature_name(feature) for feature in features)


def feature_available(feature: str) -> bool:
    return _feature_available(feature)


def missing_features(*features: str) -> tuple[str, ...]:
    normalized = _normalize_features(features)
    return tuple(feature for feature in normalized if not feature_available(feature))


def skip_module_if_feature_missing(*features: str) -> None:
    missing = missing_features(*features)
    if not missing:
        return
    joined = ", ".join(missing)
    pytest.skip(f"{joined} optional feature stack is not installed", allow_module_level=True)


def required_features_for_path(path: str) -> FeatureRequirements:
    return FEATURE_TEST_REQUIREMENTS.get(path, ())


def feature_test_targets(*features: str, prefixes: tuple[str, ...] = ("tests/unit_tests/",)) -> list[str]:
    requested = set(_normalize_features(features))
    if not requested:
        return []

    targets: list[str] = []
    for path, requirements in FEATURE_TEST_REQUIREMENTS.items():
        normalized_requirements = set(_normalize_features(requirements))
        if normalized_requirements.issubset(requested) and any(
            path.startswith(prefix) for prefix in prefixes
        ):
            targets.append(path)
    return sorted(targets)


__all__ = [
    "FEATURE_MODULES",
    "FEATURE_TEST_PATHS",
    "FEATURE_TEST_REQUIREMENTS",
    "feature_available",
    "feature_test_targets",
    "missing_features",
    "required_features_for_path",
    "skip_module_if_feature_missing",
]
