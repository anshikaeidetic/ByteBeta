"""Regression tests for shared optional-feature metadata and test targeting."""

from __future__ import annotations

from pathlib import Path

import byte._optional_features as _optional_features
from byte._testing import optional_deps as _optional_deps

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_INTEGRATION_TESTS = {
    "tests/integration_tests/test_redis_onnx.py",
    "tests/integration_tests/test_sqlite_faiss_onnx.py",
    "tests/integration_tests/test_sqlite_milvus_sbert.py",
}


def test_optional_integration_paths_are_registered() -> None:
    missing = OPTIONAL_INTEGRATION_TESTS.difference(_optional_deps.FEATURE_TEST_REQUIREMENTS)
    assert not missing
    assert all((ROOT / path).is_file() for path in OPTIONAL_INTEGRATION_TESTS)


def test_feature_test_targets_require_all_declared_features() -> None:
    openai_targets = _optional_deps.feature_test_targets("openai")

    assert "tests/unit_tests/embedding/test_embedding_openai.py" in openai_targets
    assert "tests/unit_tests/test_session.py" not in openai_targets
    assert "tests/unit_tests/adapter/test_openai.py" not in openai_targets


def test_feature_test_targets_include_combined_feature_requirements() -> None:
    openai_targets = _optional_deps.feature_test_targets("openai", "pillow")

    assert "tests/unit_tests/adapter/test_openai.py" in openai_targets


def test_provider_agnostic_reasoning_reuse_requires_semantic_cache_stack() -> None:
    groq_only = _optional_deps.feature_test_targets("groq")
    full_stack = _optional_deps.feature_test_targets(
        "groq", "onnx", "sqlalchemy", "faiss"
    )

    target = "tests/unit_tests/adapter/test_provider_agnostic_reasoning_reuse.py"
    assert target not in groq_only
    assert target in full_stack


def test_registry_reuses_production_feature_metadata() -> None:
    assert _optional_deps.FEATURE_MODULES["sqlalchemy"] == _optional_features.feature_spec(
        "sqlalchemy"
    ).module_groups
    assert _optional_deps.feature_available("sql") == _optional_deps.feature_available(
        "sqlalchemy"
    )


def test_milvus_sbert_feature_accepts_either_supported_backend(monkeypatch) -> None:
    available = {"sentence_transformers", "chromadb"}

    monkeypatch.setattr(
        _optional_features,
        "_module_available",
        lambda module_name: module_name in available,
    )

    assert _optional_deps.feature_available("milvus_sbert") is True

    available = {"sentence_transformers", "pymilvus"}
    assert _optional_deps.feature_available("milvus_sbert") is True

    available = {"sentence_transformers"}
    assert _optional_deps.feature_available("milvus_sbert") is False
