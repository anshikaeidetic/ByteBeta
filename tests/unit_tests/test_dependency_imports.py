"""Dependency-boundary tests for optional integrations and lazy imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import byte.utils as byte_utils
from byte.utils._optional_imports import _missing_library_error, lazy_optional_module

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]


def test_check_library_does_not_auto_install_by_default(monkeypatch) -> None:
    monkeypatch.delenv("BYTE_AUTO_INSTALL_OPTIONAL_DEPS", raising=False)

    with patch("byte.utils._optional_imports.importlib.util.find_spec", return_value=None):
        with pytest.raises(ModuleNotFoundError):
            byte_utils._check_library("missing.module", package="missing-package")


def test_check_library_rejects_hidden_auto_install_even_when_env_requests_it(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_AUTO_INSTALL_OPTIONAL_DEPS", "1")

    with patch("byte.utils._optional_imports.importlib.util.find_spec", return_value=None):
        with pytest.raises(ModuleNotFoundError):
            byte_utils._check_library("missing.module", package="missing-package")


def test_import_ruamel_uses_importable_module_name() -> None:
    with patch(
        "byte.utils._optional_imports.importlib.util.find_spec", return_value=object()
    ) as mock_find:
        byte_utils.import_ruamel()

    mock_find.assert_called_once_with("ruamel.yaml")


def test_import_onnxruntime_reports_generic_package_name() -> None:
    with patch("byte.utils._optional_imports.importlib.util.find_spec", return_value=None):
        with pytest.raises(ModuleNotFoundError, match=r"byteai-cache\[onnx\]"):
            byte_utils.import_onnxruntime()


def test_benchmark_plans_package_does_not_eagerly_import_plan_modules() -> None:
    for name in [
        "byte.benchmarking.plans",
        "byte.benchmarking.plans.deepseek_reasoning_reuse",
        "byte.benchmarking.plans.deepseek_runtime_optimization",
        "byte.benchmarking.plans.openai_mixed_100",
        "byte.benchmarking.plans.openai_mixed_1000",
    ]:
        sys.modules.pop(name, None)

    plans = importlib.import_module("byte.benchmarking.plans")

    assert hasattr(plans, "__getattr__")
    assert "byte.benchmarking.plans.deepseek_reasoning_reuse" not in sys.modules
    assert "byte.benchmarking.plans.deepseek_runtime_optimization" not in sys.modules
    assert "byte.benchmarking.plans.openai_mixed_100" not in sys.modules
    assert "byte.benchmarking.plans.openai_mixed_1000" not in sys.modules


def test_byte_package_keeps_public_reexports_lazy() -> None:
    for name in ["byte", "byte.client", "byte.config", "byte.core"]:
        sys.modules.pop(name, None)

    byte_package = importlib.import_module("byte")

    assert byte_package.PRODUCT_NAME == "Byte"
    assert "byte.client" not in sys.modules
    assert "byte.config" not in sys.modules
    assert "byte.core" not in sys.modules

    assert byte_package.Config is not None
    assert "byte.config" in sys.modules


def test_byte_package_exposes_lazy_submodules() -> None:
    byte_package = importlib.import_module("byte")

    assert byte_package.telemetry is importlib.import_module("byte.telemetry")
    assert byte_package.mcp_gateway is importlib.import_module("byte.mcp_gateway")


def test_importing_optional_onnx_modules_is_lightweight() -> None:
    for name in [
        "byte.embedding.onnx",
        "byte.similarity_evaluation.onnx",
        "onnxruntime",
        "transformers",
        "huggingface_hub",
    ]:
        sys.modules.pop(name, None)

    embedding_module = importlib.import_module("byte.embedding.onnx")
    similarity_module = importlib.import_module("byte.similarity_evaluation.onnx")

    assert embedding_module.Onnx is not None
    assert similarity_module.OnnxModelEvaluation is not None
    assert "onnxruntime" not in sys.modules
    assert "transformers" not in sys.modules
    assert "huggingface_hub" not in sys.modules


def test_onnx_embedding_fails_on_first_use_with_install_guidance(monkeypatch) -> None:
    module = importlib.import_module("byte.embedding.onnx")
    encoder = module.Onnx()
    monkeypatch.setattr(
        module,
        "transformers",
        lazy_optional_module("byte_missing_transformers_module", package="transformers"),
    )

    with pytest.raises(ModuleNotFoundError, match="transformers"):
        _ = encoder.dimension


def test_onnx_similarity_fails_on_first_use_with_install_guidance(monkeypatch) -> None:
    module = importlib.import_module("byte.similarity_evaluation.onnx")
    evaluator = module.OnnxModelEvaluation()
    monkeypatch.setattr(
        module,
        "transformers",
        lazy_optional_module("byte_missing_transformers_module", package="transformers"),
    )

    with pytest.raises(ModuleNotFoundError, match="transformers"):
        evaluator.inference("hello", ["world"])


def test_openrouter_import_is_lazy_until_client_use(monkeypatch) -> None:
    module = importlib.import_module("byte._backends.openrouter")
    monkeypatch.setattr(module, "get_pooled_sync_client", lambda provider, factory, **kwargs: factory(**kwargs))
    monkeypatch.setattr(
        module,
        "_load_openai_clients",
        lambda: (_ for _ in ()).throw(_missing_library_error("openai", "openai")),
    )

    with pytest.raises(ModuleNotFoundError, match="openai"):
        module._get_client(api_key="test")


def test_groq_import_is_lazy_until_client_use(monkeypatch) -> None:
    module = importlib.import_module("byte._backends.groq")
    monkeypatch.setattr(module, "get_pooled_sync_client", lambda provider, factory, **kwargs: factory(**kwargs))
    monkeypatch.setattr(
        module,
        "load_optional_attr",
        lambda module_name, attr_name, package=None: (_ for _ in ()).throw(
            _missing_library_error(module_name, package)
        ),
    )

    with pytest.raises(ModuleNotFoundError, match="groq"):
        module._get_client(api_key="test")


def test_onnx_extra_includes_complete_runtime_stack() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    onnx_deps = pyproject["project"]["optional-dependencies"]["onnx"]

    assert any(dep.startswith("onnxruntime") for dep in onnx_deps)
    assert any(dep.startswith("transformers") for dep in onnx_deps)
    assert any(dep.startswith("huggingface-hub") for dep in onnx_deps)
    assert any(dep.startswith("sentencepiece") for dep in onnx_deps)
