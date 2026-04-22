"""Regression tests for core hotspot facades and lazy optional imports."""

from __future__ import annotations

import importlib
import sys

from byte import Cache


def _remove_modules(*prefixes: str) -> None:
    for name in tuple(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def test_core_hotspot_facades_preserve_public_imports() -> None:
    gemini = importlib.import_module("byte._backends.gemini")
    optimization_memory = importlib.import_module("byte.processor.optimization_memory")
    model_router = importlib.import_module("byte.processor.model_router")
    h2o_runtime = importlib.import_module("byte.h2o._runtime_engine")
    response_finalize = importlib.import_module("byte.adapter.pipeline._response_finalize")

    assert gemini.ChatCompletion is not None
    assert gemini.Audio is not None
    assert gemini.Speech is not None
    assert gemini.Image is not None
    assert optimization_memory.PromptPieceStore is not None
    assert optimization_memory.ArtifactMemoryStore is not None
    assert optimization_memory.WorkflowPlanStore is not None
    assert optimization_memory.SessionDeltaStore is not None
    assert model_router.ModelRouteDecision is not None
    assert model_router.route_request_model is not None
    assert h2o_runtime.H2ORuntime is not None
    assert h2o_runtime.get_huggingface_runtime is not None
    assert response_finalize.finalize_sync_llm_response is not None
    assert response_finalize.finalize_async_llm_response is not None


def test_cache_memory_facade_preserves_feature_methods() -> None:
    cache_obj = Cache()

    assert hasattr(cache_obj, "remember_tool_result")
    assert hasattr(cache_obj, "remember_interaction")
    assert hasattr(cache_obj, "remember_execution_result")
    assert hasattr(cache_obj, "remember_failure_pattern")
    assert hasattr(cache_obj, "remember_artifact")
    assert hasattr(cache_obj, "remember_reasoning_result")
    assert hasattr(cache_obj, "export_memory_snapshot")


def test_gemini_and_h2o_facades_do_not_import_hosted_or_model_stacks() -> None:
    _remove_modules(
        "byte._backends.gemini",
        "byte._backends.gemini_audio",
        "byte._backends.gemini_chat",
        "byte._backends.gemini_clients",
        "byte._backends.gemini_images",
        "byte._backends.gemini_messages",
        "byte._backends.gemini_responses",
        "byte.h2o._runtime_engine",
        "byte.h2o._runtime_factory",
        "byte.h2o._runtime_runtime",
        "byte.h2o._runtime_generation",
        "byte.h2o._runtime_decode",
        "byte.h2o._runtime_tokens",
        "google",
        "torch",
        "transformers",
    )

    importlib.import_module("byte._backends.gemini")
    importlib.import_module("byte.h2o._runtime_engine")

    assert "google.genai" not in sys.modules
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
