from __future__ import annotations

from unittest.mock import patch

import pytest

from byte import Cache, Config
from byte.adapter.adapter import aadapt, adapt
from byte.manager import manager_factory
from byte.processor.pre import get_prompt
from byte.similarity_evaluation import ExactMatchEvaluation
from byte.telemetry import OpenTelemetryConfigError


def _build_cache(tmp_path) -> object:
    cache_obj = Cache()
    cache_obj.init(
        pre_embedding_func=get_prompt,
        embedding_func=lambda value, **_: value,
        data_manager=manager_factory("map", data_dir=str(tmp_path)),
        similarity_evaluation=ExactMatchEvaluation(),
        config=Config(
            telemetry_enabled=True,
            telemetry_attributes={"service.name": "byteai-unit-tests"},
            enable_token_counter=False,
        ),
    )
    return cache_obj


def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> object:
    update_cache_func(llm_data)
    return llm_data


def _install_test_tracer() -> object:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("byteai.tests")
    return patch("byte.telemetry.trace.get_tracer", return_value=tracer), provider, exporter


def test_library_telemetry_binds_report_observer(tmp_path) -> None:
    cache_obj = _build_cache(tmp_path)

    assert cache_obj.telemetry.enabled is True
    assert cache_obj.telemetry.observer in cache_obj.report._observers

    cache_obj.close()

    assert cache_obj.telemetry is None


def test_library_telemetry_emits_stage_spans_for_sync_pipeline(tmp_path) -> object:
    tracer_patch, provider, exporter = _install_test_tracer()

    def llm_handler(**kwargs) -> object:
        return "HELLO"

    with tracer_patch:
        cache_obj = _build_cache(tmp_path)
        try:
            first = adapt(
                llm_handler,
                lambda data: data,
                _update_cache_callback,
                cache_obj=cache_obj,
                prompt="Say hello",
            )
            second = adapt(
                llm_handler,
                lambda data: data,
                _update_cache_callback,
                cache_obj=cache_obj,
                prompt="Say hello",
            )
        finally:
            cache_obj.close()
            provider.shutdown()

    span_names = {span.name for span in exporter.get_finished_spans()}

    assert first == "HELLO"
    assert second == "HELLO"
    assert "byteai.cache.pre_process" in span_names
    assert "byteai.cache.embedding" in span_names
    assert "byteai.cache.search" in span_names
    assert "byteai.cache.llm_request" in span_names
    assert "byteai.cache.save" in span_names
    assert "byteai.cache.get_data" in span_names
    assert "byteai.cache.evaluation" in span_names
    assert cache_obj.report.hint_cache_count == 1


@pytest.mark.asyncio
async def test_library_telemetry_emits_stage_spans_for_async_pipeline(tmp_path) -> object:
    tracer_patch, provider, exporter = _install_test_tracer()

    async def llm_handler(**kwargs) -> object:
        return "ASYNC-HELLO"

    with tracer_patch:
        cache_obj = _build_cache(tmp_path)
        try:
            first = await aadapt(
                llm_handler,
                lambda data: data,
                _update_cache_callback,
                cache_obj=cache_obj,
                prompt="Say hello async",
            )
            second = await aadapt(
                llm_handler,
                lambda data: data,
                _update_cache_callback,
                cache_obj=cache_obj,
                prompt="Say hello async",
            )
        finally:
            cache_obj.close()
            provider.shutdown()

    span_names = {span.name for span in exporter.get_finished_spans()}

    assert first == "ASYNC-HELLO"
    assert second == "ASYNC-HELLO"
    assert "byteai.cache.pre_process" in span_names
    assert "byteai.cache.search" in span_names
    assert "byteai.cache.llm_request" in span_names


def test_library_telemetry_requires_optional_dependency(tmp_path) -> None:
    with patch("byte.telemetry.trace", None), patch("byte.telemetry.metrics", None):
        cache_obj = Cache()
        with pytest.raises(OpenTelemetryConfigError):
            cache_obj.init(
                pre_embedding_func=get_prompt,
                embedding_func=lambda value, **_: value,
                data_manager=manager_factory("map", data_dir=str(tmp_path)),
                similarity_evaluation=ExactMatchEvaluation(),
                config=Config(telemetry_enabled=True, enable_token_counter=False),
            )
