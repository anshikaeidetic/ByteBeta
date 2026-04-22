"""Adapter API contract tests for public provider helpers and capability matrix."""

import os
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest
from ruamel.yaml.constructor import ConstructorError

from byte import Cache, Config, cache
from byte._backends import openai
from byte.adapter.api import (
    clear_router_alias_registry,
    clear_router_runtime,
    get,
    init_cache,
    init_exact_cache,
    init_hybrid_cache,
    init_normalized_cache,
    init_safe_semantic_cache,
    init_similar_cache,
    init_similar_cache_from_config,
    provider_capabilities,
    provider_capability_matrix,
    put,
    register_router_alias,
    router_registry_summary,
    supports_capability,
)
from byte.embedding import Onnx as EmbeddingOnnx
from byte.manager import CacheBase, VectorBase, get_data_manager
from byte.processor.post import nop
from byte.processor.pre import get_prompt, last_content, normalized_last_content
from byte.similarity_evaluation import SearchDistanceEvaluation
from byte.utils import import_ruamel
from byte.utils.response import get_message_from_openai_answer

import_ruamel()

from ruamel.yaml import YAML

faiss_file = "faiss.index"


@pytest.fixture(autouse=True)
def reset_openai_adapter_state() -> object:
    openai.ChatCompletion.llm = None
    openai.ChatCompletion.cache_args = {}
    yield
    openai.ChatCompletion.llm = None
    openai.ChatCompletion.cache_args = {}


class MockOnnx:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def dimension(self) -> object:
        return 768

    def to_embeddings(self, data, **_) -> object:
        import numpy as np

        return np.ones(768).astype("float32")


@pytest.mark.requires_feature("sqlalchemy", "faiss")
@patch("byte.embedding.onnx.Onnx", MockOnnx)
def test_byte_api() -> None:
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)

    cache.init(pre_embedding_func=get_prompt)
    put("test_byte_api_hello", "foo")
    assert get("test_byte_api_hello") == "foo"

    inner_cache = Cache()
    init_similar_cache(
        data_dir="./",
        cache_obj=inner_cache,
        post_func=nop,
        config=Config(similarity_threshold=0.1),
    )

    put("api-hello1", "foo1", cache_obj=inner_cache)
    put("api-hello2", "foo2", cache_obj=inner_cache)
    put("api-hello3", "foo3", cache_obj=inner_cache)

    messages = get(
        "hello", cache_obj=inner_cache, top_k=3, hit_callback=lambda x: print("hit_callback", x)
    )
    assert len(messages) == 3
    assert "foo1" in messages
    assert "foo2" in messages
    assert "foo3" in messages

    assert get("api-hello1") is None


@pytest.mark.requires_feature("sqlalchemy", "faiss")
@patch("byte.embedding.onnx.Onnx", MockOnnx)
def test_none_scale_data() -> object:
    if os.path.isfile(faiss_file):
        os.remove(faiss_file)

    def init_cache() -> object:
        embedding_onnx = EmbeddingOnnx()
        cache_base = CacheBase("sqlite")
        vector_base = VectorBase("faiss", dimension=embedding_onnx.dimension, top_k=10)
        data_manager = get_data_manager(cache_base, vector_base)

        evaluation = SearchDistanceEvaluation()
        inner_cache = Cache()
        inner_cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
            post_process_messages_func=nop,
            config=Config(similarity_threshold=0.1),
        )
        return inner_cache

    inner_cache = init_cache()
    put("api-hello1", "foo1", cache_obj=inner_cache)

    with suppress(PermissionError):
        os.remove("sqlite.db")
    inner_cache = init_cache()
    print("hello", get("api-hello1", cache_obj=inner_cache))
    assert get("api-hello1", cache_obj=inner_cache) is None


@pytest.mark.requires_feature("sqlalchemy", "faiss")
@patch("byte.embedding.onnx.Onnx", MockOnnx)
def test_init_with_config(tmp_path) -> None:
    yaml_path = tmp_path / "test.yaml"

    config = {
        "storage_config": {
            "manager": "sqlite,faiss",
            "data_dir": str(tmp_path / "test-config"),
        },
        "model_source": "onnx",
        "evaluation": "distance",
        "pre_function": "get_prompt",
        "post_function": "first",
        "config": {"similarity_threshold": 0.1},
    }

    with open(yaml_path, "w+", encoding="utf-8") as f:
        yaml = YAML()
        yaml.dump(config, f)

    init_similar_cache_from_config(
        config_dir=str(yaml_path.resolve()),
    )

    put("api-hello", "foo")
    assert get("api-hello") == "foo"


def _make_mock_client(mock_response) -> object:
    """Helper to create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_response)
    return mock_client


def test_exact_cache_round_trip(tmp_path) -> None:
    exact_cache = Cache()
    init_exact_cache(
        data_dir=str(tmp_path),
        cache_obj=exact_cache,
        config=Config(enable_token_counter=False),
    )

    put("Alpha  Beta", "exact-hit", cache_obj=exact_cache)

    assert get("Alpha  Beta", cache_obj=exact_cache) == "exact-hit"
    assert get("alpha beta", cache_obj=exact_cache) is None


def test_normalized_cache_reuses_formatting_variants(tmp_path) -> None:
    normalized_cache = Cache()
    init_normalized_cache(
        data_dir=str(tmp_path),
        cache_obj=normalized_cache,
        config=Config(enable_token_counter=False),
    )

    put("  Hello,   WORLD!!  ", "normalized-hit", cache_obj=normalized_cache)

    assert get("hello world", cache_obj=normalized_cache) == "normalized-hit"


def test_normalized_cache_reuses_template_variants(tmp_path) -> None:
    normalized_cache = Cache()
    init_normalized_cache(
        data_dir=str(tmp_path),
        cache_obj=normalized_cache,
        config=Config(enable_token_counter=False),
    )

    put(
        "Reply with exactly TOKYO and nothing else.",
        "TOKYO",
        cache_obj=normalized_cache,
    )

    assert (
        get(
            "Keep the answer to TOKYO. Byte benchmark request. Reply with exactly TOKYO and nothing else.",
            cache_obj=normalized_cache,
        )
        == "TOKYO"
    )


def test_normalized_cache_reuses_labeled_classification_variants(tmp_path) -> None:
    normalized_cache = Cache()
    init_normalized_cache(
        data_dir=str(tmp_path),
        cache_obj=normalized_cache,
        config=Config(enable_token_counter=False),
    )

    put(
        'Classify the sentiment.\nLabels: POSITIVE, NEGATIVE, NEUTRAL\nReview: "I absolutely loved this movie."',
        "POSITIVE",
        cache_obj=normalized_cache,
    )

    assert (
        get(
            'Review: "I absolutely loved this movie."\nLabels: POSITIVE, NEGATIVE, NEUTRAL\nClassify the sentiment and answer with one label only.',
            cache_obj=normalized_cache,
        )
        == "POSITIVE"
    )


def test_init_cache_dispatches_normalized_mode(tmp_path) -> None:
    normalized_cache = Cache()
    init_cache(
        mode="normalized",
        data_dir=str(tmp_path),
        cache_obj=normalized_cache,
        config=Config(enable_token_counter=False),
    )

    put("  Cache,   PLEASE!! ", "generic-hit", cache_obj=normalized_cache)

    assert get("cache please", cache_obj=normalized_cache) == "generic-hit"


def test_provider_capability_matrix_exposes_shared_runtime_and_media_support() -> None:
    matrix = provider_capability_matrix()

    assert matrix["openai"]["speech_generation"]
    assert matrix["openai"]["image_generation"]
    assert matrix["openai"]["vision_inputs"]
    assert matrix["openai"]["execution_verified_memory"]
    assert matrix["openai"]["streaming_memory_recording"]
    assert matrix["openai"]["streaming_verified_reuse"]
    assert matrix["openai"]["distributed_vector_search"]
    assert matrix["openai"]["memory_export_parquet"]
    assert matrix["openai"]["memory_export_sqlite_dump"]
    assert matrix["openai"]["evidence_aware_verification"]
    assert matrix["openai"]["source_context_gap_detection"]
    assert matrix["openai"]["strict_cache_revalidation"]
    assert matrix["openai"]["focus_conditioned_context_distillation"]
    assert matrix["openai"]["global_aux_context_budgeting"]
    assert matrix["openai"]["cross_note_context_deduplication"]
    assert matrix["mistral"]["chat_completion"]
    assert matrix["cohere"]["chat_completion"]
    assert matrix["bedrock"]["chat_completion"]
    assert matrix["llama_cpp"]["chat_completion"]
    assert matrix["openai"]["unified_provider_router"]
    assert matrix["openai"]["fallback_routing"]
    assert matrix["openai"]["semantic_signal_routing"]
    assert matrix["anthropic"]["chat_completion"]
    assert matrix["anthropic"]["document_inputs"]
    assert matrix["anthropic"]["vision_inputs"]
    assert not matrix["anthropic"]["speech_generation"]
    assert matrix["gemini"]["image_generation"]
    assert matrix["gemini"]["speech_generation"]
    assert matrix["gemini"]["audio_transcription"]
    assert matrix["groq"]["audio_transcription"]
    assert matrix["groq"]["speech_generation"]
    assert matrix["openrouter"]["image_generation"]
    assert matrix["ollama"]["vision_inputs"]
    assert matrix["deepseek"]["chat_completion"]
    assert matrix["deepseek"]["coding_tasks"]
    assert matrix["deepseek"]["failure_memory"]
    assert matrix["groq"]["budget_aware_serving"]
    assert matrix["openrouter"]["tenant_safe_global_learning"]
    assert matrix["ollama"]["failure_memory"]


def test_provider_capabilities_and_supports_capability_helpers() -> None:
    openai_caps = provider_capabilities("openai")
    assert openai_caps["provider"] == "openai"
    assert openai_caps["capability_scope"] == "byte_adapter_surface"
    assert supports_capability("openai", "audio_transcription")
    assert supports_capability("deepseek", "coding_tasks")
    assert supports_capability("gemini", "audio_transcription")
    assert provider_capabilities("deep-seek")["provider"] == "deepseek"
    assert provider_capabilities("llama.cpp")["provider"] == "llama_cpp"
    assert supports_capability("llama-cpp", "coding_tasks")
    assert provider_capabilities("unknown-provider") == {}


def test_router_alias_registry_helpers() -> None:
    clear_router_alias_registry()
    clear_router_runtime()

    registered = register_router_alias(
        "fast-chat",
        ["openai/gpt-4o-mini", "gemini/gemini-2.0-flash"],
    )
    summary = router_registry_summary()

    assert registered["alias"] == "fast-chat"
    assert summary["aliases"]["fast-chat"] == [
        "openai/gpt-4o-mini",
        "gemini/gemini-2.0-flash",
    ]


@pytest.mark.requires_feature("sqlalchemy", "faiss")
@patch("byte.embedding.onnx.Onnx", MockOnnx)
def test_init_with_new_config(tmp_path) -> None:
    yaml_path = tmp_path / "test_new.yaml"

    config = {
        "storage_config": {
            "manager": "sqlite,faiss",
            "data_dir": str(tmp_path / "test-new-config"),
        },
        "embedding": "onnx",
        "embedding_config": {"model": "sentence-transformers/paraphrase-albert-small-v2"},
        "evaluation": "distance",
        "evaluation_config": {
            "max_distance": 4.0,
            "positive": False,
        },
        "pre_context_function": "concat",
        "post_function": "first",
    }

    with open(yaml_path, "w+", encoding="utf-8") as f:
        yaml = YAML()
        yaml.dump(config, f)

    init_similar_cache_from_config(
        config_dir=str(yaml_path.resolve()),
    )

    question = "calculate 1+3"
    expect_answer = "the result is 4"

    datas = {
        "choices": [
            {
                "message": {"content": expect_answer, "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": 1677825464,
        "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion.chunk",
    }
    mock_client = _make_mock_client(datas)

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
        )

        assert get_message_from_openai_answer(response) == expect_answer, response

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    answer_text = get_message_from_openai_answer(response)
    assert answer_text == expect_answer, answer_text


def test_init_with_config_rejects_unsafe_yaml(tmp_path) -> None:
    yaml_path = tmp_path / "unsafe.yaml"
    yaml_path.write_text(
        "config:\n  prompts: !!python/object/apply:os.system ['echo unsafe']\n",
        encoding="utf-8",
    )

    with pytest.raises(ConstructorError):
        init_similar_cache_from_config(config_dir=str(yaml_path))


@pytest.mark.requires_feature("onnx", "sqlalchemy", "faiss")
def test_hybrid_cache_hits_normalized_prompt_before_upstream(tmp_path) -> None:
    hybrid_cache = Cache()
    init_hybrid_cache(
        data_dir=str(tmp_path),
        cache_obj=hybrid_cache,
        pre_func=last_content,
        normalized_pre_func=normalized_last_content,
        config=Config(enable_token_counter=False),
    )

    expect_answer = "normalized cache hit"
    datas = {
        "choices": [
            {
                "message": {"content": expect_answer, "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": 1677825464,
        "id": "chatcmpl-hybrid-hit",
        "model": "gpt-4o-mini",
        "object": "chat.completion",
    }
    mock_client = _make_mock_client(datas)

    with patch("byte.adapter.openai._get_client", return_value=mock_client):
        first = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "  Hello,   WORLD!!  "}],
            cache_obj=hybrid_cache,
        )
        second = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello world"}],
            cache_obj=hybrid_cache,
        )

    assert get_message_from_openai_answer(first) == expect_answer
    assert get_message_from_openai_answer(second) == expect_answer
    assert second["byte"] is True
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.requires_feature("onnx", "sqlalchemy", "faiss")
def test_hybrid_cache_warm_seeds_normalized_and_semantic_chain(tmp_path) -> None:
    hybrid_cache = Cache()
    init_hybrid_cache(
        data_dir=str(tmp_path),
        cache_obj=hybrid_cache,
        config=Config(enable_token_counter=False),
    )

    result = hybrid_cache.warm(
        [
            {
                "question": "Reply with exactly TOKYO and nothing else.",
                "answer": "TOKYO",
            }
        ]
    )

    assert result["seeded"] == 1
    assert result["cache_writes"] >= 3
    assert (
        get(
            "Keep the answer to TOKYO. Reply with exactly TOKYO and nothing else.",
            cache_obj=hybrid_cache,
        )
        == "TOKYO"
    )


@pytest.mark.requires_feature("onnx", "sqlalchemy", "faiss")
def test_init_cache_can_prewarm_hot_prompts(tmp_path) -> None:
    hybrid_cache = Cache()
    init_cache(
        mode="hybrid",
        data_dir=str(tmp_path),
        cache_obj=hybrid_cache,
        config=Config(enable_token_counter=False),
        warm_data=[
            {
                "question": "Reply with exactly BYTE_WARM and nothing else.",
                "answer": "BYTE_WARM",
            }
        ],
    )

    assert (
        get("Reply with exactly BYTE_WARM and nothing else.", cache_obj=hybrid_cache) == "BYTE_WARM"
    )


@pytest.mark.requires_feature("onnx", "sqlalchemy", "faiss")
def test_init_safe_semantic_cache_raises_default_guard_rails(tmp_path) -> None:
    semantic_cache = Cache()
    init_safe_semantic_cache(
        data_dir=str(tmp_path),
        cache_obj=semantic_cache,
        config=Config(
            enable_token_counter=False,
            similarity_threshold=0.2,
            semantic_min_token_overlap=0.0,
            semantic_max_length_ratio=4.0,
            cache_admission_min_score=0.0,
        ),
    )

    assert semantic_cache.config.similarity_threshold == 0.85
    assert semantic_cache.config.semantic_min_token_overlap == 0.30
    assert semantic_cache.config.semantic_max_length_ratio == 3.0
    assert semantic_cache.config.semantic_enforce_canonical_match is True
    assert semantic_cache.config.cache_admission_min_score == 0.65
