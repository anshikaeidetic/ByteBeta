import os
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import byte.processor.post
import byte.processor.pre
from byte import Cache, Config, cache
from byte.embedding import (
    SBERT,
    Cohere,
    Data2VecAudio,
    FastText,
    Huggingface,
    Onnx,
    OpenAI,
    PaddleNLP,
    Rwkv,
    Timm,
    UForm,
    ViT,
)
from byte.embedding.base import BaseEmbedding
from byte.manager import manager_factory
from byte.manager.data_manager import DataManager
from byte.manager.tiered_cache import TieredCacheManager
from byte.processor.context import (
    ConcatContextProcess,
    SelectiveContextProcess,
    SummarizationContextProcess,
)
from byte.processor.post import temperature_softmax
from byte.processor.pre import get_prompt
from byte.similarity_evaluation import (
    CohereRerankEvaluation,
    ExactMatchEvaluation,
    GuardedSimilarityEvaluation,
    KReciprocalEvaluation,
    NumpyNormEvaluation,
    OnnxModelEvaluation,
    SbertCrossencoderEvaluation,
    SearchDistanceEvaluation,
    SequenceMatchEvaluation,
    SimilarityEvaluation,
    TimeEvaluation,
)
from byte.utils import import_ruamel


def init_similar_cache(
    data_dir: str = "api_cache",
    cache_obj: Cache | None = None,
    pre_func: Callable = get_prompt,
    embedding: BaseEmbedding | None = None,
    data_manager: DataManager | None = None,
    evaluation: SimilarityEvaluation | None = None,
    post_func: Callable = temperature_softmax,
    config: Config | None = None,
    warm_data: Any | None = None,
    **kwargs,
) -> Any:
    """Provide a quick way to initialize cache for api service

    :param data_dir: cache data storage directory
    :type data_dir: str
    :param cache_obj: specify to initialize the Cache object, if not specified, initialize the global object
    :type cache_obj: Optional[Cache]
    :param pre_func: pre-processing of the cache input text
    :type pre_func: Callable
    :param embedding: embedding object
    :type embedding: BaseEmbedding
    :param data_manager: data manager object
    :type data_manager: DataManager
    :param evaluation: similarity evaluation object
    :type evaluation: SimilarityEvaluation
    :param post_func: post-processing of the cached result list, the most similar result is taken by default
    :type post_func: Callable[[List[Any]], Any]
    :param config: cache configuration, the core is similar threshold
    :type config: Config
    :return: None

    Example:
        .. code-block:: python

            from byte.adapter.api import put, get, init_similar_cache

            init_similar_cache()
            put("hello", "foo")
            print(get("hello"))
    """
    if not embedding:
        embedding = Onnx()
    if not data_manager:
        data_manager = manager_factory(
            "sqlite,faiss",
            data_dir=data_dir,
            vector_params={"dimension": embedding.dimension},
        )
    data_manager = _wrap_data_manager(data_manager, config)
    if not evaluation:
        evaluation = SearchDistanceEvaluation()
    cache_obj = cache_obj if cache_obj else cache
    cache_obj.init(
        pre_embedding_func=pre_func,
        embedding_func=embedding.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluation,
        post_process_messages_func=post_func,
        config=config,
    )
    _set_cache_stage_metadata(cache_obj, stage="semantic", pipeline="semantic")
    return _warm_cache_if_needed(cache_obj, warm_data)


def _clone_config(config: Config | None) -> Config:
    return deepcopy(config) if config is not None else Config()


def _wrap_data_manager(data_manager: DataManager, config: Config | None) -> DataManager:
    if not config or not getattr(config, "tiered_cache", False):
        return data_manager
    if isinstance(data_manager, TieredCacheManager):
        return data_manager
    return TieredCacheManager(
        data_manager,
        tier1_max_size=config.tier1_max_size,
        promotion_threshold=config.tier1_promotion_threshold,
        promotion_window_s=config.tier1_promotion_window_s,
        promote_on_write=config.tier1_promote_on_write,
        async_write_back=config.async_write_back,
        async_write_back_queue_size=config.async_write_back_queue_size,
    )


def _warm_cache_if_needed(cache_obj: Cache, warm_data: Any | None) -> Cache:
    if warm_data:
        cache_obj.warm(warm_data)
    return cache_obj


def _set_cache_stage_metadata(cache_obj: Cache, *, stage: str, pipeline: str) -> Cache:
    cache_obj.byte_cache_stage = str(stage or "").strip().lower()
    cache_obj.byte_cache_pipeline = str(pipeline or "").strip().lower()
    return cache_obj


def _safe_semantic_config(config: Config | None) -> Config:
    cfg = _clone_config(config)
    # Threshold 0.85: balances hit rate (more cache reuse) vs. false-positives.
    # Research on semantic caching (GPTCache, CacheBlend) shows 0.85-0.90 is the
    # sweet spot for general workloads — above 0.90 misses too many rephrased queries,
    # below 0.80 risks returning wrong answers for different questions.
    if cfg.similarity_threshold < 0.85:
        cfg.similarity_threshold = 0.85
    # Token overlap 0.30 allows synonyms / minor reformulations to hit cache.
    if cfg.semantic_min_token_overlap <= 0:
        cfg.semantic_min_token_overlap = 0.30
    if cfg.semantic_max_length_ratio is None or cfg.semantic_max_length_ratio > 3.0:
        cfg.semantic_max_length_ratio = 3.0
    cfg.semantic_enforce_canonical_match = True
    # Admission floor: responses with quality score below 0.65 are not cached,
    # which prevents empty / garbage LLM outputs from polluting the cache.
    if cfg.cache_admission_min_score <= 0:
        cfg.cache_admission_min_score = 0.65
    cfg.tiered_cache = True
    # Async write-back: cache writes happen in background — LLM responses are
    # returned to the client immediately without blocking on disk writes.
    try:
        cfg.async_write_back = True
    except Exception:
        pass
    if not cfg.semantic_allowed_categories:
        # Include all category types emitted by the intent extractor so that
        # code, explanation, and chat requests are cached semantically.
        cfg.semantic_allowed_categories = [
            "question_answer",
            "summarization",
            "comparison",
            "instruction",
            "code_fix",
            "code_explanation",
            "code_refactor",
            "test_generation",
            "documentation",
            "translation",
            "creative",
            "chat",
            "general",
        ]
    return cfg


def _normalized_pre_func(pre_func: Callable) -> Callable:
    if pre_func is byte.processor.pre.get_prompt:
        return byte.processor.pre.normalized_get_prompt
    if pre_func is byte.processor.pre.last_content:
        return byte.processor.pre.normalized_last_content

    def _wrapped(data, **kwargs) -> Any:
        return byte.processor.pre.canonicalize_text(pre_func(data, **kwargs))

    _wrapped.__name__ = f"normalized_{getattr(pre_func, '__name__', 'pre_func')}"
    return _wrapped


def init_safe_semantic_cache(
    data_dir: str = "semantic_cache",
    cache_obj: Cache | None = None,
    pre_func: Callable = get_prompt,
    embedding: BaseEmbedding | None = None,
    data_manager: DataManager | None = None,
    evaluation: SimilarityEvaluation | None = None,
    post_func: Callable = temperature_softmax,
    config: Config | None = None,
    warm_data: Any | None = None,
    **kwargs,
) -> Any:
    """Initialize a semantic cache with stricter false-positive guard rails.

    This helper intentionally prefers lower hit rate over unsafe reuse by
    tightening similarity thresholds, token overlap, answer-admission quality,
    and canonical matching defaults.
    """
    safe_config = _safe_semantic_config(config)
    guarded_evaluation = GuardedSimilarityEvaluation(
        evaluation or SearchDistanceEvaluation(),
        min_token_overlap=safe_config.semantic_min_token_overlap,
        max_length_ratio=safe_config.semantic_max_length_ratio,
        enforce_canonical_match=safe_config.semantic_enforce_canonical_match,
    )
    return init_similar_cache(
        data_dir=data_dir,
        cache_obj=cache_obj,
        pre_func=pre_func,
        embedding=embedding,
        data_manager=data_manager,
        evaluation=guarded_evaluation,
        post_func=post_func,
        config=safe_config,
        warm_data=warm_data,
    )


def init_exact_cache(
    data_dir: str = "exact_cache",
    cache_obj: Cache | None = None,
    pre_func: Callable = get_prompt,
    post_func: Callable = temperature_softmax,
    config: Config | None = None,
    next_cache: Cache | None = None,
    warm_data: Any | None = None,
    **kwargs,
) -> Any:
    """Initialize a lightweight exact-match cache."""
    cache_obj = cache_obj if cache_obj else cache
    cache_obj.init(
        pre_embedding_func=pre_func,
        embedding_func=lambda data, **_: data,
        data_manager=manager_factory("map", data_dir=data_dir),
        similarity_evaluation=ExactMatchEvaluation(),
        post_process_messages_func=post_func,
        config=_clone_config(config),
        next_cache=next_cache,
    )
    _set_cache_stage_metadata(
        cache_obj,
        stage="exact",
        pipeline="hybrid" if next_cache else "exact",
    )
    return _warm_cache_if_needed(cache_obj, warm_data)


def init_normalized_cache(
    data_dir: str = "normalized_cache",
    cache_obj: Cache | None = None,
    pre_func: Callable = get_prompt,
    normalized_pre_func: Callable | None = None,
    post_func: Callable = temperature_softmax,
    config: Config | None = None,
    next_cache: Cache | None = None,
    warm_data: Any | None = None,
    **kwargs,
) -> Any:
    """Initialize an exact-match cache on normalized request text."""
    cache_obj = cache_obj if cache_obj else cache
    cache_obj.init(
        pre_embedding_func=normalized_pre_func or _normalized_pre_func(pre_func),
        embedding_func=lambda data, **_: data,
        data_manager=manager_factory("map", data_dir=data_dir),
        similarity_evaluation=ExactMatchEvaluation(),
        post_process_messages_func=post_func,
        config=_clone_config(config),
        next_cache=next_cache,
    )
    _set_cache_stage_metadata(
        cache_obj,
        stage="normalized",
        pipeline="hybrid" if next_cache else "normalized",
    )
    return _warm_cache_if_needed(cache_obj, warm_data)


def init_cache(
    mode: str = "normalized",
    data_dir: str = "byte_cache",
    cache_obj: Cache | None = None,
    pre_func: Callable = get_prompt,
    normalized_pre_func: Callable | None = None,
    embedding: BaseEmbedding | None = None,
    data_manager: DataManager | None = None,
    evaluation: SimilarityEvaluation | None = None,
    post_func: Callable = temperature_softmax,
    config: Config | None = None,
    exact_config: Config | None = None,
    normalized_config: Config | None = None,
    semantic_config: Config | None = None,
    warm_data: Any | None = None,
    **kwargs,
) -> Any:
    """Initialize ByteAI Cache with a provider-agnostic cache strategy.

    This helper is shared by all adapters because the strategy lives in the
    core cache layer, not in any single provider integration.
    """
    mode = (mode or "normalized").lower()
    if mode == "exact":
        return init_exact_cache(
            data_dir=data_dir,
            cache_obj=cache_obj,
            pre_func=pre_func,
            post_func=post_func,
            config=config,
            warm_data=warm_data,
        )
    if mode == "normalized":
        return init_normalized_cache(
            data_dir=data_dir,
            cache_obj=cache_obj,
            pre_func=pre_func,
            normalized_pre_func=normalized_pre_func,
            post_func=post_func,
            config=config,
            warm_data=warm_data,
        )
    if mode == "semantic":
        return init_safe_semantic_cache(
            data_dir=data_dir,
            cache_obj=cache_obj,
            pre_func=pre_func,
            embedding=embedding,
            data_manager=data_manager,
            evaluation=evaluation,
            post_func=post_func,
            config=_clone_config(semantic_config or config),
            warm_data=warm_data,
        )
    if mode == "hybrid":
        return init_hybrid_cache(
            data_dir=data_dir,
            cache_obj=cache_obj,
            pre_func=pre_func,
            normalized_pre_func=normalized_pre_func,
            embedding=embedding,
            data_manager=data_manager,
            evaluation=evaluation,
            post_func=post_func,
            config=config,
            exact_config=exact_config,
            normalized_config=normalized_config,
            semantic_config=semantic_config,
            warm_data=warm_data,
        )
    raise ValueError(
        f"Unsupported cache mode: {mode}. Choose from exact, normalized, semantic, hybrid."
    )


def init_hybrid_cache(
    data_dir: str = "hybrid_cache",
    cache_obj: Cache | None = None,
    pre_func: Callable = get_prompt,
    normalized_pre_func: Callable | None = None,
    embedding: BaseEmbedding | None = None,
    data_manager: DataManager | None = None,
    evaluation: SimilarityEvaluation | None = None,
    post_func: Callable = temperature_softmax,
    config: Config | None = None,
    exact_config: Config | None = None,
    normalized_config: Config | None = None,
    semantic_config: Config | None = None,
    warm_data: Any | None = None,
    **kwargs,
) -> Any:
    """Build a layered cache pipeline: exact -> normalized -> semantic.

    This improves hit rate without forcing users to choose between
    cheap exact matches and broader semantic reuse.
    """
    exact_cache = cache_obj if cache_obj else cache
    semantic_cache = Cache()
    normalized_cache = Cache()

    semantic_dir = os.path.join(data_dir, "semantic")
    normalized_dir = os.path.join(data_dir, "normalized")
    exact_dir = os.path.join(data_dir, "exact")

    init_safe_semantic_cache(
        data_dir=semantic_dir,
        cache_obj=semantic_cache,
        pre_func=pre_func,
        embedding=embedding,
        data_manager=data_manager,
        evaluation=evaluation,
        post_func=post_func,
        config=_clone_config(semantic_config or config),
    )
    _set_cache_stage_metadata(semantic_cache, stage="semantic", pipeline="hybrid")

    init_normalized_cache(
        data_dir=normalized_dir,
        cache_obj=normalized_cache,
        pre_func=pre_func,
        normalized_pre_func=normalized_pre_func,
        post_func=post_func,
        config=_clone_config(normalized_config or config),
        next_cache=semantic_cache,
    )
    _set_cache_stage_metadata(normalized_cache, stage="normalized", pipeline="hybrid")

    init_exact_cache(
        data_dir=exact_dir,
        cache_obj=exact_cache,
        pre_func=pre_func,
        post_func=post_func,
        config=_clone_config(exact_config or config),
        next_cache=normalized_cache,
    )
    _set_cache_stage_metadata(exact_cache, stage="exact", pipeline="hybrid")

    return _warm_cache_if_needed(exact_cache, warm_data)


def init_similar_cache_from_config(config_dir: str, cache_obj: Cache | None = None) -> Any:
    import_ruamel()
    from ruamel.yaml import YAML

    if config_dir:
        with open(config_dir, encoding="utf-8") as f:
            yaml = YAML(typ="safe", pure=True)
            init_conf = yaml.load(f)
    else:
        init_conf = {}

    # Accept the older config key when present.
    embedding = init_conf.get("model_source", "")
    if not embedding:
        embedding = init_conf.get("embedding", "onnx")
    # Accept the older config key when present.
    embedding_config = init_conf.get("model_config", {})
    if not embedding_config:
        embedding_config = init_conf.get("embedding_config", {})
    embedding_model = _get_model(embedding, embedding_config)

    storage_config = init_conf.get("storage_config", {})
    storage_config.setdefault("manager", "sqlite,faiss")
    storage_config.setdefault("data_dir", "byte_data")
    storage_config.setdefault("vector_params", {})
    storage_config["vector_params"] = storage_config["vector_params"] or {}
    storage_config["vector_params"]["dimension"] = embedding_model.dimension
    data_manager = manager_factory(**storage_config)

    eval_strategy = init_conf.get("evaluation", "distance")
    # Accept the older config key when present.
    eval_config = init_conf.get("evaluation_kws", {})
    if not eval_config:
        eval_config = init_conf.get("evaluation_config", {})
    evaluation = _get_eval(eval_strategy, eval_config)

    cache_obj = cache_obj if cache_obj else cache

    pre_process = init_conf.get("pre_context_function")
    if pre_process:
        pre_func = _get_pre_context_function(pre_process, init_conf.get("pre_context_config"))
        pre_func = pre_func.pre_process
    else:
        pre_process = init_conf.get("pre_function", "get_prompt")
        pre_func = _get_pre_func(pre_process)

    post_process = init_conf.get("post_function", "first")
    post_func = _get_post_func(post_process)

    config_kws = init_conf.get("config", {}) or {}
    config = Config(**config_kws)
    data_manager = _wrap_data_manager(data_manager, config)
    if config.semantic_min_token_overlap > 0 or config.semantic_enforce_canonical_match:
        evaluation = GuardedSimilarityEvaluation(
            evaluation,
            min_token_overlap=config.semantic_min_token_overlap,
            max_length_ratio=config.semantic_max_length_ratio,
            enforce_canonical_match=config.semantic_enforce_canonical_match,
        )

    cache_obj.init(
        pre_embedding_func=pre_func,
        embedding_func=embedding_model.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluation,
        post_process_messages_func=post_func,
        config=config,
    )
    _set_cache_stage_metadata(cache_obj, stage="semantic", pipeline="semantic")

    return init_conf


def _get_model(model_src, model_config=None) -> Any:
    model_src = model_src.lower()
    model_config = model_config or {}

    if model_src == "onnx":
        return Onnx(**model_config)
    if model_src == "huggingface":
        return Huggingface(**model_config)
    if model_src == "sbert":
        return SBERT(**model_config)
    if model_src == "fasttext":
        return FastText(**model_config)
    if model_src == "data2vecaudio":
        return Data2VecAudio(**model_config)
    if model_src == "timm":
        return Timm(**model_config)
    if model_src == "vit":
        return ViT(**model_config)
    if model_src == "openai":
        return OpenAI(**model_config)
    if model_src == "cohere":
        return Cohere(**model_config)
    if model_src == "rwkv":
        return Rwkv(**model_config)
    if model_src == "paddlenlp":
        return PaddleNLP(**model_config)
    if model_src == "uform":
        return UForm(**model_config)


def _get_eval(strategy, kws=None) -> Any:
    strategy = strategy.lower()
    kws = kws or {}

    if "distance" in strategy:
        return SearchDistanceEvaluation(**kws)
    if "np" in strategy:
        return NumpyNormEvaluation(**kws)
    if "exact" in strategy:
        return ExactMatchEvaluation()
    if "onnx" in strategy:
        return OnnxModelEvaluation(**kws)
    if "kreciprocal" in strategy:
        return KReciprocalEvaluation(**kws)
    if "cohere" in strategy:
        return CohereRerankEvaluation(**kws)
    if "sequence_match" in strategy:
        return SequenceMatchEvaluation(**kws)
    if "time" in strategy:
        return TimeEvaluation(**kws)
    if "sbert_crossencoder" in strategy:
        return SbertCrossencoderEvaluation(**kws)


def _get_pre_func(pre_process) -> Any:
    return getattr(byte.processor.pre, pre_process)


def _get_pre_context_function(pre_context_process, kws=None) -> Any:
    pre_context_process = pre_context_process.lower()
    kws = kws or {}
    if pre_context_process == "summarization":
        return SummarizationContextProcess(**kws)
    if pre_context_process == "selective":
        return SelectiveContextProcess(**kws)
    if pre_context_process == "concat":
        return ConcatContextProcess()


def _get_post_func(post_process) -> Any:
    return getattr(byte.processor.post, post_process)
