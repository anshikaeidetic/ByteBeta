from __future__ import annotations

import os
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from byte.config import Config
from byte.embedding.string import to_embeddings as string_embedding
from byte.manager import get_data_manager
from byte.manager.data_manager import DataManager
from byte.processor.post import temperature_softmax
from byte.processor.pre import last_content
from byte.prompt_distillation import PromptModuleRegistry
from byte.quantization.vector import maybe_wrap_vector_data_manager
from byte.report import Report
from byte.security import maybe_wrap_data_manager
from byte.similarity_evaluation import ExactMatchEvaluation, SimilarityEvaluation
from byte.telemetry import build_library_telemetry
from byte.utils.async_ops import run_sync
from byte.utils.cache_func import cache_all
from byte.utils.error import NotInitError
from byte.utils.log import byte_log

_LIVE_CACHES = weakref.WeakSet()
_DEFAULT_CACHE: object | None = None
_DEFAULT_CACHE_LOCK = threading.RLock()
_T = TypeVar("_T")


@dataclass(frozen=True)
class MemoryStoreSpec:
    attr: str
    summary_key: str
    stats_method: str
    supports_limit: bool = True


def _best_effort(
    default: _T,
    operation: Callable[[], _T],
    *,
    message: str,
    log_errors: bool = True,
) -> _T:
    """Run lifecycle bookkeeping behind a best-effort boundary."""

    try:
        return operation()
    except Exception as exc:  # lifecycle cleanup boundary
        if log_errors and not os.getenv("IS_CI"):
            byte_log.error("%s: %s", message, exc)
        return default


def _close_cache_instance(cache_obj, suppress_errors: bool = False) -> None:
    def close_cache() -> None:
        if cache_obj is None or getattr(cache_obj, "_closed", False):
            return
        telemetry = getattr(cache_obj, "telemetry", None)
        if telemetry is not None:
            telemetry.shutdown()
            cache_obj.telemetry = None
        data_manager = getattr(cache_obj, "data_manager", None)
        if data_manager is not None:
            data_manager.close()
        cache_obj._closed = True

    _best_effort(
        None,
        close_cache,
        message="Failed to close cache cleanly",
        log_errors=not suppress_errors,
    )


def _close_live_caches() -> None:
    for cache_obj in list(_LIVE_CACHES):
        _close_cache_instance(cache_obj, suppress_errors=True)


class _CacheLifecycleMixin:
    def __init__(self) -> None:
        self.has_init = False
        self._closed = False
        self.cache_enable_func = None
        self.pre_embedding_func = None
        self.embedding_func = None
        self.data_manager: DataManager | None = None
        self.similarity_evaluation: SimilarityEvaluation | None = None
        self.post_process_messages_func = None
        self.config = Config()
        self.report = Report(owner_cache=self)
        self.telemetry = None
        self.next_cache = None
        self.memory_scope: str | None = None
        self.intent_graph = None
        self._reset_memory_store_refs()
        _LIVE_CACHES.add(self)

    @staticmethod
    def _memory_store_specs() -> tuple[MemoryStoreSpec, ...]:
        return (
            MemoryStoreSpec("tool_result_store", "tool_results", "tool_memory_stats"),
            MemoryStoreSpec("ai_memory_store", "ai_memory", "ai_memory_stats"),
            MemoryStoreSpec("execution_memory_store", "execution_memory", "execution_memory_stats"),
            MemoryStoreSpec(
                "failure_memory_store",
                "failure_memory",
                "failure_memory_stats",
                supports_limit=False,
            ),
            MemoryStoreSpec("patch_pattern_store", "patch_patterns", "patch_pattern_stats"),
            MemoryStoreSpec("prompt_piece_store", "prompt_pieces", "prompt_piece_stats"),
            MemoryStoreSpec("prompt_module_registry", "prompt_modules", "prompt_module_stats"),
            MemoryStoreSpec("artifact_memory_store", "artifact_memory", "artifact_memory_stats"),
            MemoryStoreSpec("workflow_plan_store", "workflow_plans", "workflow_plan_stats"),
            MemoryStoreSpec("session_delta_store", "session_deltas", "session_delta_stats"),
            MemoryStoreSpec("reasoning_memory_store", "reasoning_memory", "reasoning_memory_stats"),
        )

    def _reset_memory_store_refs(self) -> None:
        for spec in self._memory_store_specs():
            setattr(self, spec.attr, None)

    def _iter_memory_stores(self) -> Any:
        for spec in self._memory_store_specs():
            yield spec, getattr(self, spec.attr, None)

    def _memory_runtime_ready(self) -> bool:
        if self.intent_graph is None:
            return False
        return all(store is not None for _, store in self._iter_memory_stores())

    def _build_local_memory_stores(self, config: Config) -> dict[str, Any]:
        from byte.processor.ai_memory import AIMemoryStore
        from byte.processor.execution import (
            ExecutionMemoryStore,
            FailureMemoryStore,
            PatchPatternStore,
        )
        from byte.processor.optimization_memory import (
            ArtifactMemoryStore,
            PromptPieceStore,
            SessionDeltaStore,
            WorkflowPlanStore,
        )
        from byte.processor.reasoning_reuse import ReasoningMemoryStore
        from byte.processor.tool_result import ToolResultStore

        return {
            "tool_result_store": ToolResultStore(),
            "ai_memory_store": AIMemoryStore(
                max_entries=config.memory_max_entries,
                embedding_preview_dims=config.memory_embedding_preview_dims,
                embedding_codec=config.vector_codec,
                embedding_bits=config.vector_bits,
            ),
            "execution_memory_store": ExecutionMemoryStore(max_entries=config.memory_max_entries),
            "failure_memory_store": FailureMemoryStore(),
            "patch_pattern_store": PatchPatternStore(),
            "prompt_piece_store": PromptPieceStore(
                max_entries=config.memory_max_entries * 2,
                codec_name=config.vector_codec,
                bits=config.vector_bits,
            ),
            "prompt_module_registry": PromptModuleRegistry(
                max_entries=config.memory_max_entries * 2,
                artifact_version=config.prompt_distillation_artifact_version,
            ),
            "artifact_memory_store": ArtifactMemoryStore(
                max_entries=config.memory_max_entries,
                codec_name=config.vector_codec,
                bits=config.vector_bits,
            ),
            "workflow_plan_store": WorkflowPlanStore(max_entries=config.memory_max_entries),
            "session_delta_store": SessionDeltaStore(max_entries=config.memory_max_entries * 2),
            "reasoning_memory_store": ReasoningMemoryStore(
                max_entries=config.memory_max_entries,
                codec_name=config.vector_codec,
                bits=config.vector_bits,
            ),
        }

    def _ensure_memory_runtime(self) -> None:
        if not self._memory_runtime_ready():
            self._init_memory_runtime(self.config)

    def _init_memory_runtime(self, config: Config | None) -> None:
        from byte.processor.intent import IntentGraph
        from byte.processor.shared_memory import get_shared_memory

        config = config or Config()
        self.memory_scope = config.memory_scope
        self._reset_memory_store_refs()
        if config.memory_scope:
            shared = get_shared_memory(config.memory_scope)
            self.intent_graph = shared.intent_graph
            for spec in self._memory_store_specs():
                setattr(self, spec.attr, getattr(shared, spec.attr))
        else:
            self.intent_graph = IntentGraph(window_size=config.fingerprint_window)
            for attr, store in self._build_local_memory_stores(config).items():
                setattr(self, attr, store)

    def _configure_telemetry(self, config: Config) -> None:
        self.report.attach_owner(self)
        if self.telemetry is not None:
            self.telemetry.shutdown()
        self.telemetry = build_library_telemetry(self, config)

    def _maybe_apply_cost_aware_eviction(self, config: Any) -> None:
        """Swap the data manager's eviction policy in-place when the config asks for
        COST_AWARE (arXiv 2508.07675). The existing MemoryCacheEviction already
        delegates to CostAwareCacheEviction under the same class, so we just need
        to re-instantiate it with the new policy name and replace the reference.
        """
        policy = str(getattr(config, "eviction_policy", "") or "").upper().replace("-", "_")
        if policy not in ("COST_AWARE",):
            return
        dm = getattr(self, "data_manager", None)
        current = getattr(dm, "eviction_base", None) if dm is not None else None
        if current is None:
            return
        # Already COST_AWARE — no-op.
        if str(getattr(current, "policy", "") or "").upper() == "COST_AWARE":
            return
        try:
            from byte.manager.eviction.memory_cache import (
                MemoryCacheEviction,  # pylint: disable=import-outside-toplevel
            )
            fallback_maxsize = getattr(getattr(current, "_cache", None), "maxsize", 1000)
            new_base = MemoryCacheEviction(
                policy="COST_AWARE",
                maxsize=getattr(current, "_maxsize", fallback_maxsize),
                clean_size=getattr(current, "_clean_size", 100),
                on_evict=dm._clear if hasattr(dm, "_clear") else None,
            )
            # Seed with existing IDs so we don't lose tracking on swap.
            try:
                existing_ids = list(self.data_manager.s.get_ids(deleted=False) or [])
                if existing_ids:
                    new_base.put(existing_ids)
            except Exception:  # pragma: no cover - defensive
                pass
            dm.eviction_base = new_base
            byte_log.info("Cost-aware eviction policy activated (arXiv 2508.07675)")
        except Exception as exc:  # pragma: no cover - defensive
            byte_log.warning("Failed to apply COST_AWARE eviction: %s", exc)

    def init(
        self,
        cache_enable_func=cache_all,
        pre_embedding_func=last_content,
        pre_func=None,
        embedding_func=string_embedding,
        data_manager: DataManager | None = None,
        similarity_evaluation=None,
        post_process_messages_func=temperature_softmax,
        post_func=None,
        config=None,
        next_cache=None,
    ) -> None:
        """Pass parameters to initialize ByteAI Cache.

        :param cache_enable_func: a function to enable cache, defaults to ``cache_all``
        :param pre_embedding_func: a function to preprocess embedding, defaults to ``last_content``
        :param pre_func: a function to preprocess embedding, same as ``pre_embedding_func``
        :param embedding_func: a function to extract embeddings from requests for similarity search, defaults to ``string_embedding``
        :param data_manager: a ``DataManager`` module, defaults to ``get_data_manager()``
        :param similarity_evaluation: a module to calculate embedding similarity, defaults to ``ExactMatchEvaluation()``
        :param post_process_messages_func: a function to post-process messages, defaults to ``temperature_softmax`` with a default temperature of 0.0
        :param post_func: a function to post-process messages, same as ``post_process_messages_func``
        :param config: a module to pass configurations, defaults to ``Config()``
        :param next_cache: customized method for next cache
        """
        if data_manager is None:
            data_manager = get_data_manager()
        if similarity_evaluation is None:
            similarity_evaluation = ExactMatchEvaluation()
        if config is None:
            config = Config()
        self.has_init = True
        self.cache_enable_func = cache_enable_func
        self.pre_embedding_func = pre_func if pre_func else pre_embedding_func
        self.embedding_func = embedding_func
        self.data_manager = maybe_wrap_data_manager(
            maybe_wrap_vector_data_manager(data_manager, config),
            config,
        )
        self.similarity_evaluation = similarity_evaluation
        self.post_process_messages_func = post_func if post_func else post_process_messages_func
        self.config = config
        self.next_cache = next_cache
        self._closed = False
        self._configure_telemetry(config)
        self._init_memory_runtime(config)
        self._maybe_apply_cost_aware_eviction(config)

    def import_data(
        self,
        questions: list[Any],
        answers: list[Any],
        session_ids: list[str | None] | None = None,
    ) -> None:
        """Import data to byte."""
        from byte.processor.batching import batch_embed

        self.data_manager.import_data(  # type: ignore
            questions=questions,
            answers=answers,
            embedding_datas=batch_embed(self.embedding_func, questions),
            session_ids=session_ids if session_ids else [None for _ in range(len(questions))],
        )

    async def aimport_data(
        self,
        questions: list[Any],
        answers: list[Any],
        session_ids: list[str | None] | None = None,
    ) -> None:
        await run_sync(self.import_data, questions, answers, session_ids)

    def flush(self) -> None:
        """Flush data."""
        self.data_manager.flush()
        if self.next_cache:
            self.next_cache.data_manager.flush()

    async def aflush(self) -> None:
        """Flush data without blocking the event loop."""
        await self.data_manager.aflush()
        if self.next_cache:
            await self.next_cache.data_manager.aflush()

    def close(self, suppress_errors: bool = False) -> None:
        """Close the underlying data manager."""
        _close_cache_instance(self, suppress_errors=suppress_errors)

    async def aclose(self, suppress_errors: bool = False) -> None:
        """Close the underlying data manager without blocking the event loop."""
        await run_sync(self.close, suppress_errors=suppress_errors)

    def invalidate_by_query(self, query: str) -> bool:
        """Delete a cached entry whose question matches *query*."""
        if not self.has_init:
            raise NotInitError()

        # Only warn about the legacy path when no modern invalidate_by_query exists —
        # otherwise this fires every request and is pure noise, since the modern
        # method is called directly below.
        if not hasattr(self.data_manager, "invalidate_by_query"):
            byte_log.warning(
                "Legacy data_manager invalidation path in use; implement invalidate_by_query()."
            )
        try:
            return bool(
                self.data_manager.invalidate_by_query(query, embedding_func=self.embedding_func)
            )
        except Exception as exc:
            byte_log.error("invalidate_by_query failed: %s", exc)
        return False

    async def ainvalidate_by_query(self, query: str) -> bool:
        """Async wrapper for query invalidation."""
        return await run_sync(self.invalidate_by_query, query)

    def __del__(self) -> None:
        _close_cache_instance(self, suppress_errors=True)

    def clear(self) -> None:
        """Delete **all** cached entries."""
        if not self.has_init:
            raise NotInitError()

        if hasattr(self.data_manager, "data") and hasattr(self.data_manager.data, "clear"):
            self.data_manager.data.clear()
        elif hasattr(self.data_manager, "s"):
            scalar = self.data_manager.s
            # Prefer a native clear() if the backend provides one, otherwise fall
            # back to the abstract "soft-delete everything then sweep" sequence
            # which every CacheStorage implementation supports by contract.
            cleared_native = False
            native_clear = getattr(scalar, "clear", None)
            if callable(native_clear):
                try:
                    native_clear()
                    cleared_native = True
                except Exception as exc:  # pylint: disable=W0703
                    byte_log.warning("Scalar storage clear() failed, falling back: %s", exc)
            if not cleared_native:
                try:
                    all_ids = scalar.get_ids(deleted=False)
                    if all_ids:
                        scalar.mark_deleted(all_ids)
                    scalar.clear_deleted_data()
                except Exception as exc:  # pylint: disable=W0703
                    byte_log.warning("Scalar storage clear fallback failed: %s", exc)
        if self.intent_graph is not None:
            self.intent_graph.clear()
        for _, store in self._iter_memory_stores():
            clear_func = getattr(store, "clear", None)
            if callable(clear_func):
                clear_func()

    async def aclear(self) -> None:
        """Async wrapper for clearing all cache state."""
        await run_sync(self.clear)

    def cost_summary(self) -> dict[str, Any]:
        """Return a summary of cache performance and estimated cost savings."""
        total_requests = self.report.op_pre.count
        cache_hits = self.report.hint_cache_count
        cache_misses = total_requests - cache_hits
        hit_ratio = round(cache_hits / total_requests, 4) if total_requests else 0.0

        result = {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_ratio": hit_ratio,
            "avg_latency_with_cache_ms": round(
                self.report.average_pre_time() * 1000 + self.report.average_search_time() * 1000, 2
            ),
            "avg_llm_latency_ms": round(self.report.average_llm_time() * 1000, 2),
            "total_llm_calls_avoided": cache_hits,
            "total_time_saved_s": (
                round(cache_hits * self.report.average_llm_time(), 2)
                if self.report.op_llm.count
                else 0
            ),
        }

        def build_budget_summary() -> dict[str, Any]:
            from byte.adapter.runtime_state import (
                get_adaptive_threshold,
                get_budget_tracker,
                get_quality_scorer,
            )
            from byte.processor.policy import policy_stats

            budget = get_budget_tracker(self).summary()
            extras: dict[str, Any] = {"budget": budget}

            if getattr(self.config, "adaptive_threshold", False):
                extras["adaptive_threshold"] = get_adaptive_threshold(self).stats()

            quality = get_quality_scorer(self).stats()
            if quality["total_scored"] > 0:
                extras["quality"] = quality
            extras["global_policy"] = policy_stats()
            return extras

        result.update(_best_effort({}, build_budget_summary, message="cost_summary enrichment failed"))

        memory = self.memory_summary()
        result.update(memory)
        return result

    def warm(self, data) -> Any:
        """Pre-populate the cache with Q&A pairs (eliminates cold-start)."""
        from byte.processor.warmer import CacheWarmer

        warmer = CacheWarmer(self)
        if isinstance(data, (str, type(None))):
            from pathlib import Path

            return warmer.warm_from_file(Path(data))
        if isinstance(data, list):
            return warmer.warm_from_dict(data)
        raise ValueError("data must be a list of dicts or a file path")

    def quality_stats(self) -> Any:
        """Return quality scoring statistics."""
        def read_quality_stats() -> Any:
            from byte.adapter.adapter import get_quality_scorer

            return get_quality_scorer(self).stats()

        return _best_effort({}, read_quality_stats, message="quality_stats unavailable")

    def feedback(self, query: str, thumbs_up: bool) -> Any:
        """Record user feedback for a cached query."""
        def record_feedback() -> Any:
            from byte.adapter.adapter import get_quality_scorer

            return get_quality_scorer(self).record_feedback(query, thumbs_up)

        return _best_effort({}, record_feedback, message="feedback recording failed")

    @staticmethod
    def set_openai_key() -> None:
        """Set OpenAI API key from environment variable."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            byte_log.warning(
                "OPENAI_API_KEY environment variable is not set. Set it before making API calls."
            )

    @staticmethod
    def set_azure_openai_key() -> None:
        """Set Azure OpenAI configuration from environment variables."""
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
        if not api_key:
            byte_log.warning("AZURE_OPENAI_API_KEY or OPENAI_API_KEY not set.")
        if not endpoint:
            byte_log.warning("AZURE_OPENAI_ENDPOINT or OPENAI_API_BASE not set.")
