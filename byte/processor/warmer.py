"""Smart Cache Warming Engine.

Pre-populates the cache with Q&A pairs so there is no cold-start
penalty. Supports loading from dicts, JSON files, and CSV files.
"""

import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

byte_log = logging.getLogger("byte")
_T = TypeVar("_T")


def _best_effort(default: _T, operation: Callable[[], _T], *, message: str) -> _T:
    """Run a cache-warming boundary operation without surfacing warm-time failures."""

    try:
        return operation()
    except Exception as exc:  # best-effort warming boundary
        byte_log.debug("%s: %s", message, exc)
        return default


class CacheWarmer:
    """Pre-populates a ByteAI cache instance with known Q&A pairs.

    :param cache: an initialised ``Cache`` instance
    """

    def __init__(self, cache: Any) -> None:
        self._cache = cache
        self._seeded = 0
        self._skipped = 0
        self._cache_writes = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm_from_dict(self, data: list[dict[str, str]]) -> dict[str, int]:
        """Import a list of Q&A dicts into the cache.

        Each dict must have ``"question"`` and ``"answer"`` keys.

        :param data: list of {"question": ..., "answer": ...}
        :return: summary with counts of seeded / skipped entries
        """
        for item in data:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if not q or not a:
                self._skipped += 1
                continue
            request_kwargs = item.get("request_kwargs")
            if not isinstance(request_kwargs, dict):
                request_kwargs = {"messages": [{"role": "user", "content": q}]}
                if item.get("model"):
                    request_kwargs["model"] = item["model"]
            self._seed_one(q, a, request_kwargs=request_kwargs)
        return self._summary()

    def warm_from_file(self, path: str | Path) -> dict[str, int]:
        """Import Q&A pairs from a JSON or CSV file.

        JSON: expects a list of ``{"question": ..., "answer": ...}``
        CSV: expects columns ``question,answer`` (with header row)

        :param path: path to file
        :return: summary with counts
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Warm file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return self.warm_from_dict(data)
        elif suffix == ".csv":
            return self._warm_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix} (use .json or .csv)")

    def warm_from_history(self, max_entries: int = 500) -> dict[str, int]:
        """Analyse existing cache and re-warm entries to refresh embeddings.

        Useful after changing the embedding model or similarity evaluator.

        :param max_entries: max entries to re-process
        :return: summary with counts
        """
        def seed_history() -> None:
            dm = self._cache.data_manager
            if hasattr(dm, "data") and hasattr(dm.data, "values"):
                count = 0
                for entry in list(dm.data.values()):
                    if count >= max_entries:
                        break
                    q = entry[0] if isinstance(entry, tuple) and len(entry) > 1 else None
                    a = entry[1] if isinstance(entry, tuple) and len(entry) > 1 else None
                    if q and a:
                        self._seed_one(str(q), str(a))
                        count += 1

        _best_effort(None, seed_history, message="warm_from_history")

        return self._summary()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seed_one(
        self, question: str, answer: str, *, request_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Attempt to seed a single Q&A pair into the cache."""

        def seed_entry() -> int:
            return self._seed_cache_chain(
                self._cache,
                question,
                answer,
                request_kwargs=request_kwargs,
            )

        writes = _best_effort(0, seed_entry, message="Cache warm skip")
        if writes:
            self._cache_writes += writes
            self._seeded += 1
        else:
            self._skipped += 1

    def _seed_cache_chain(
        self,
        cache_obj: Any,
        question: str,
        answer: str,
        *,
        request_kwargs: dict[str, Any] | None = None,
    ) -> int:
        from byte.adapter.pipeline.context import _apply_request_namespaces
        from byte.manager.scalar_data.base import Answer, DataType

        if not hasattr(cache_obj, "embedding_func") or cache_obj.embedding_func is None:
            raise ValueError("Cache is not fully initialized for warming")

        effective_request = request_kwargs or {"messages": [{"role": "user", "content": question}]}
        pre_store_data, pre_embedding_data = self._preprocess_question(
            cache_obj,
            question,
            request_kwargs=effective_request,
        )
        if hasattr(cache_obj, "config") and cache_obj.config is not None:
            pre_embedding_data = _apply_request_namespaces(
                pre_embedding_data,
                effective_request,
                cache_obj,
            )
        embedding = cache_obj.embedding_func(pre_embedding_data)
        cache_obj.data_manager.save(
            pre_store_data,
            Answer(answer, DataType.STR),
            embedding,
        )

        writes = 1
        if getattr(cache_obj, "next_cache", None) is not None:
            writes += self._seed_cache_chain(
                cache_obj.next_cache,
                question,
                answer,
                request_kwargs=effective_request,
            )
        return writes

    def _preprocess_question(
        self,
        cache_obj: Any,
        question: str,
        *,
        request_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        pre_func = getattr(cache_obj, "pre_embedding_func", None)
        if pre_func is None:
            return question, question

        payloads = []
        if isinstance(request_kwargs, dict) and request_kwargs:
            payloads.append(dict(request_kwargs))
        payloads.extend(
            [
                {"messages": [{"role": "user", "content": question}]},
                {"prompt": question},
                {"input": question},
            ]
        )
        for payload in payloads:
            def preprocess_payload() -> Any:
                cache_config = getattr(cache_obj, "config", None)
                return pre_func(
                    payload,
                    prompts=getattr(cache_config, "prompts", None),
                    cache_config=cache_config,
                )
            result = _best_effort(None, preprocess_payload, message="warm preprocessing skipped")
            if isinstance(result, tuple):
                if result[1] in (None, ""):
                    continue
                return result
            if result not in (None, ""):
                return result, result
        return question, question

    def _warm_csv(self, path: Path) -> dict[str, int]:
        """Parse CSV with question,answer columns."""
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                data.append(
                    {
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                    }
                )
        return self.warm_from_dict(data)

    def _summary(self) -> dict[str, int]:
        return {
            "seeded": self._seeded,
            "skipped": self._skipped,
            "cache_writes": self._cache_writes,
            "total_processed": self._seeded + self._skipped,
        }
