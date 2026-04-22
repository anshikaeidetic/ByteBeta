"""Multi-tier cache manager with Tier-0 memory and optional write-behind."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any


class _Tier1Ref:
    """Marker object returned from ``search`` for in-memory hits."""

    def __init__(self, key: Any) -> None:
        self.key = key

    def __repr__(self) -> str:
        return f"<tier1:{str(self.key)[:24]}>"


class _Tier1ScalarData:
    """Minimal scalar-data object compatible with the adapter pipeline."""

    def __init__(self, question: Any, answer: Any, embedding_data: Any) -> None:
        self.question = question
        self.answers = [answer]
        self.embedding_data = embedding_data
        self.create_on = datetime.now()


class TieredCacheManager:
    """Two-tier cache wrapper with Tier-0 memory and optional write-behind."""

    def __init__(
        self,
        backend: Any,
        tier1_max_size: int = 1000,
        promotion_threshold: int = 2,
        promotion_window_s: float = 300.0,
        promote_on_write: bool = True,
        async_write_back: bool = False,
        async_write_back_queue_size: int = 10000,
    ) -> None:
        self._backend = backend
        self._tier1_max = max(1, int(tier1_max_size or 1000))
        self._promotion_threshold = max(1, int(promotion_threshold or 2))
        self._promotion_window = max(0.001, float(promotion_window_s or 300.0))
        self._promote_on_write = bool(promote_on_write)
        self._async_write_back = bool(async_write_back)

        self._tier1: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._access_counts: dict[Any, list[float]] = {}

        self._tier1_hits = 0
        self._tier2_hits = 0
        self._misses = 0
        self._promotions = 0
        self._demotions = 0
        self._queued_writes = 0
        self._write_back_failures = 0

        self._save_queue: queue.Queue[Any] | None = None
        self._worker: threading.Thread | None = None
        self._stop_sentinel = object()
        if self._async_write_back:
            self._save_queue = queue.Queue(
                maxsize=max(1, int(async_write_back_queue_size or 10000))
            )
            self._worker = threading.Thread(
                target=self._write_worker,
                name="byte-tiered-writeback",
                daemon=True,
            )
            self._worker.start()

    # ------------------------------------------------------------------
    # DataManager-compatible interface
    # ------------------------------------------------------------------

    def save(self, question, answer, embedding_data, **kwargs) -> Any | None:
        """Save to Tier-0 immediately, then backend sync or async."""
        key = self._make_key(embedding_data)
        if self._promote_on_write or self._async_write_back:
            self._store_tier1(key, _Tier1ScalarData(question, answer, embedding_data))

        if not self._async_write_back:
            return self._backend.save(question, answer, embedding_data, **kwargs)

        task = (question, answer, embedding_data, dict(kwargs))
        try:
            assert self._save_queue is not None
            self._save_queue.put_nowait(task)
            with self._lock:
                self._queued_writes += 1
            return None
        except queue.Full:
            with self._lock:
                self._write_back_failures += 1
            return self._backend.save(question, answer, embedding_data, **kwargs)

    def search(self, embedding_data, **kwargs) -> Any:
        """Search Tier-0 first, then fall back to the backend."""
        key = self._make_key(embedding_data)
        with self._lock:
            entry = self._tier1.get(key)
            if entry is not None:
                self._tier1.move_to_end(key)
                self._tier1_hits += 1
                return [(1.0, _Tier1Ref(key))]

        results = self._backend.search(embedding_data, **kwargs)
        if results and results[0]:
            self._tier2_hits += 1
            self._record_access(key)
            if self._should_promote(key):
                promoted = self._promote_from_backend(key, results[0])
                if promoted:
                    return [(1.0, _Tier1Ref(key))]
            return results

        self._misses += 1
        return results

    def get_scalar_data(self, search_data, **kwargs) -> Any | None:
        ref = self._extract_tier1_ref(search_data)
        if ref is not None:
            with self._lock:
                entry = self._tier1.get(ref.key)
                if entry is not None:
                    self._tier1.move_to_end(ref.key)
                    return entry
        if hasattr(search_data, "question") and hasattr(search_data, "answers"):
            return search_data
        if hasattr(self._backend, "get_scalar_data"):
            return self._backend.get_scalar_data(search_data, **kwargs)
        return None

    def hit_cache_callback(self, search_data) -> Any | None:
        if self._extract_tier1_ref(search_data) is not None:
            return None
        if hasattr(self._backend, "hit_cache_callback"):
            return self._backend.hit_cache_callback(search_data)
        return None

    def add_session(self, search_data, *args, **kwargs) -> Any | None:
        if self._extract_tier1_ref(search_data) is not None:
            return None
        if hasattr(self._backend, "add_session"):
            return self._backend.add_session(search_data, *args, **kwargs)
        return None

    def flush(self) -> None:
        """Flush queued writes and then flush the backend."""
        if self._save_queue is not None:
            self._save_queue.join()
        if hasattr(self._backend, "flush"):
            self._backend.flush()

    async def aflush(self) -> None:
        """Flush queued writes without blocking the event loop."""
        if self._save_queue is not None:
            await asyncio.to_thread(self._save_queue.join)
        if hasattr(self._backend, "aflush"):
            await self._backend.aflush()
        elif hasattr(self._backend, "flush"):
            await asyncio.to_thread(self._backend.flush)

    def close(self) -> None:
        """Drain queued writes, stop the worker, then close the backend."""
        if self._save_queue is not None:
            self._save_queue.join()
            self._save_queue.put(self._stop_sentinel)
            if self._worker is not None:
                self._worker.join(timeout=2.0)
            self._save_queue = None
            self._worker = None
        if hasattr(self._backend, "close"):
            self._backend.close()

    async def aclose(self) -> None:
        """Drain queued writes, stop the worker, then close the backend."""
        if self._save_queue is not None:
            await asyncio.to_thread(self._save_queue.join)
            await asyncio.to_thread(self._save_queue.put, self._stop_sentinel)
            if self._worker is not None:
                await asyncio.to_thread(self._worker.join, 2.0)
            self._save_queue = None
            self._worker = None
        if hasattr(self._backend, "aclose"):
            await self._backend.aclose()
        elif hasattr(self._backend, "close"):
            await asyncio.to_thread(self._backend.close)

    async def asave(self, question, answer, embedding_data, **kwargs) -> Any:
        return await asyncio.to_thread(self.save, question, answer, embedding_data, **kwargs)

    async def asearch(self, embedding_data, **kwargs) -> Any:
        return await asyncio.to_thread(self.search, embedding_data, **kwargs)

    async def aget_scalar_data(self, search_data, **kwargs) -> Any:
        return await asyncio.to_thread(self.get_scalar_data, search_data, **kwargs)

    async def ahit_cache_callback(self, search_data) -> Any:
        return await asyncio.to_thread(self.hit_cache_callback, search_data)

    async def aadd_session(self, search_data, *args, **kwargs) -> Any:
        return await asyncio.to_thread(self.add_session, search_data, *args, **kwargs)

    async def areport_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ) -> Any | None:
        if hasattr(self._backend, "areport_cache"):
            return await self._backend.areport_cache(
                user_question,
                cache_question,
                cache_question_id,
                cache_answer,
                similarity_value,
                cache_delta_time,
            )
        if hasattr(self._backend, "report_cache"):
            return await asyncio.to_thread(
                self._backend.report_cache,
                user_question,
                cache_question,
                cache_question_id,
                cache_answer,
                similarity_value,
                cache_delta_time,
            )
        return None

    # ------------------------------------------------------------------
    # Tier management
    # ------------------------------------------------------------------

    def _make_key(self, embedding_data) -> Any:
        if hasattr(embedding_data, "tobytes"):
            return embedding_data.tobytes()
        if isinstance(embedding_data, list):
            return tuple(embedding_data)
        return embedding_data

    def _record_access(self, key: Any) -> None:
        now = time.time()
        accesses = self._access_counts.setdefault(key, [])
        cutoff = now - self._promotion_window
        accesses[:] = [value for value in accesses if value > cutoff]
        accesses.append(now)

    def _should_promote(self, key: Any) -> bool:
        return len(self._access_counts.get(key, [])) >= self._promotion_threshold

    def _promote_from_backend(self, key: Any, search_data: Any) -> bool:
        if not hasattr(self._backend, "get_scalar_data"):
            self._store_tier1(key, search_data)
            return True
        try:
            scalar = self._backend.get_scalar_data(search_data)
        except Exception:  # pylint: disable=broad-except
            return False
        if scalar is None:
            return False
        self._store_tier1(key, scalar)
        return True

    def _store_tier1(self, key: Any, data: Any) -> None:
        with self._lock:
            if key in self._tier1:
                self._tier1[key] = data
                self._tier1.move_to_end(key)
                return
            while len(self._tier1) >= self._tier1_max:
                self._tier1.popitem(last=False)
                self._demotions += 1
            self._tier1[key] = data
            self._promotions += 1

    def evict_tier1(self, key: Any | None = None) -> None:
        with self._lock:
            if key and key in self._tier1:
                del self._tier1[key]
                self._demotions += 1
            elif not key and self._tier1:
                self._tier1.popitem(last=False)
                self._demotions += 1

    # ------------------------------------------------------------------
    # Async write-behind
    # ------------------------------------------------------------------

    def _write_worker(self) -> None:
        assert self._save_queue is not None
        while True:
            task = self._save_queue.get()
            try:
                if task is self._stop_sentinel:
                    return
                question, answer, embedding_data, kwargs = task
                self._backend.save(question, answer, embedding_data, **kwargs)
            except Exception:  # pylint: disable=broad-except
                with self._lock:
                    self._write_back_failures += 1
            finally:
                self._save_queue.task_done()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total_hits = self._tier1_hits + self._tier2_hits
        total_requests = total_hits + self._misses
        pending = self._save_queue.qsize() if self._save_queue is not None else 0
        return {
            "tier1_size": len(self._tier1),
            "tier1_max_size": self._tier1_max,
            "tier1_hits": self._tier1_hits,
            "tier2_hits": self._tier2_hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "tier1_hit_ratio": (
                round(self._tier1_hits / total_requests, 4) if total_requests else 0.0
            ),
            "promotions": self._promotions,
            "demotions": self._demotions,
            "promote_on_write": self._promote_on_write,
            "async_write_back": self._async_write_back,
            "pending_async_writes": pending,
            "queued_writes": self._queued_writes,
            "write_back_failures": self._write_back_failures,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tier1_ref(search_data: Any) -> _Tier1Ref | None:
        if (
            isinstance(search_data, tuple)
            and len(search_data) >= 2
            and isinstance(search_data[1], _Tier1Ref)
        ):
            return search_data[1]
        return None

    # ------------------------------------------------------------------
    # Delegate everything else to backend
    # ------------------------------------------------------------------

    def __getattr__(self, name) -> Any:
        return getattr(self._backend, name)
