import threading
import time
from types import SimpleNamespace

from byte.manager.scalar_data.base import CacheData
from byte.manager.tiered_cache import TieredCacheManager


class SlowBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.store = {}

    def save(self, question, answer, embedding_data, **kwargs) -> object:  # pylint: disable=unused-argument
        time.sleep(0.2)
        key = embedding_data
        with self._lock:
            self.store[key] = (question, answer, embedding_data)
        return key

    def search(self, embedding_data, **kwargs) -> object:  # pylint: disable=unused-argument
        with self._lock:
            if embedding_data in self.store:
                return [(1.0, embedding_data)]
        return [None]

    def get_scalar_data(self, search_data, **kwargs) -> object:  # pylint: disable=unused-argument
        key = search_data[1]
        with self._lock:
            question, answer, embedding_data = self.store[key]
        return SimpleNamespace(
            question=question,
            answers=[answer],
            embedding_data=embedding_data,
            create_on=None,
        )

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_tiered_cache_can_return_from_tier1_before_writeback_finishes() -> None:
    backend = SlowBackend()
    manager = TieredCacheManager(
        backend,
        tier1_max_size=8,
        promote_on_write=True,
        async_write_back=True,
    )
    started = time.perf_counter()
    manager.save("question", "answer", "emb-key")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1

    search_data = manager.search("emb-key")[0]
    scalar = manager.get_scalar_data(search_data)
    assert scalar.question == "question"
    assert scalar.answers[0] == "answer"
    assert manager.stats()["tier1_hits"] == 1

    manager.flush()
    assert "emb-key" in backend.store
    manager.close()


class PromotingBackend:
    def __init__(self) -> None:
        self.scalar = CacheData("question", "answer", embedding_data="emb-key")

    def search(self, embedding_data, **kwargs) -> object:  # pylint: disable=unused-argument
        return [(1.0, embedding_data)]

    def get_scalar_data(self, search_data, **kwargs) -> object:  # pylint: disable=unused-argument
        if isinstance(search_data, CacheData):
            raise AssertionError("Tiered cache should return promoted scalar data directly.")
        return self.scalar

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_tiered_cache_returns_promoted_cachedata_without_backend_roundtrip() -> None:
    manager = TieredCacheManager(
        PromotingBackend(),
        tier1_max_size=8,
        promotion_threshold=1,
        promote_on_write=False,
    )

    first_search = manager.search("emb-key")[0]
    first_scalar = manager.get_scalar_data(first_search)
    assert first_scalar.question == "question"

    second_search = manager.search("emb-key")[0]
    second_scalar = manager.get_scalar_data(second_search)
    assert second_scalar.question == "question"
    assert second_scalar.answers[0].answer == "answer"
