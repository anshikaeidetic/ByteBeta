from collections.abc import Callable
from typing import Any

import cachetools

from byte.manager.eviction.base import EvictionBase


def popitem_wrapper(func, wrapper_func, clean_size) -> Any:
    def wrapper(*args, **kwargs) -> None:
        keys = []
        try:
            keys = [func(*args, **kwargs)[0] for _ in range(clean_size)]
        except KeyError:
            pass
        wrapper_func(keys)

    return wrapper


class MemoryCacheEviction(EvictionBase):
    """eviction: Memory Cache

    :param policy: eviction strategy
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param clean_size: will clean the size of data when the size of cache data reaches the max size
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]


    """

    def __init__(
        self,
        policy: str = "LRU",
        maxsize: int = 1000,
        clean_size: int = 0,
        on_evict: Callable[[list[Any]], None] | None = None,
        **kwargs,
    ) -> None:
        self._policy = policy.upper()
        # COST_AWARE (arXiv 2508.07675) — delegates to CostAwareCacheEviction.
        if self._policy in ("COST_AWARE", "COST-AWARE"):
            from byte.manager.eviction.cost_aware import CostAwareCacheEviction
            self._delegate = CostAwareCacheEviction(
                maxsize=maxsize, clean_size=clean_size, on_evict=on_evict,
                **{k: v for k, v in kwargs.items() if k in ("default_score", "recency_half_life_s")},
            )
            self._cache = None
            return

        self._delegate = None
        if self._policy == "LRU":
            self._cache = cachetools.LRUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "LFU":
            self._cache = cachetools.LFUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "FIFO":
            self._cache = cachetools.FIFOCache(maxsize=maxsize, **kwargs)
        elif self._policy == "RR":
            self._cache = cachetools.RRCache(maxsize=maxsize, **kwargs)
        else:
            raise ValueError(f"Unknown policy {policy}")

        self._cache.popitem = popitem_wrapper(self._cache.popitem, on_evict, clean_size)

    def put(self, objs: list[Any]) -> None:
        if self._delegate is not None:
            self._delegate.put(objs)
            return
        for obj in objs:
            self._cache[obj] = True

    def get(self, obj: Any) -> Any:
        if self._delegate is not None:
            return self._delegate.get(obj)
        return self._cache.get(obj)

    def put_with_score(self, key: Any, score: float) -> None:
        """Only meaningful when policy == COST_AWARE."""
        if self._delegate is not None and hasattr(self._delegate, "put_with_score"):
            self._delegate.put_with_score(key, score)
        else:
            self.put([key])

    @property
    def policy(self) -> str:
        return self._policy
