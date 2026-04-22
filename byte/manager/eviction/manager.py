
# pylint: disable=import-outside-toplevel
from collections.abc import Callable
from typing import Any

from byte.utils.error import NotFoundError


class EvictionBase:
    """
    EvictionBase to evict the cache data.
    """

    def __init__(self) -> None:
        raise OSError(
            "EvictionBase is designed to be instantiated, "
            "please using the `EvictionBase.get(name, policy, maxsize, clean_size)`."
        )

    @staticmethod
    def get(
        name: str,
        policy: str = "LRU",
        maxsize: int = 1000,
        clean_size: int = 0,
        on_evict: Callable[[list[Any]], None] | None = None,
        **kwargs,
    ) -> Any:
        if not clean_size:
            clean_size = int(maxsize * 0.2)
        if name in "memory":
            from byte.manager.eviction.memory_cache import MemoryCacheEviction

            eviction_base = MemoryCacheEviction(policy, maxsize, clean_size, on_evict, **kwargs)
            return eviction_base
        if name == "redis":
            from byte.manager.eviction.redis_eviction import RedisCacheEviction

            if policy == "LRU":
                policy = None
            eviction_base = RedisCacheEviction(policy=policy, **kwargs)
            return eviction_base
        if name == "no_op_eviction":
            from byte.manager.eviction.distributed_cache import NoOpEviction

            eviction_base = NoOpEviction()
            return eviction_base

        else:
            raise NotFoundError("eviction base", name)
