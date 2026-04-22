from abc import ABC
from typing import Any

from byte.manager.eviction.distributed_cache import DistributedEviction
from byte.utils import lazy_optional_module
from byte.utils.log import byte_log

redis = lazy_optional_module("redis", package="redis")
redis_om = lazy_optional_module("redis_om", package="redis-om")


class RedisCacheEviction(DistributedEviction, ABC):
    """Distributed cache eviction strategy backed by Redis.

    :param host: Redis host.
    :param port: Redis port.
    :param policy: Redis eviction policy (allkeys-lru, volatile-lru, etc.).
        See https://redis.io/docs/reference/eviction/ for options.
    :param maxmemory: Redis maxmemory limit string (e.g. ``"256mb"``).
    :param global_key_prefix: Namespace prefix for all managed keys.
    :param ttl: TTL in seconds; each read extends the expiry (sliding TTL).
    :param maxmemory_samples: Number of keys sampled per eviction cycle.
    :param kwargs: Forwarded to ``redis_om.get_redis_connection``.
    """

    def __init__(
        self,
        host="localhost",
        port=6379,
        maxmemory: str | None = None,
        policy: str | None = None,
        global_key_prefix="byte",
        ttl: int | None = None,
        maxmemory_samples: int | None = None,
        **kwargs,
    ) -> None:
        self._redis = redis_om.get_redis_connection(host=host, port=port, **kwargs)
        if maxmemory:
            self._redis.config_set("maxmemory", maxmemory)
        if maxmemory_samples:
            self._redis.config_set("maxmemory-samples", maxmemory_samples)
        if policy:
            self._redis.config_set("maxmemory-policy", policy)
            self._policy = policy.lower()

        self._global_key_prefix = global_key_prefix
        self._ttl = ttl

    def _create_key(self, key: str) -> str:
        return f"{self._global_key_prefix}:evict:{key}"

    def put(self, objs: list[str], expire=False) -> None:
        ttl = self._ttl if expire else None
        for key in objs:
            self._redis.set(self._create_key(key), "True", ex=ttl)

    def get(self, obj: str) -> Any | None:
        key = self._create_key(obj)
        try:
            value = self._redis.get(key)
            # sliding TTL: extend expiry on each read to implement LRU-style eviction
            if self._ttl:
                self._redis.expire(key, self._ttl)
            return value
        except redis.RedisError:
            byte_log.warning("Error getting key %s from cache", obj)
            return None

    @property
    def policy(self) -> str:
        return self._policy
