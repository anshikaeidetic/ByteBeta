from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from byte.manager.eviction import redis_eviction as redis_eviction_module
from byte.manager.eviction.redis_eviction import RedisCacheEviction


class _FakeRedisError(Exception):
    pass


class _FakeRedisConnection:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.config_calls: list[tuple[str, object]] = []
        self.expire_calls: list[tuple[str, int]] = []
        self.fail_get = False

    def config_set(self, key: str, value) -> None:
        self.config_calls.append((key, value))

    def set(self, key: str, value: str, ex=None) -> None:
        self.values[key] = value
        self.expire_calls.append((key, ex))

    def get(self, key: str) -> object:
        if self.fail_get:
            raise _FakeRedisError("boom")
        return self.values.get(key)

    def expire(self, key: str, ttl: int) -> None:
        self.expire_calls.append((key, ttl))


def test_redis_cache_eviction_configures_connection_and_ttl_behavior() -> None:
    fake_connection = _FakeRedisConnection()

    fake_redis_om = SimpleNamespace(get_redis_connection=mock.Mock(return_value=fake_connection))
    with mock.patch.object(redis_eviction_module, "redis_om", fake_redis_om):
        eviction = RedisCacheEviction(
            host="redis",
            port=6380,
            maxmemory="32mb",
            policy="allkeys-lru",
            global_key_prefix="byte",
            ttl=30,
            maxmemory_samples=7,
        )

        eviction.put(["alpha"], expire=True)
        assert eviction.get("alpha") == "True"
        assert eviction.policy == "allkeys-lru"

    assert ("maxmemory", "32mb") in fake_connection.config_calls
    assert ("maxmemory-samples", 7) in fake_connection.config_calls
    assert ("maxmemory-policy", "allkeys-lru") in fake_connection.config_calls
    assert ("byte:evict:alpha", 30) in fake_connection.expire_calls


def test_redis_cache_eviction_returns_none_on_redis_error() -> None:
    fake_connection = _FakeRedisConnection()
    fake_connection.fail_get = True

    fake_redis_om = SimpleNamespace(get_redis_connection=mock.Mock(return_value=fake_connection))
    fake_redis = SimpleNamespace(RedisError=_FakeRedisError)
    with mock.patch.object(redis_eviction_module, "redis_om", fake_redis_om):
        with mock.patch.object(redis_eviction_module, "redis", fake_redis):
            eviction = RedisCacheEviction()
            assert eviction.get("missing") is None
