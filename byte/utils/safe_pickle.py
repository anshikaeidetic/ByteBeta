import io
import pickle
from typing import Any

import cachetools

from byte.utils.error import CacheError

MAP_CACHE_MAGIC = b"BYTE-MAP1\x00"


class RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, file, safe_globals: dict[tuple, Any]) -> None:
        super().__init__(file)
        self._safe_globals = safe_globals

    def find_class(self, module, name) -> Any:
        candidate = self._safe_globals.get((module, name))
        if candidate is None:
            raise CacheError(f"Refused to load unsafe cache payload class: {module}.{name}")
        return candidate


def _coerce_cache_container(loaded: Any, *, max_size: int | None = None) -> Any:
    if isinstance(loaded, cachetools.Cache):
        return loaded
    if isinstance(loaded, dict):
        cache_size = int(max_size or max(len(loaded), 1))
        container = cachetools.LRUCache(cache_size)
        container.update(loaded)
        return container
    raise CacheError("Unsupported cache payload in map data manager.")


def dump_map_cache(handle, payload: Any) -> None:
    handle.write(MAP_CACHE_MAGIC)
    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_map_cache(handle, *, safe_globals: dict[tuple, Any], max_size: int | None = None) -> Any:
    payload = handle.read()
    if not payload:
        cache_size = int(max_size or 1)
        return cachetools.LRUCache(cache_size)

    if payload.startswith(MAP_CACHE_MAGIC):
        body = payload[len(MAP_CACHE_MAGIC) :]
    else:
        # Read older restricted payloads while new flushes use the versioned header.
        body = payload

    loaded = RestrictedUnpickler(io.BytesIO(body), safe_globals).load()
    return _coerce_cache_container(loaded, max_size=max_size)
