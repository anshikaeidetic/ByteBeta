import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from byte.utils.error import ConcurrencyLimitError, RateLimitError

_ADMIN_PREFIXES = (
    "/byte/mcp",
    "/memory",
)
_ADMIN_PATHS = {
    "/put",
    "/get",
    "/flush",
    "/invalidate",
    "/clear",
    "/stats",
    "/warm",
    "/feedback",
    "/quality",
    "/metrics",
    "/download",
}


def request_category(path: str) -> str:
    normalized = str(path or "").strip()
    if normalized in _ADMIN_PATHS:
        return "admin"
    if any(normalized.startswith(prefix) for prefix in _ADMIN_PREFIXES):
        return "admin"
    return "public"


@dataclass
class RequestLease:
    category: str
    runtime: "RequestGuardRuntime"
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.runtime.release(self.category)
        self.released = True


class RequestGuardRuntime:
    def __init__(self, clock=None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._rate_windows: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._inflight: dict[str, int] = defaultdict(int)

    def enter(
        self,
        *,
        path: str,
        actor: str,
        public_rate_limit: int,
        admin_rate_limit: int,
        public_inflight_limit: int,
        admin_inflight_limit: int,
    ) -> RequestLease:
        category = request_category(path)
        now = float(self._clock())
        rate_limit = admin_rate_limit if category == "admin" else public_rate_limit
        inflight_limit = admin_inflight_limit if category == "admin" else public_inflight_limit
        with self._lock:
            if rate_limit > 0:
                bucket = self._rate_windows[(category, actor)]
                cutoff = now - 60.0
                while bucket and bucket[0] <= cutoff:
                    bucket.popleft()
                if len(bucket) >= rate_limit:
                    raise RateLimitError(
                        f"Byte {category} request rate limit exceeded. Retry later."
                    )
                bucket.append(now)
            if inflight_limit > 0 and self._inflight[category] >= inflight_limit:
                raise ConcurrencyLimitError(
                    f"Byte {category} request concurrency limit exceeded. Retry later."
                )
            if inflight_limit > 0:
                self._inflight[category] += 1
        return RequestLease(category=category, runtime=self)

    def release(self, category: str) -> None:
        with self._lock:
            if self._inflight.get(category, 0) > 0:
                self._inflight[category] -= 1

    def reset(self) -> None:
        with self._lock:
            self._rate_windows.clear()
            self._inflight.clear()
