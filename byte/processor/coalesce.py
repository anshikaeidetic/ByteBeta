"""Request coalescing — deduplicate concurrent identical LLM calls.

When multiple callers submit the same query simultaneously, only one LLM
call is made.  The remaining callers block (or ``await``) until the first
call completes, then all receive the same result.

This is completely transparent to callers and never alters the output.
"""

import threading
from typing import Any


class RequestCoalescer:
    """Thread-safe in-flight request tracker.

    Usage in the ``adapt()`` pipeline::

        coalescer = RequestCoalescer()

        # Before calling LLM:
        result = coalescer.get_if_inflight(key)
        if result is not _SENTINEL:
            return result           # free cache hit from another thread

        # Call LLM normally…
        llm_result = llm_handler(...)

        # After LLM returns:
        coalescer.complete(key, llm_result)
        return llm_result
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        # key → _Waiter
        self._inflight: dict[str, _Waiter] = {}

    def get_if_inflight(self, key: str, timeout: float = 30.0) -> Any:
        """Return the result if another thread is already processing *key*.

        If no in-flight request exists for *key*, register one and return
        the sentinel ``_SENTINEL`` so the caller knows it must proceed with
        the actual LLM call.

        :param key: the cache/embedding key for the request
        :param timeout: max seconds to wait for the in-flight result
        :return: LLM result or ``_SENTINEL``
        """
        if not self._enabled:
            return _SENTINEL

        with self._lock:
            if key in self._inflight:
                waiter = self._inflight[key]
            else:
                # We are the first — register ourselves
                self._inflight[key] = _Waiter()
                return _SENTINEL

        # Another thread is already processing this key — wait for it
        result = waiter.wait(timeout)
        if result is _SENTINEL:
            # Timed out — proceed with our own LLM call
            return _SENTINEL
        return result

    def complete(self, key: str, result: Any) -> None:
        """Signal that the LLM call for *key* has finished.

        Wakes all threads waiting on this key and removes the in-flight entry.

        :param key: the cache/embedding key
        :param result: the LLM result to broadcast
        """
        if not self._enabled:
            return

        with self._lock:
            waiter = self._inflight.pop(key, None)

        if waiter is not None:
            waiter.set(result)

    def cancel(self, key: str) -> None:
        """Cancel an in-flight request (e.g. on LLM error).

        Releases waiters with ``_SENTINEL`` so they fall through to their
        own LLM calls.
        """
        if not self._enabled:
            return

        with self._lock:
            waiter = self._inflight.pop(key, None)

        if waiter is not None:
            waiter.set(_SENTINEL)

    @property
    def inflight_count(self) -> int:
        """Number of currently in-flight requests."""
        with self._lock:
            return len(self._inflight)


class _Waiter:
    """Internal synchronisation primitive — one producer, N consumers."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._result: Any = _SENTINEL
        self._done = False

    def wait(self, timeout: float) -> Any:
        with self._condition:
            if not self._done:
                self._condition.wait(timeout=timeout)
            return self._result

    def set(self, value: Any) -> None:
        with self._condition:
            self._result = value
            self._done = True
            self._condition.notify_all()


class _SentinelType:
    """Unique sentinel to distinguish 'no result yet' from ``None``."""

    def __repr__(self) -> str:
        return "<SENTINEL>"


_SENTINEL = _SentinelType()
