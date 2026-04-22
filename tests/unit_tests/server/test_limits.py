import pytest

from byte.utils.error import ConcurrencyLimitError, RateLimitError
from byte_server.limits import RequestGuardRuntime, request_category


def test_request_category_separates_public_and_admin_paths() -> None:
    assert request_category("/byte/gateway/chat") == "public"
    assert request_category("/byte/mcp/tools") == "admin"
    assert request_category("/memory/recent") == "admin"
    assert request_category("/metrics") == "admin"


def test_request_guard_runtime_enforces_rate_limit() -> None:
    ticks = iter([0.0, 1.0, 2.0])
    runtime = RequestGuardRuntime(clock=lambda: next(ticks))

    lease = runtime.enter(
        path="/byte/gateway/chat",
        actor="actor",
        public_rate_limit=1,
        admin_rate_limit=0,
        public_inflight_limit=0,
        admin_inflight_limit=0,
    )
    lease.release()

    with pytest.raises(RateLimitError):
        runtime.enter(
            path="/byte/gateway/chat",
            actor="actor",
            public_rate_limit=1,
            admin_rate_limit=0,
            public_inflight_limit=0,
            admin_inflight_limit=0,
        )


def test_request_guard_runtime_enforces_concurrency_limit() -> None:
    runtime = RequestGuardRuntime(clock=lambda: 0.0)

    lease = runtime.enter(
        path="/byte/gateway/chat",
        actor="actor",
        public_rate_limit=0,
        admin_rate_limit=0,
        public_inflight_limit=1,
        admin_inflight_limit=0,
    )
    with pytest.raises(ConcurrencyLimitError):
        runtime.enter(
            path="/byte/gateway/chat",
            actor="actor",
            public_rate_limit=0,
            admin_rate_limit=0,
            public_inflight_limit=1,
            admin_inflight_limit=0,
        )
    lease.release()
