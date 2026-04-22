from __future__ import annotations

import asyncio

from fastapi import FastAPI

from byte_server._server_lifespan import build_server_lifespan
from byte_server._server_state import ServerRuntimeState, ServerServices


class _DummyTelemetry:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _services(runtime: ServerRuntimeState) -> ServerServices:
    return ServerServices(
        get_runtime_state=lambda: runtime,
        get_run_in_threadpool=lambda: (lambda *args, **kwargs: None),
        get_h2o_runtime_stats=dict,
        default_backend=object(),
        router_registry_summary=dict,
        gateway_root="/byte/gateway",
        mcp_root="/byte/mcp",
        prometheus_content_type="text/plain",
    )


def test_server_lifespan_shuts_down_telemetry_once() -> None:
    runtime = ServerRuntimeState()
    telemetry = _DummyTelemetry()
    runtime.telemetry_runtime = telemetry
    lifespan = build_server_lifespan(_services(runtime))

    async def _run() -> None:
        async with lifespan(FastAPI()):
            assert runtime.telemetry_runtime is telemetry

    asyncio.run(_run())

    assert telemetry.shutdown_calls == 1
    assert runtime.telemetry_runtime is None
