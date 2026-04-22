"""Process lifecycle helpers for the Byte server."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from byte_server._server_state import ServerServices


def build_server_lifespan(
    services: ServerServices,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Return the FastAPI lifespan handler for process-scoped runtime cleanup."""

    @asynccontextmanager
    async def server_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            runtime = services.runtime_state()
            if runtime.telemetry_runtime is not None:
                runtime.telemetry_runtime.shutdown()
                runtime.telemetry_runtime = None

    return server_lifespan


__all__ = ["build_server_lifespan"]
