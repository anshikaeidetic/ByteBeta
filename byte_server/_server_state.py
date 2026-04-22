"""Shared runtime state and service bindings for the Byte server."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from byte import Cache, cache
from byte.core import LazyCacheProxy
from byte.mcp_gateway import MCPGateway
from byte_server.limits import RequestGuardRuntime


@dataclass
class ServerRuntimeState:
    gateway_cache: Cache | None = None
    cache_dir: str = ""
    cache_file_key: str = ""
    gateway_mode: str = "backend"
    gateway_cache_mode: str = "exact"
    gateway_routes: dict[str, list[str]] = field(default_factory=dict)
    telemetry_runtime: Any = None
    mcp_gateway: MCPGateway = field(default_factory=MCPGateway)
    request_guard_runtime: RequestGuardRuntime = field(default_factory=RequestGuardRuntime)
    control_plane_runtime: Any = None


@dataclass(frozen=True)
class ServerServices:
    get_runtime_state: Callable[[], ServerRuntimeState]
    get_run_in_threadpool: Callable[[], Callable[..., Any]]
    get_h2o_runtime_stats: Callable[[], dict[str, Any]]
    default_backend: Any
    router_registry_summary: Callable[[], dict[str, Any]]
    gateway_root: str
    mcp_root: str
    prometheus_content_type: str
    base_cache: Cache | LazyCacheProxy = cache

    def runtime_state(self) -> ServerRuntimeState:
        return self.get_runtime_state()

    def active_cache(self) -> Cache | LazyCacheProxy:
        runtime = self.runtime_state()
        return runtime.gateway_cache if runtime.gateway_cache is not None else self.base_cache

    def run_in_threadpool(self) -> Callable[..., Any]:
        return self.get_run_in_threadpool()

    def h2o_runtime_stats(self) -> dict[str, Any]:
        return self.get_h2o_runtime_stats()

    def control_plane(self) -> Any:
        return self.runtime_state().control_plane_runtime


__all__ = ["ServerRuntimeState", "ServerServices"]
