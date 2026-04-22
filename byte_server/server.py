"""Byte server facade and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import byte.adapter
from byte import PRODUCT_NAME, PRODUCT_SHORT_NAME, Cache, Config, __version__, cache
from byte._backends import openai as default_backend
from byte.adapter.api import (
    clear_router_alias_registry,
    init_similar_cache,
    init_similar_cache_from_config,
    register_router_alias,
    router_registry_summary,
)
from byte.h2o import h2o_runtime_stats
from byte.mcp_gateway import MCPGateway
from byte.security import (
    DEFAULT_SECURITY_MAX_ARCHIVE_BYTES,
    DEFAULT_SECURITY_MAX_ARCHIVE_MEMBERS,
    DEFAULT_SECURITY_MAX_REQUEST_BYTES,
    DEFAULT_SECURITY_MAX_UPLOAD_BYTES,
)
from byte.utils import import_fastapi, import_pydantic, import_starlette
from byte_server._control_plane import ControlPlaneRuntime
from byte_server._server_gateway import (
    _active_cache as _gateway_active_cache,
)
from byte_server._server_gateway import (
    _configure_server_telemetry as _gateway_configure_server_telemetry,
)
from byte_server._server_gateway import (
    _env_provider_keys as _gateway_env_provider_keys,
)
from byte_server._server_gateway import (
    _init_gateway_cache as _gateway_init_gateway_cache,
)
from byte_server._server_gateway import (
    _model_can_run_without_credentials as _gateway_model_can_run_without_credentials,
)
from byte_server._server_gateway import (
    _server_backend_api_key as _gateway_server_backend_api_key,
)
from byte_server._server_gateway import (
    _should_use_routed_gateway as _gateway_should_use_routed_gateway,
)
from byte_server._server_lifespan import build_server_lifespan
from byte_server._server_routes_cache import register_cache_routes
from byte_server._server_routes_chat import register_chat_routes
from byte_server._server_routes_control import register_config_routes, register_control_routes
from byte_server._server_routes_mcp import register_mcp_routes
from byte_server._server_routes_proxy import register_proxy_routes
from byte_server._server_security import (
    _apply_security_overrides,
    register_security_middleware,
)
from byte_server._server_security import (
    _readiness_payload as _security_readiness_payload,
)
from byte_server._server_state import ServerRuntimeState, ServerServices
from byte_server.limits import RequestGuardRuntime
from byte_server.models import (
    CacheData,
    FeedbackData,
    MCPToolCall,
    MCPToolRegistration,
    MemoryArtifactExportData,
    MemoryArtifactImportData,
    MemoryImportData,
    WarmData,
)
from byte_server.telemetry import TelemetryRuntime, TelemetrySettings

import_fastapi()
import_pydantic()
import_starlette()

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

gateway_cache: Cache | None = None
cache_dir = ""
cache_file_key = ""
gateway_mode = "backend"
gateway_cache_mode = "exact"
gateway_routes: dict[str, list[str]] = {}
telemetry_runtime = None
control_plane_runtime = None
mcp_gateway = MCPGateway()
request_guard_runtime = RequestGuardRuntime()
runtime_state = ServerRuntimeState()
byte_adapter = byte.adapter
BYTE_GATEWAY_ROOT = "/byte/gateway"
BYTE_MCP_ROOT = "/byte/mcp"
_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
services = ServerServices(
    get_runtime_state=lambda: sync_runtime_state(),
    get_run_in_threadpool=lambda: run_in_threadpool,
    get_h2o_runtime_stats=lambda: h2o_runtime_stats(),
    default_backend=default_backend,
    router_registry_summary=router_registry_summary,
    gateway_root=BYTE_GATEWAY_ROOT,
    mcp_root=BYTE_MCP_ROOT,
    prometheus_content_type=_PROMETHEUS_CONTENT_TYPE,
)
app = FastAPI(
    title=PRODUCT_SHORT_NAME,
    version=__version__,
    description=f"{PRODUCT_NAME} gateway and cache server.",
    lifespan=build_server_lifespan(services),
)

# ── CORS middleware (module-level so uvicorn-driven servers are covered) ──
# Configurable via env: BYTE_CORS_ORIGINS (comma-separated) and BYTE_CORS_ALLOW_CREDENTIALS.
# Defaults to permissive so demo.html works out of the box; restrict in production.
try:
    from starlette.middleware.cors import CORSMiddleware as _CORSMiddleware

    _raw_origins = str(os.environ.get("BYTE_CORS_ORIGINS", "") or "").strip()
    _cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]
    _cors_creds = str(os.environ.get("BYTE_CORS_ALLOW_CREDENTIALS", "") or "").lower() in ("1", "true", "yes")
    if "*" in _cors_origins and _cors_creds:
        _cors_creds = False  # browsers reject this combination
    app.add_middleware(
        _CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=_cors_creds,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization", "Content-Type", "Accept",
            "X-Byte-Admin-Token", "X-Byte-Provider", "X-Byte-Provider-Base",
        ],
        expose_headers=["X-Byte-Cache-Hit", "X-Byte-Provider", "X-Byte-Model"],
    )
except Exception:  # pragma: no cover - defensive
    pass

_APP_CONFIGURED = False

__all__ = [
    "CacheData",
    "FeedbackData",
    "MCPToolCall",
    "MCPToolRegistration",
    "MemoryArtifactExportData",
    "MemoryArtifactImportData",
    "MemoryImportData",
    "WarmData",
    "app",
    "create_app",
    "main",
]


def sync_runtime_state() -> ServerRuntimeState:
    """Synchronize singleton route state with the active cache and runtimes."""

    runtime_state.gateway_cache = gateway_cache
    runtime_state.cache_dir = cache_dir
    runtime_state.cache_file_key = cache_file_key
    runtime_state.gateway_mode = gateway_mode
    runtime_state.gateway_cache_mode = gateway_cache_mode
    runtime_state.gateway_routes = gateway_routes
    runtime_state.telemetry_runtime = telemetry_runtime
    runtime_state.mcp_gateway = mcp_gateway
    runtime_state.request_guard_runtime = request_guard_runtime
    runtime_state.control_plane_runtime = control_plane_runtime
    return runtime_state


def _active_cache() -> Any:
    return _gateway_active_cache(services)


def _configure_server_telemetry(args: Any) -> None:
    global telemetry_runtime
    _gateway_configure_server_telemetry(
        services,
        args,
        telemetry_runtime_cls=TelemetryRuntime,
        telemetry_settings_cls=TelemetrySettings,
        readiness_provider=lambda: _security_readiness_payload(services),
    )
    telemetry_runtime = runtime_state.telemetry_runtime


def _env_provider_keys() -> dict[str, Any]:
    return _gateway_env_provider_keys()


def _init_gateway_cache(
    mode: str,
    cache_dir: str,
    cache_obj: Cache | None = None,
    config: Config | None = None,
) -> Cache:
    return _gateway_init_gateway_cache(mode, cache_dir, cache_obj=cache_obj, config=config)


def _model_can_run_without_credentials(model_name: str) -> bool:
    return _gateway_model_can_run_without_credentials(services, model_name)


def _readiness_payload() -> tuple[bool, dict[str, Any]]:
    return _security_readiness_payload(services)


def _server_backend_api_key() -> str | None:
    return _gateway_server_backend_api_key()


def _should_use_routed_gateway(model_name: str) -> bool:
    return _gateway_should_use_routed_gateway(services, model_name)


def _register_app_components() -> None:
    global _APP_CONFIGURED
    if _APP_CONFIGURED:
        return
    register_security_middleware(app, services)
    register_cache_routes(app, services)
    register_chat_routes(app, services)
    register_proxy_routes(app, services)
    register_mcp_routes(app, services)
    register_control_routes(app, services)
    register_config_routes(app, services)
    _APP_CONFIGURED = True


def _security_config_kwargs(args: Any, allowed_egress_hosts: list[str]) -> dict[str, Any]:
    return dict(
        enable_token_counter=False,
        compliance_profile=args.compliance_profile,
        security_mode=args.security_mode,
        security_encryption_key=args.security_encryption_key,
        security_admin_token=args.security_admin_token,
        security_audit_log_path=args.security_audit_log,
        security_export_root=args.security_export_root,
        security_require_https=args.security_require_https,
        security_trust_proxy_headers=args.security_trust_proxy_headers,
        security_allow_provider_host_override=args.security_allow_provider_host_override,
        security_allowed_egress_hosts=allowed_egress_hosts,
        security_max_request_bytes=args.security_max_request_bytes,
        security_max_upload_bytes=args.security_max_upload_bytes,
        security_max_archive_bytes=args.security_max_archive_bytes,
        security_max_archive_members=args.security_max_archive_members,
        security_rate_limit_public_per_minute=args.security_rate_limit_public_per_minute,
        security_rate_limit_admin_per_minute=args.security_rate_limit_admin_per_minute,
        security_max_inflight_public=args.security_max_inflight_public,
        security_max_inflight_admin=args.security_max_inflight_admin,
    )


def create_app() -> FastAPI:
    """Create and return the configured Byte FastAPI application."""

    _register_app_components()
    sync_runtime_state()
    return app


def main() -> None:
    """Run the Byte server from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--host", default="localhost")
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("-d", "--cache-dir", default="byte_data")
    parser.add_argument("-k", "--cache-file-key", default="")
    parser.add_argument("-f", "--cache-config-file", default=None)
    parser.add_argument("-g", "--gateway", type=bool, default=False)
    parser.add_argument("-gf", "--gateway-cache-config-file", default=None)
    parser.add_argument("--gateway-cache-mode", default="normalized", choices=["semantic", "exact", "normalized", "hybrid"])
    parser.add_argument("--gateway-enable-routing", action="store_true")
    parser.add_argument("--gateway-cheap-route-target", default=None)
    parser.add_argument("--gateway-expensive-route-target", default=None)
    parser.add_argument("--gateway-tool-route-target", default=None)
    parser.add_argument("--gateway-default-route-target", default=None)
    parser.add_argument("--gateway-coder-route-target", default=None)
    parser.add_argument("--gateway-reasoning-route-target", default=None)
    parser.add_argument("--gateway-verifier-route-target", default=None)
    parser.add_argument("--gateway-mode", default="backend", choices=["backend", "adaptive"])
    parser.add_argument("--router-aliases-file", default=None)
    parser.add_argument("--routing-strategy", default="priority", choices=["priority", "round_robin", "simple_shuffle", "latency", "cost", "health_weighted"])
    parser.add_argument("--routing-retries", type=int, default=0)
    parser.add_argument("--routing-backoff-ms", type=float, default=0.0)
    parser.add_argument("--routing-cooldown-s", type=float, default=15.0)
    parser.add_argument("--compliance-profile", default=None, choices=["hipaa", "soc2"])
    parser.add_argument("--security-mode", action="store_true")
    parser.add_argument("--security-encryption-key", default=None)
    parser.add_argument("--security-admin-token", default=None)
    parser.add_argument("--security-audit-log", default=None)
    parser.add_argument("--security-export-root", default=None)
    parser.add_argument("--security-require-https", action="store_true")
    parser.add_argument("--security-trust-proxy-headers", action="store_true")
    parser.add_argument("--security-allow-provider-host-override", action="store_true")
    parser.add_argument("--security-allowed-egress-hosts", default="")
    parser.add_argument("--security-max-request-bytes", type=int, default=DEFAULT_SECURITY_MAX_REQUEST_BYTES)
    parser.add_argument("--security-max-upload-bytes", type=int, default=DEFAULT_SECURITY_MAX_UPLOAD_BYTES)
    parser.add_argument("--security-max-archive-bytes", type=int, default=DEFAULT_SECURITY_MAX_ARCHIVE_BYTES)
    parser.add_argument("--security-max-archive-members", type=int, default=DEFAULT_SECURITY_MAX_ARCHIVE_MEMBERS)
    parser.add_argument("--security-rate-limit-public-per-minute", type=int, default=0)
    parser.add_argument("--security-rate-limit-admin-per-minute", type=int, default=0)
    parser.add_argument("--security-max-inflight-public", type=int, default=0)
    parser.add_argument("--security-max-inflight-admin", type=int, default=0)
    parser.add_argument("--otel-enabled", action="store_true")
    parser.add_argument("--otel-endpoint", default=None)
    parser.add_argument("--otel-protocol", default=None, choices=["http/protobuf", "grpc"])
    parser.add_argument("--otel-headers", default="")
    parser.add_argument("--otel-insecure", action="store_true")
    parser.add_argument("--otel-disable-traces", action="store_true")
    parser.add_argument("--otel-disable-metrics", action="store_true")
    parser.add_argument("--otel-export-interval-ms", type=int, default=5000)
    parser.add_argument("--otel-service-namespace", default="byteai")
    parser.add_argument("--otel-environment", default="")
    parser.add_argument("--otel-resource-attributes", default="")
    parser.add_argument("--datadog-enabled", action="store_true")
    parser.add_argument("--datadog-agent-host", default=None)
    parser.add_argument("--datadog-service", default=None)
    parser.add_argument("--datadog-env", default=None)
    parser.add_argument("--datadog-version", default=None)
    parser.add_argument("--cors-origins", default="")
    parser.add_argument("--cors-allow-credentials", action="store_true")
    parser.add_argument(
        "--control-plane-db",
        default=os.getenv("BYTE_CONTROL_PLANE_DB", "byte_control_plane.db"),
    )
    parser.add_argument(
        "--control-plane-worker-url",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--memory-service-url",
        default=os.getenv("BYTE_MEMORY_SERVICE_URL", ""),
    )
    parser.add_argument(
        "--internal-auth-token",
        default=os.getenv("BYTE_INTERNAL_TOKEN", ""),
    )
    parser.add_argument("--replay-enabled", action="store_true")
    parser.add_argument("--replay-sample-rate", type=float, default=0.05)
    args = parser.parse_args()

    global cache_dir, cache_file_key, gateway_cache, gateway_mode, gateway_cache_mode, gateway_routes, telemetry_runtime, control_plane_runtime
    allowed_egress_hosts = [item.strip() for item in str(args.security_allowed_egress_hosts or "").split(",") if item.strip()]
    base_security_config = Config(**_security_config_kwargs(args, allowed_egress_hosts))

    if args.cache_config_file:
        init_conf = init_similar_cache_from_config(config_dir=args.cache_config_file)
        cache_dir = init_conf.get("storage_config", {}).get("data_dir", "")
        _apply_security_overrides(cache, base_security_config)
    else:
        init_similar_cache(args.cache_dir, config=base_security_config)
        cache_dir = args.cache_dir
    cache_file_key = args.cache_file_key

    if args.gateway:
        gateway_cache = Cache()
        gateway_mode = args.gateway_mode
        gateway_routes = {}
        clear_router_alias_registry()
        if args.router_aliases_file:
            with open(args.router_aliases_file, encoding="utf-8") as alias_file:
                gateway_routes = json.load(alias_file) or {}
            for alias_name, targets in gateway_routes.items():
                register_router_alias(alias_name, list(targets or []))
        if args.gateway_cache_config_file:
            init_similar_cache_from_config(config_dir=args.gateway_cache_config_file, cache_obj=gateway_cache)
            _apply_security_overrides(gateway_cache, base_security_config)
        else:
            gateway_config = Config(
                **_security_config_kwargs(args, allowed_egress_hosts),
                model_routing=args.gateway_enable_routing,
                routing_cheap_model=args.gateway_cheap_route_target,
                routing_expensive_model=args.gateway_expensive_route_target,
                routing_tool_model=args.gateway_tool_route_target,
                routing_default_model=args.gateway_default_route_target,
                routing_coder_model=args.gateway_coder_route_target,
                routing_reasoning_model=args.gateway_reasoning_route_target,
                routing_verifier_model=args.gateway_verifier_route_target,
                routing_model_aliases=gateway_routes,
                routing_strategy=args.routing_strategy,
                routing_retry_attempts=args.routing_retries,
                routing_retry_backoff_ms=args.routing_backoff_ms,
                routing_cooldown_seconds=args.routing_cooldown_s,
            )
            _init_gateway_cache(mode=args.gateway_cache_mode, cache_dir="byte_gateway_cache", cache_obj=gateway_cache, config=gateway_config)
            gateway_cache_mode = (args.gateway_cache_mode or "exact").lower()

        # NB: CORS middleware is now registered at module load time (above) so
        # uvicorn-driven servers also get it. --cors-origins / --cors-allow-credentials
        # CLI flags are honoured via BYTE_CORS_ORIGINS / BYTE_CORS_ALLOW_CREDENTIALS env vars
        # set from args below, before the app starts serving real traffic.
        if args.cors_origins:
            os.environ["BYTE_CORS_ORIGINS"] = str(args.cors_origins)
        if args.cors_allow_credentials:
            os.environ["BYTE_CORS_ALLOW_CREDENTIALS"] = "1"

    control_plane_runtime = ControlPlaneRuntime(
        db_path=str(args.control_plane_db or "byte_control_plane.db"),
        worker_urls=list(args.control_plane_worker_url or []),
        memory_service_url=str(args.memory_service_url or ""),
        internal_auth_token=str(args.internal_auth_token or ""),
        replay_enabled=bool(args.replay_enabled),
        replay_sample_rate=float(args.replay_sample_rate or 0.05),
    )
    create_app()
    _configure_server_telemetry(args)
    telemetry_runtime = runtime_state.telemetry_runtime
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


create_app()

# Also wire a default ControlPlaneRuntime when the module is loaded outside of
# main() (e.g. `uvicorn byte_server.server:app`). Without this, /byte/control/*
# endpoints return 503 because runtime_state.control_plane_runtime is None.
if control_plane_runtime is None:
    try:
        control_plane_runtime = ControlPlaneRuntime(
            db_path=os.environ.get("BYTE_CONTROL_PLANE_DB", "byte_control_plane.db"),
            worker_urls=[],
            memory_service_url=os.environ.get("BYTE_MEMORY_SERVICE_URL", ""),
            internal_auth_token=os.environ.get("BYTE_INTERNAL_AUTH_TOKEN", ""),
            replay_enabled=bool(os.environ.get("BYTE_REPLAY_ENABLED", "")),
            replay_sample_rate=float(os.environ.get("BYTE_REPLAY_SAMPLE_RATE", "0.05")),
        )
        runtime_state.control_plane_runtime = control_plane_runtime
    except Exception as _cp_err:  # pragma: no cover - defensive
        import logging as _logging
        _logging.getLogger(__name__).warning("Could not initialise default ControlPlaneRuntime: %s", _cp_err)


# Also wire a default gateway cache when the module is loaded outside of
# main() (e.g. `uvicorn byte_server.server:app`). Without this,
# /v1/chat/completions and /byte/gateway/* endpoints return 500 because
# runtime_state.gateway_cache is None.
if gateway_cache is None:
    try:
        # "adaptive" mode lets byte_adapter dispatch to the right provider (OpenAI,
        # DeepSeek, Anthropic, Groq, …) based on the `model` name in each request.
        # "backend" would force every request through the single hard-coded OpenAI
        # backend, which fails for any non-OpenAI key.
        _gw_mode       = os.environ.get("BYTE_GATEWAY_MODE", "adaptive")
        _gw_cache_mode = os.environ.get("BYTE_GATEWAY_CACHE_MODE", "normalized")
        gateway_cache  = Cache()
        _init_gateway_cache(
            mode=_gw_cache_mode,
            cache_dir="byte_gateway_cache",
            cache_obj=gateway_cache,
        )
        gateway_mode       = _gw_mode
        gateway_cache_mode = _gw_cache_mode
        runtime_state.gateway_cache       = gateway_cache
        runtime_state.gateway_mode        = gateway_mode
        runtime_state.gateway_cache_mode  = gateway_cache_mode
    except Exception as _gw_err:  # pragma: no cover - defensive
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Could not initialise default gateway cache: %s", _gw_err
        )


if __name__ == "__main__":
    main()
