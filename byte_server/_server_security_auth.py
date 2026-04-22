"""Authentication and security-setting helpers for the Byte server."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import HTTPException, Request

from byte import Config
from byte.security import short_digest
from byte_server._server_gateway import (
    _env_provider_keys,
    _model_can_run_without_credentials,
    _server_backend_api_key,
    _should_use_routed_gateway,
)
from byte_server._server_state import ServerServices


def _cfg(services: ServerServices) -> Config:
    config = getattr(services.active_cache(), "config", None)
    return config if isinstance(config, Config) else Config(enable_token_counter=False)


def _flag(value: bool) -> int:
    return 1 if value else 0


def _request_host(request: Request) -> str:
    host = str(request.headers.get("host", "") or "").split(":", 1)[0].strip().lower()
    if host:
        return host
    if request.client and request.client.host:
        return str(request.client.host).strip().lower()
    return ""


def _request_actor(request: Request) -> str:
    for key in ("x-byte-actor", "authorization", "x-byte-admin-token"):
        value = str(request.headers.get(key, "") or "").strip()
        if value:
            return short_digest(value)
    return short_digest(_request_host(request) or "anonymous")


def _request_scheme(services: ServerServices, request: Request) -> str:
    config = _cfg(services)
    trust_proxy = bool(getattr(config, "security_trust_proxy_headers", False))
    if trust_proxy:
        forwarded = (
            str(request.headers.get("x-forwarded-proto", "") or "").split(",", 1)[0].strip().lower()
        )
        if forwarded:
            return forwarded
    return str(request.url.scheme or "").lower()


def _security_enabled(services: ServerServices) -> bool:
    return bool(getattr(_cfg(services), "security_mode", False))


def _admin_token(services: ServerServices) -> str | None:
    token = getattr(_cfg(services), "security_admin_token", None) or os.getenv("BYTE_ADMIN_TOKEN")
    token = str(token or "").strip()
    return token or None


def _audit_log_path(services: ServerServices) -> Path | None:
    path = getattr(_cfg(services), "security_audit_log_path", None) or os.getenv("BYTE_AUDIT_LOG_PATH")
    return Path(path).expanduser().resolve() if path else None


def _security_limit(services: ServerServices, name: str, default: int) -> int:
    return int(getattr(_cfg(services), name, None) or default)


def _allowed_egress_hosts(services: ServerServices) -> list[str]:
    configured = getattr(_cfg(services), "security_allowed_egress_hosts", None) or []
    if configured:
        return [str(item).strip().lower() for item in configured if str(item).strip()]
    raw_env = str(os.getenv("BYTE_ALLOWED_EGRESS_HOSTS", "") or "").strip()
    return [item.strip().lower() for item in raw_env.split(",") if item.strip()] if raw_env else []


def _extract_admin_token(request: Request) -> str:
    token = str(request.headers.get("x-byte-admin-token", "") or "").strip()
    if token:
        return token
    auth_header = str(request.headers.get("authorization", "") or "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header.partition(" ")[2].strip()
    return ""

def _admin_auth_required(services: ServerServices) -> bool:
    config = _cfg(services)
    return bool(_security_enabled(services) or getattr(config, "security_require_admin_auth", False) or _admin_token(services))


def _require_admin(services: ServerServices, request: Request, action: str) -> None:
    from ._server_security_audit import _audit_event

    if not _admin_auth_required(services):
        return
    expected = _admin_token(services)
    if not expected:
        _audit_event(services, request, action, status="denied", metadata={"reason": "admin_token_not_configured"})
        raise HTTPException(status_code=503, detail="ByteAI Cache admin auth is required by security policy but no admin token is configured.")
    provided = _extract_admin_token(request)
    if not provided or not hmac.compare_digest(provided, expected):
        _audit_event(services, request, action, status="denied", metadata={"reason": "invalid_admin_token"})
        raise HTTPException(status_code=403, detail="ByteAI Cache admin token required.")

def _chat_uses_server_credentials(services: ServerServices, request: Request, model_name: str) -> bool:
    if _model_can_run_without_credentials(services, model_name):
        return False
    if str(request.headers.get("authorization", "") or "").strip():
        return False
    if _should_use_routed_gateway(services, model_name):
        return bool(_env_provider_keys())
    return bool(_server_backend_api_key())
