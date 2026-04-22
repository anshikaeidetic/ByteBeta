"""Request, upload, archive, and proxy validation helpers for the Byte server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from fastapi import HTTPException, Request

from byte import Config
from byte.security import (
    DEFAULT_SECURITY_MAX_ARCHIVE_BYTES,
    DEFAULT_SECURITY_MAX_ARCHIVE_MEMBERS,
    DEFAULT_SECURITY_MAX_REQUEST_BYTES,
    DEFAULT_SECURITY_MAX_UPLOAD_BYTES,
    ensure_byte_limit,
    maybe_wrap_data_manager,
    validate_declared_content_length,
    validate_outbound_target,
)
from byte.utils.error import CacheError
from byte_server._server_state import ServerServices

from ._server_security_auth import (
    _allowed_egress_hosts,
    _cfg,
    _security_enabled,
    _security_limit,
)


def _resolve_artifact_path(services: ServerServices, path: str) -> str:
    root = getattr(_cfg(services), "security_export_root", None) or os.getenv("BYTE_SECURITY_EXPORT_ROOT")
    candidate = Path(path).expanduser()
    if not root:
        if _security_enabled(services):
            raise HTTPException(status_code=503, detail="Byte security mode requires a configured export root for artifact import and export.")
        return str(candidate.resolve())
    root_path = Path(root).expanduser().resolve()
    resolved = (root_path / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Artifact path must stay inside ByteAI Cache's configured export root.") from exc
    return str(resolved)


def _sanitize_proxy_request_payload(
    services: ServerServices, payload: dict[str, Any]
) -> dict[str, Any]:
    config = _cfg(services)
    request_payload = dict(payload or {})
    allow_override = bool(getattr(config, "security_allow_provider_host_override", False) or str(os.getenv("BYTE_ALLOW_PROVIDER_HOST_OVERRIDE", "")).strip().lower() in {"1", "true", "yes", "on"})
    for field_name in ("api_base", "base_url", "host"):
        value = request_payload.get(field_name)
        if value in (None, ""):
            continue
        if not allow_override:
            raise HTTPException(status_code=403, detail=f"Client-supplied provider host overrides are disabled by ByteAI Cache security policy: {field_name}.")
        try:
            request_payload[field_name] = validate_outbound_target(str(value), allowed_hosts=_allowed_egress_hosts(services))
        except CacheError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
    return request_payload


def _apply_security_overrides(cache_obj: Any, template: Config) -> None:
    if cache_obj is None or not getattr(cache_obj, "has_init", False):
        return
    config = getattr(cache_obj, "config", None)
    if not isinstance(config, Config):
        return
    for name in (
        "security_encryption_key",
        "security_admin_token",
        "security_audit_log_path",
        "security_export_root",
        "security_max_request_bytes",
        "security_max_upload_bytes",
        "security_max_archive_bytes",
        "security_max_archive_members",
        "security_rate_limit_public_per_minute",
        "security_rate_limit_admin_per_minute",
        "security_max_inflight_public",
        "security_max_inflight_admin",
    ):
        setattr(config, name, getattr(template, name))
    if template.compliance_profile:
        config.compliance_profile = template.compliance_profile
        config.security_mode = True
        config.security_redact_logs = True
        config.security_redact_reports = True
        config.security_redact_memory = True
        config.security_require_admin_auth = True
        config.security_disable_cache_file_endpoint = True
        config.security_encrypt_artifacts = True
        config.security_allow_provider_host_override = False
    if template.security_mode:
        config.security_mode = True
        config.security_require_admin_auth = True
        config.security_disable_cache_file_endpoint = True
    if template.security_require_https:
        config.security_require_https = True
    if template.security_trust_proxy_headers:
        config.security_trust_proxy_headers = True
    if template.security_allow_provider_host_override:
        config.security_allow_provider_host_override = True
    if template.security_allowed_egress_hosts:
        config.security_allowed_egress_hosts = list(template.security_allowed_egress_hosts)
    cache_obj.data_manager = maybe_wrap_data_manager(cache_obj.data_manager, config)


def _cache_file_disabled(services: ServerServices) -> bool:
    config = _cfg(services)
    return bool(_security_enabled(services) or getattr(config, "security_disable_cache_file_endpoint", False))


def _security_max_archive_bytes(services: ServerServices) -> int:
    return _security_limit(services, "security_max_archive_bytes", DEFAULT_SECURITY_MAX_ARCHIVE_BYTES)


def _security_max_archive_members(services: ServerServices) -> int:
    return _security_limit(services, "security_max_archive_members", DEFAULT_SECURITY_MAX_ARCHIVE_MEMBERS)


def _security_max_upload_bytes(services: ServerServices) -> int:
    return _security_limit(services, "security_max_upload_bytes", DEFAULT_SECURITY_MAX_UPLOAD_BYTES)


async def _read_json_object(
    services: ServerServices, request: Request
) -> dict[str, Any]:
    limit = _security_limit(services, "security_max_request_bytes", DEFAULT_SECURITY_MAX_REQUEST_BYTES)
    try:
        raw_length = str(request.headers.get("content-length", "") or "").strip()
        validate_declared_content_length(int(raw_length) if raw_length else None, limit=limit, label="Request body")
        body = await request.body()
        ensure_byte_limit(len(body), limit=limit, label="Request body")
        payload = json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be UTF-8 encoded JSON.") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    except CacheError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return payload


async def _read_upload_bytes(
    services: ServerServices, upload: Any, *, label: str
) -> bytes:
    try:
        payload = cast(bytes, await upload.read())
        ensure_byte_limit(
            (len(payload) if payload else 0),
            limit=_security_limit(
                services,
                "security_max_upload_bytes",
                DEFAULT_SECURITY_MAX_UPLOAD_BYTES,
            ),
            label=label,
        )
        return payload
    except CacheError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

def _extract_cache_skip_flag(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    request_payload = dict(payload or {})
    cache_skip = bool(request_payload.pop("cache_skip", False))
    if cache_skip:
        return request_payload, True
    messages = request_payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return request_payload, False
    for index in {0, len(messages) - 1}:
        message = messages[index]
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and "/cache_skip " in content:
            message["content"] = content.replace("/cache_skip ", "", 1)
            return request_payload, True
    return request_payload, False
