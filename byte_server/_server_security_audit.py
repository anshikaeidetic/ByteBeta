"""Audit logging and route-error helpers for the Byte server."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request

from byte.security import short_digest
from byte.utils.error import ByteErrorCode, classify_error
from byte.utils.log import byte_log, log_byte_error
from byte_server._server_state import ServerServices

from ._server_security_auth import (
    _audit_log_path,
    _request_actor,
    _request_host,
    _request_scheme,
)


def _audit_event(
    services: ServerServices,
    request: Request,
    action: str,
    *,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = _audit_log_path(services)
    if target is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "status": status,
        "path": str(request.url.path),
        "method": request.method,
        "scheme": _request_scheme(services, request),
        "host_digest": short_digest(_request_host(request) or ""),
        "actor_digest": _request_actor(request),
        "metadata": metadata or {},
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")

def _raise_route_error(
    services: ServerServices,
    request: Request,
    action: str,
    exc: Exception,
    *,
    public_detail: str,
) -> None:
    if isinstance(exc, HTTPException):
        _audit_event(
            services,
            request,
            action,
            status="denied" if exc.status_code < 500 else "error",
            metadata={"detail": str(exc.detail), "error_code": ByteErrorCode.SERVER_ROUTE.value},
        )
        raise exc

    # Surface upstream provider errors (auth failures, rate limits, etc.) with
    # their real HTTP status and message so the operator sees what actually
    # failed — rather than a generic "Byte gateway request failed."
    # Covers both ProviderRequestError (our own class) AND wrapped SDK errors
    # (openai.AuthenticationError, anthropic.APIError, etc.) which carry a
    # status_code / http_status / response attribute after wrap_error().
    from byte.utils.error import (
        ProviderRequestError as _ProviderRequestError,  # pylint: disable=import-outside-toplevel
    )
    _raw_status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "http_status", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
    )
    if isinstance(exc, _ProviderRequestError) or (_raw_status and 400 <= int(_raw_status) <= 599):
        try:
            _status = int(_raw_status or 502)
        except Exception:  # pragma: no cover
            _status = 502
        if _status < 400 or _status > 599:
            _status = 502
        # Extract a human-readable detail. SDK errors have .body/.message/.response.
        detail = str(exc)
        body = getattr(exc, "body", None)
        if isinstance(body, dict) and body.get("error"):
            err = body["error"]
            if isinstance(err, dict) and err.get("message"):
                detail = str(err["message"])
        msg = getattr(exc, "message", None)
        if not detail and msg:
            detail = str(msg)
        _audit_event(
            services,
            request,
            action,
            status="denied" if _status < 500 else "error",
            metadata={
                "provider": getattr(exc, "provider", ""),
                "provider_status": _status,
                "error_code": ByteErrorCode.PROVIDER_TRANSPORT.value if hasattr(ByteErrorCode, "PROVIDER_TRANSPORT") else "",
            },
        )
        raise HTTPException(status_code=_status, detail=detail) from exc

    info = classify_error(
        exc,
        code=getattr(exc, "code", None) or ByteErrorCode.SERVER_ROUTE,
        boundary="server.route",
        public_detail=public_detail,
    )
    log_byte_error(
        byte_log,
        logging.ERROR,
        "route handling failed",
        error=exc,
        code=info.code,
        boundary="server.route",
        stage=action,
    )
    _audit_event(
        services,
        request,
        action,
        status="error",
        metadata={
            "error_type": info.error_type,
            "error_code": info.code,
            "error_digest": short_digest(str(exc) or type(exc).__name__),
        },
    )
    raise HTTPException(status_code=500, detail=public_detail) from exc
