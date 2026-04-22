"""Security middleware and streaming-response protection for the Byte server."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from byte.utils.error import ByteErrorCode, ConcurrencyLimitError, RateLimitError
from byte.utils.log import byte_log, log_byte_error
from byte_server._server_state import ServerServices

from ._server_security_audit import _audit_event
from ._server_security_auth import (
    _cfg,
    _request_actor,
    _request_host,
    _request_scheme,
    _security_enabled,
)


def _wrap_streaming_response(
    response: StreamingResponse,
    lease: Any | None,
    *,
    finalize_request: Callable[..., None] | None = None,
) -> StreamingResponse:
    """Release the lease and finalize telemetry when the stream completes."""

    if getattr(response, "body_iterator", None) is None:
        if lease is not None:
            lease.release()
        if finalize_request is not None:
            finalize_request(error=None)
        return response

    original_iterator = response.body_iterator

    async def wrapped_iterator() -> AsyncIterator[Any]:
        stream_error: Exception | None = None
        try:
            async for chunk in original_iterator:
                yield chunk
        except Exception as exc:  # pragma: no cover - boundary path
            stream_error = exc
            raise
        finally:
            if lease is not None:
                lease.release()
            if finalize_request is not None:
                finalize_request(error=stream_error)

    response.body_iterator = wrapped_iterator()
    return response


def register_security_middleware(app: FastAPI, services: ServerServices) -> None:
    """Install request security, rate-limit, and telemetry middleware."""

    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next: Any) -> Any:
        config = _cfg(services)
        require_https = bool(_security_enabled(services) or getattr(config, "security_require_https", False) or str(os.getenv("BYTE_REQUIRE_HTTPS", "")).strip().lower() in {"1", "true", "yes", "on"})
        loopback = _request_host(request) in {"127.0.0.1", "::1", "localhost", "testserver"}
        if require_https and not loopback and _request_scheme(services, request) != "https":
            _audit_event(services, request, "transport.enforce_https", status="denied", metadata={"reason": "https_required"})
            return JSONResponse(status_code=403, content={"detail": "HTTPS is required by the active ByteAI Cache security policy."})

        request_start = time.perf_counter()
        runtime = services.runtime_state()
        telemetry_runtime = runtime.telemetry_runtime
        span = telemetry_runtime.start_request_span(request) if telemetry_runtime is not None else None
        lease = None
        response = None
        error = None
        stream_finalize_registered = False
        try:
            if request.method != "OPTIONS":
                lease = runtime.request_guard_runtime.enter(
                    path=str(request.url.path),
                    actor=_request_actor(request),
                    public_rate_limit=int(getattr(config, "security_rate_limit_public_per_minute", 0) or 0),
                    admin_rate_limit=int(getattr(config, "security_rate_limit_admin_per_minute", 0) or 0),
                    public_inflight_limit=int(getattr(config, "security_max_inflight_public", 0) or 0),
                    admin_inflight_limit=int(getattr(config, "security_max_inflight_admin", 0) or 0),
                )
            response = await call_next(request)
            if lease is not None:
                if isinstance(response, StreamingResponse) and getattr(response, "body_iterator", None) is not None:
                    lease_to_release = lease
                    status_code = int(response.status_code)
                    stream_finalize_registered = True

                    def _finalize_request(*, error: Exception | None = None) -> None:
                        runtime_state = services.runtime_state()
                        stream_telemetry = runtime_state.telemetry_runtime
                        if stream_telemetry is not None:
                            stream_telemetry.finish_request_span(
                                span,
                                request,
                                status_code=status_code,
                                duration_s=time.perf_counter() - request_start,
                                error=error,
                            )

                    response = _wrap_streaming_response(
                        response,
                        lease_to_release,
                        finalize_request=_finalize_request,
                    )
                else:
                    lease.release()
                lease = None
            return response
        except Exception as exc:  # noqa: BLE001, RUF100 - middleware route normalization boundary
            error = exc
            if isinstance(exc, RateLimitError):
                _audit_event(services, request, "gateway.rate_limit", status="denied")
                response = JSONResponse(status_code=429, content={"detail": str(exc)})
                return response
            if isinstance(exc, ConcurrencyLimitError):
                _audit_event(services, request, "gateway.concurrency_limit", status="denied")
                response = JSONResponse(status_code=503, content={"detail": str(exc)})
                return response
            if isinstance(exc, HTTPException):
                response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
                return response
            log_byte_error(
                byte_log,
                logging.ERROR,
                "security middleware request failed",
                error=exc,
                code=getattr(exc, "code", None) or ByteErrorCode.SERVER_MIDDLEWARE,
                boundary="server.middleware",
                stage=str(request.url.path),
            )
            if _security_enabled(services):
                response = JSONResponse(status_code=500, content={"detail": "Byte gateway internal error."})
                return response
            raise
        finally:
            if lease is not None:
                lease.release()
            if response is not None and _security_enabled(services):
                response.headers["Cache-Control"] = "no-store"
                response.headers["Pragma"] = "no-cache"
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["Referrer-Policy"] = "no-referrer"
            runtime = services.runtime_state()
            telemetry_runtime = runtime.telemetry_runtime
            if telemetry_runtime is not None and not stream_finalize_registered:
                telemetry_runtime.finish_request_span(
                    span,
                    request,
                    status_code=response.status_code if response is not None else getattr(error, "status_code", 500),
                    duration_s=time.perf_counter() - request_start,
                    error=error,
                )


__all__ = ["_wrap_streaming_response", "register_security_middleware"]
