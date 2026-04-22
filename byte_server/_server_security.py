"""Security, metrics, and middleware helpers for the Byte server."""

from __future__ import annotations

from byte.security import sanitize_request_preview, short_digest
from byte_server._server_security_audit import _audit_event, _raise_route_error
from byte_server._server_security_auth import (
    _admin_auth_required,
    _admin_token,
    _allowed_egress_hosts,
    _audit_log_path,
    _cfg,
    _chat_uses_server_credentials,
    _extract_admin_token,
    _flag,
    _request_actor,
    _request_host,
    _request_scheme,
    _require_admin,
    _security_enabled,
    _security_limit,
)
from byte_server._server_security_metrics import _prometheus_metrics_payload, _readiness_payload
from byte_server._server_security_middleware import (
    _wrap_streaming_response,
    register_security_middleware,
)
from byte_server._server_security_validation import (
    _apply_security_overrides,
    _cache_file_disabled,
    _extract_cache_skip_flag,
    _read_json_object,
    _read_upload_bytes,
    _resolve_artifact_path,
    _sanitize_proxy_request_payload,
    _security_max_archive_bytes,
    _security_max_archive_members,
    _security_max_upload_bytes,
)

__all__ = [
    "_admin_auth_required",
    "_admin_token",
    "_allowed_egress_hosts",
    "_apply_security_overrides",
    "_audit_event",
    "_audit_log_path",
    "_cache_file_disabled",
    "_cfg",
    "_chat_uses_server_credentials",
    "_extract_admin_token",
    "_extract_cache_skip_flag",
    "_flag",
    "_prometheus_metrics_payload",
    "_raise_route_error",
    "_read_json_object",
    "_read_upload_bytes",
    "_readiness_payload",
    "_request_actor",
    "_request_host",
    "_request_scheme",
    "_require_admin",
    "_resolve_artifact_path",
    "_sanitize_proxy_request_payload",
    "_security_enabled",
    "_security_limit",
    "_security_max_archive_bytes",
    "_security_max_archive_members",
    "_security_max_upload_bytes",
    "_wrap_streaming_response",
    "register_security_middleware",
    "sanitize_request_preview",
    "short_digest",
]
