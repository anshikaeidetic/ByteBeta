from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeVar


class ByteErrorCode(str, Enum):
    """Stable internal error codes used in logs, telemetry, and audit records."""

    PROVIDER_CONFIG = "provider.config"
    PROVIDER_TRANSPORT = "provider.transport"
    PROVIDER_RESPONSE = "provider.response"
    PROVIDER_AUTH = "provider.auth"
    PIPELINE_CACHE_LOOKUP = "pipeline.cache_lookup"
    PIPELINE_CACHE_SAVE = "pipeline.cache_save"
    PIPELINE_VERIFIER = "pipeline.verifier"
    PIPELINE_ESCALATION = "pipeline.escalation"
    ROUTER_NO_TARGET = "router.no_target"
    ROUTER_DISPATCH = "router.dispatch"
    ROUTER_RETRY_EXHAUSTED = "router.retry_exhausted"
    TELEMETRY_BOOTSTRAP = "telemetry.bootstrap"
    TELEMETRY_EXPORT = "telemetry.export"
    TELEMETRY_SHUTDOWN = "telemetry.shutdown"
    STORAGE_READ = "storage.read"
    STORAGE_WRITE = "storage.write"
    STORAGE_SEARCH = "storage.search"
    SERVER_ROUTE = "server.route"
    SERVER_MIDDLEWARE = "server.middleware"
    SERVER_REDACTION = "server.redaction"


@dataclass(frozen=True)
class ByteErrorInfo:
    """Normalized error metadata for Byte internal boundaries."""

    code: str
    boundary: str = ""
    retryable: bool | None = None
    provider: str = ""
    http_status: int | None = None
    public_detail: str = ""
    error_type: str = ""


class CacheError(Exception):
    """Byte base error."""

    def __init__(
        self,
        *args,
        code: ByteErrorCode | str | None = None,
        retryable: bool | None = None,
        provider: str | None = None,
        http_status: int | None = None,
        public_detail: str | None = None,
    ) -> None:
        super().__init__(*args)
        self.code = _code_value(code)
        self.retryable = retryable
        self.provider = str(provider or "")
        self.http_status = http_status
        self.public_detail = str(public_detail or "")


class NotInitError(CacheError):
    """Raise when the cache has been used before it's inited"""

    def __init__(self) -> None:
        super().__init__("The cache should be inited before using")


class NotFoundError(CacheError):
    """Raise when getting an unsupported store."""

    def __init__(self, store_type, current_type_name) -> None:
        super().__init__(f"Unsupported ${store_type}: {current_type_name}")


class ParamError(CacheError):
    """Raise when receiving an invalid param."""


class PipInstallError(CacheError):
    """Raise when failed to install package."""

    def __init__(self, package) -> None:
        super().__init__(f"Ran into error installing {package}.")


class FeatureNotSupportedError(CacheError):
    """Raise when a surfaced capability is unavailable."""


class SecurityPolicyError(CacheError):
    """Raise when a request violates Byte security policy."""


class ArtifactValidationError(CacheError):
    """Raise when an import or export artifact is invalid."""


class RateLimitError(CacheError):
    """Raise when a request exceeds an active rate limit."""


class ConcurrencyLimitError(CacheError):
    """Raise when a request exceeds active concurrency policy."""


class ProviderRequestError(CacheError):
    """Raise when an upstream provider request fails."""

    def __init__(
        self,
        *,
        provider: str,
        message: str,
        status_code: int | None = None,
        retryable: bool | None = None,
        code: ByteErrorCode | str | None = None,
        public_detail: str | None = None,
    ) -> None:
        super().__init__(
            f"{provider}: {message}",
            code=code or ByteErrorCode.PROVIDER_TRANSPORT,
            retryable=retryable,
            provider=provider,
            http_status=status_code,
            public_detail=public_detail,
        )
        self.status_code = status_code
        self.message = message


def _code_value(code: ByteErrorCode | str | None) -> str:
    if isinstance(code, ByteErrorCode):
        return code.value
    return str(code or "")


def classify_error(
    exc: Exception,
    *,
    code: ByteErrorCode | str | None = None,
    boundary: str = "",
    provider: str | None = None,
    public_detail: str | None = None,
) -> ByteErrorInfo:
    """Return stable metadata for logging or route normalization."""

    resolved_code = _code_value(code)
    if not resolved_code:
        resolved_code = _code_value(getattr(exc, "code", None))
    if not resolved_code and isinstance(exc, ProviderRequestError):
        resolved_code = ByteErrorCode.PROVIDER_TRANSPORT.value
    if not resolved_code:
        resolved_code = ByteErrorCode.SERVER_ROUTE.value
    resolved_retryable = getattr(exc, "retryable", None)
    resolved_provider = str(provider or getattr(exc, "provider", "") or "")
    resolved_status = getattr(exc, "http_status", None)
    if resolved_status is None:
        resolved_status = getattr(exc, "status_code", None)
    resolved_public_detail = str(public_detail or getattr(exc, "public_detail", "") or "")
    return ByteErrorInfo(
        code=resolved_code,
        boundary=boundary,
        retryable=resolved_retryable,
        provider=resolved_provider,
        http_status=resolved_status,
        public_detail=resolved_public_detail,
        error_type=type(exc).__name__,
    )


_ExceptionT = TypeVar("_ExceptionT", bound=Exception)


def set_error_metadata(
    exc: _ExceptionT,
    *,
    code: ByteErrorCode | str | None = None,
    retryable: bool | None = None,
    provider: str | None = None,
    http_status: int | None = None,
    public_detail: str | None = None,
) -> _ExceptionT:
    """Attach stable Byte metadata to an exception when the boundary knows more."""

    if code is not None:
        exc.code = _code_value(code)
    if retryable is not None:
        exc.retryable = retryable
    if provider:
        exc.provider = str(provider)
    if http_status is not None:
        exc.http_status = int(http_status)
    if public_detail:
        exc.public_detail = str(public_detail)
    return exc


def wrap_error(e: Exception) -> Exception:
    """Add a type to exception `e` while ensuring that the original type is not changed

    Example:
        .. code-block:: python

            import openai

            from byte.utils.error import wrap_error


            def raise_error():
                try:
                    raise openai.error.OpenAIError(message="test")
                except openai.error.OpenAIError as e:
                    raise wrap_error(e)


            try:
                raise_error()
            except openai.error.OpenAIError as e:
                print("exception:")
                print(e)

            print("over")
    """
    wrapped_type = type(e.__class__.__name__, (CacheError, e.__class__), {})
    try:
        e.__class__ = wrapped_type
        return e
    except TypeError:
        args = getattr(e, "args", ())
        try:
            wrapped = wrapped_type(*args)
        except (AttributeError, TypeError, ValueError):
            wrapped = CacheError(str(e))
        for field in ("code", "retryable", "provider", "http_status", "public_detail", "status_code"):
            if hasattr(e, field):
                setattr(wrapped, field, getattr(e, field))
        return wrapped.with_traceback(e.__traceback__)


__all__ = [
    "ArtifactValidationError",
    "ByteErrorCode",
    "ByteErrorInfo",
    "CacheError",
    "ConcurrencyLimitError",
    "FeatureNotSupportedError",
    "NotFoundError",
    "NotInitError",
    "ParamError",
    "PipInstallError",
    "ProviderRequestError",
    "RateLimitError",
    "SecurityPolicyError",
    "classify_error",
    "set_error_metadata",
    "wrap_error",
]
