import logging

import byte
from byte.utils.error import ByteErrorCode, classify_error

FORMAT = (
    "%(asctime)s - %(thread)d - %(filename)s-%(module)s:%(lineno)s - %(levelname)s: %(message)s"
)
logging.basicConfig(format=FORMAT)

byte_log = logging.getLogger(f"byte:{byte.__version__}")


def log_byte_error(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    error: Exception,
    code: ByteErrorCode | str | None = None,
    boundary: str = "",
    stage: str = "",
    provider: str = "",
    exc_info: bool = True,
) -> None:
    """Log an exception with stable Byte metadata fields."""

    info = classify_error(
        error,
        code=code,
        boundary=boundary,
        provider=provider,
    )
    logger.log(
        level,
        message,
        exc_info=exc_info,
        extra={
            "byte_error_code": info.code,
            "byte_error_boundary": info.boundary,
            "byte_error_retryable": info.retryable,
            "byte_provider": info.provider,
            "byte_stage": stage,
        },
    )


__all__ = ["byte_log", "log_byte_error"]
