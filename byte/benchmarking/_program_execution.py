"""Structured execution helpers for provider-boundary benchmark calls."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from byte.benchmarking._program_models import ExecutionConfig, ExecutionRecord, RequestCase


def execute_case_with_retries(
    case: RequestCase,
    config: ExecutionConfig,
    call: Callable[[], tuple[str, dict[str, Any]]],
) -> ExecutionRecord:
    """Execute one benchmark case and convert failures into structured records."""

    started = time.perf_counter()
    last_error: BaseException | None = None
    attempts = max(1, int(config.attempts))
    for attempt in range(1, attempts + 1):
        try:
            response_text, usage = call()
            return ExecutionRecord(
                case_id=case.case_id,
                provider=config.provider,
                phase=config.phase,
                ok=True,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
                response_text=response_text,
                usage=usage,
            )
        except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(max(0.0, config.base_sleep_seconds) + attempt - 1)

    error = {
        "provider": config.provider,
        "case_id": case.case_id,
        "phase": config.phase,
        "attempts": attempts,
        "type": type(last_error).__name__ if last_error is not None else "RuntimeError",
        "message": str(last_error or "provider call failed without an exception"),
    }
    return ExecutionRecord(
        case_id=case.case_id,
        provider=config.provider,
        phase=config.phase,
        ok=False,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 2),
        error=error,
    )


__all__ = ["execute_case_with_retries"]
