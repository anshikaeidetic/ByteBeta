
"""Route completion and coalescer commit helpers for response finalization."""

from __future__ import annotations

import time
from typing import Any

from byte.adapter.runtime_state import get_coalescer
from byte.processor.model_router import record_route_outcome
from byte.processor.policy import record_policy_event


def _record_route_completion(
    chat_cache: Any,
    context: dict[str, Any],
    response_assessment: Any,
    *,
    start_time: float,
) -> tuple[Any, bool]:
    final_route_decision = context.get("_byte_model_route")
    accepted = response_assessment.accepted if response_assessment else True
    latency_ms = (time.time() - start_time) * 1000
    record_route_outcome(
        final_route_decision,
        accepted=accepted,
        latency_ms=latency_ms,
    )
    if final_route_decision is not None and getattr(
        chat_cache.config, "tenant_policy_learning", True
    ):
        record_policy_event(
            final_route_decision.route_key,
            category=final_route_decision.category,
            event=(
                "cheap_success"
                if final_route_decision.tier == "cheap" and accepted
                else "cheap_failure" if final_route_decision.tier == "cheap" else "request_complete"
            ),
            latency_ms=latency_ms,
        )
    return final_route_decision, accepted


def _complete_coalescer(chat_cache: Any, coalesce_key: str | None, llm_data: Any) -> None:
    if coalesce_key:
        get_coalescer(chat_cache).complete(coalesce_key, llm_data)


def _cancel_coalescer(chat_cache: Any, coalesce_key: str | None) -> None:
    if coalesce_key:
        get_coalescer(chat_cache).cancel(coalesce_key)

__all__ = ["_cancel_coalescer", "_complete_coalescer", "_record_route_completion"]
