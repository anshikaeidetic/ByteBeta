"""Thin sync orchestration wrapper for the shared adapter pipeline stages."""

from __future__ import annotations

from typing import Any

from ._pipeline_bootstrap import (
    initialize_run_state,
    maybe_delegate_bypass_sync,
    maybe_embed_request_sync,
    maybe_return_coalesced_sync,
    prepare_cache_inputs_sync,
)
from ._pipeline_cache import lookup_cache_sync, record_cache_miss
from ._pipeline_common import NO_RESULT, cancel_coalesced_request
from ._pipeline_finalize import finalize_pipeline_response
from ._pipeline_persist import persist_response_sync
from ._pipeline_provider import execute_provider_stage_sync


def adapt(
    llm_handler: Any,
    cache_data_convert: Any,
    update_cache_callback: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run the sync adapter pipeline with shared stage helpers."""

    state, early_result = initialize_run_state(
        llm_handler,
        cache_data_convert,
        update_cache_callback,
        *args,
        **kwargs,
    )
    if early_result is not NO_RESULT:
        return early_result

    prepare_cache_inputs_sync(state)
    bypass_result = maybe_delegate_bypass_sync(state, recursive_adapter=adapt)
    if bypass_result is not NO_RESULT:
        return bypass_result
    coalesced_result = maybe_return_coalesced_sync(state, recursive_adapter=adapt)
    if coalesced_result is not NO_RESULT:
        return coalesced_result

    maybe_embed_request_sync(state)
    cache_result = lookup_cache_sync(state)
    if cache_result is not NO_RESULT:
        return cache_result
    record_cache_miss(state)

    provider_result = execute_provider_stage_sync(state, recursive_adapter=adapt)
    if provider_result is NO_RESULT:
        cancel_coalesced_request(state)
        return None

    persist_response_sync(state)
    return finalize_pipeline_response(state)


__all__ = ["adapt"]
