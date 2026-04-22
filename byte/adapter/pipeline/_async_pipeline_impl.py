"""Thin async orchestration wrapper for the shared adapter pipeline stages."""

from __future__ import annotations

import asyncio
from typing import Any

from ._pipeline_bootstrap import (
    initialize_run_state,
    maybe_delegate_bypass_async,
    maybe_embed_request_async,
    maybe_return_coalesced_async,
    prepare_cache_inputs_async,
)
from ._pipeline_cache import lookup_cache_async, record_cache_miss
from ._pipeline_common import NO_RESULT, cancel_coalesced_request
from ._pipeline_finalize import finalize_pipeline_response
from ._pipeline_persist import persist_response_async
from ._pipeline_provider import execute_provider_stage_async


async def aadapt(
    llm_handler: Any,
    cache_data_convert: Any,
    update_cache_callback: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run the async adapter pipeline with shared stage helpers."""

    state, early_result = initialize_run_state(
        llm_handler,
        cache_data_convert,
        update_cache_callback,
        *args,
        event_loop=asyncio.get_running_loop(),
        **kwargs,
    )
    if early_result is not NO_RESULT:
        return early_result

    await prepare_cache_inputs_async(state)
    bypass_result = await maybe_delegate_bypass_async(state, recursive_adapter=aadapt)
    if bypass_result is not NO_RESULT:
        return bypass_result
    coalesced_result = await maybe_return_coalesced_async(state, recursive_adapter=aadapt)
    if coalesced_result is not NO_RESULT:
        return coalesced_result

    await maybe_embed_request_async(state)
    cache_result = await lookup_cache_async(state)
    if cache_result is not NO_RESULT:
        return cache_result
    record_cache_miss(state)

    provider_result = await execute_provider_stage_async(state, recursive_adapter=aadapt)
    if provider_result is NO_RESULT:
        cancel_coalesced_request(state)
        return None

    await persist_response_async(state)
    return finalize_pipeline_response(state)


__all__ = ["aadapt"]
