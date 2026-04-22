"""Chat completion route for the Byte server."""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

import byte.adapter as byte_adapter
from byte.processor.intent import extract_request_intent
from byte_server._control_plane import (
    apply_memory_resolution,
    provider_mode_for_request,
)
from byte_server._server_gateway import (
    _call_chat_completion,
    _resolve_gateway_auth_kwargs,
    _should_use_routed_gateway,
)
from byte_server._server_security import (
    _admin_auth_required,
    _audit_event,
    _chat_uses_server_credentials,
    _extract_cache_skip_flag,
    _raise_route_error,
    _read_json_object,
    _require_admin,
    _sanitize_proxy_request_payload,
    sanitize_request_preview,
)
from byte_server._server_state import ServerServices

# Byte Smart Router: model swap must stay within the original provider family
# (e.g. no DeepSeek→gpt-4o) to avoid "Model Not Exist" from the upstream API.
# _PROVIDER_PREFIXES enforces this; _PROVIDER_TIER_DEFAULTS resolves cheap/strong per provider.
_PROVIDER_TIER_DEFAULTS: dict[str, tuple[str, str]] = {
    "openai":    ("gpt-4o-mini",              "gpt-4o"),
    "deepseek":  ("deepseek-chat",            "deepseek-reasoner"),
    "anthropic": ("claude-haiku-4-5-20251001", "claude-sonnet-4-6"),
    "groq":      ("llama-3.1-8b-instant",     "llama-3.3-70b-versatile"),
    "gemini":    ("gemini-2.0-flash-lite",    "gemini-1.5-pro"),
    "mistral":   ("mistral-small-latest",     "mistral-large-latest"),
    "cohere":    ("command-r",                "command-r-plus"),
}
_PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai":    ("gpt-",),
    "deepseek":  ("deepseek-",),
    "anthropic": ("claude-",),
    "groq":      ("llama-", "mixtral-", "gemma-"),
    "gemini":    ("gemini-",),
    "mistral":   ("mistral-", "codestral-"),
    "cohere":    ("command-", "c4ai-"),
}
def _build_byte_features(
    chat_response: Any,
    request: Request,
    *,
    distill_requested: bool = False,
) -> dict[str, Any]:
    """Extract a unified feature-signal snapshot for the demo UI.

    Reads from already-populated response fields and request.state — does not
    mutate anything. Every sub-object stays None/False when its source signal
    is absent so the UI can honestly reflect what ran.
    """
    resp: dict[str, Any] = chat_response if isinstance(chat_response, dict) else {}
    reasoning = resp.get("byte_reasoning") or {}
    worker = resp.get("byte_worker") or {}
    route = getattr(request.state, "byte_route_decision", None) or {}
    cascade = getattr(request.state, "byte_cascade", None) or {}
    stale_invalidated = bool(getattr(request.state, "byte_cascade_invalidated", False))
    distill = resp.get("byte_distill") or {}
    quality = resp.get("byte_quality") or {}
    cache_hit = bool(resp.get("byte", False))
    # Quality Guard runs on every non-cache response (assessment + admission gate)
    # and on cache reads (constraint scoring). Signal it as "ran" whenever a
    # response came back.
    quality_ran = bool(resp) or bool(quality)
    # Distillation: "applied" means the client requested it AND the pipeline
    # actually compressed the prompt (we can't infer the latter without context
    # plumbing, so trust the client flag — the pipeline silently no-ops for
    # short prompts under 512 chars).
    distill_applied = bool(distill.get("applied", False)) or (
        distill_requested and len(_last_user_content({"messages": resp.get("choices", []) or []})) >= 512
    )
    return {
        "quality": {
            "ran": quality_ran,
            "score": quality.get("score"),
            "accepted": quality.get("accepted"),
            "repaired": bool(quality.get("repaired", False)),
        },
        "cache": {
            "hit": cache_hit,
            "similarity": resp.get("byte_similarity"),
            "debug": {
                "model_used": resp.get("model"),
                "distill_mode": distill.get("mode") if distill else None,
                "admitted": bool(resp.get("byte_admitted", True)),
            },
        },
        "routing": {
            "enabled": True,
            "swapped": bool(route.get("swapped", False)),
            "tier": route.get("tier"),
            "from_model": route.get("from_model"),
            "to_model": route.get("to_model"),
            "worker_dispatched": bool(worker.get("worker_id")),
        },
        "distillation": {
            "applied": distill_applied,
            "tokens_saved": int(distill.get("tokens_saved", 0) or 0),
        },
        "reasoning": {
            "reused": bool(reasoning),
            "kind": reasoning.get("kind") if isinstance(reasoning, dict) else None,
            "source": reasoning.get("source") if isinstance(reasoning, dict) else None,
        },
        "cascade": {
            "escalated": bool(cascade.get("escalated", False)),
            "stale_invalidated": stale_invalidated or bool(cascade.get("stale_invalidated", False)),
            "from_tier": cascade.get("from_tier"),
            "to_tier": cascade.get("to_tier"),
        },
    }


def _last_user_content(params: dict) -> str:
    for msg in reversed(params.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            c = msg.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return str(part.get("text", ""))
    return ""


def register_chat_routes(app: FastAPI, services: ServerServices) -> None:
    @app.api_route("/v1/chat/completions", methods=["POST", "OPTIONS"])
    @app.api_route(f"{services.gateway_root}/chat", methods=["POST", "OPTIONS"])
    async def chat(request: Request) -> Any:
        runtime = services.runtime_state()
        if runtime.gateway_cache is None:
            raise HTTPException(status_code=500, detail="The Byte gateway is not enabled on this server.")
        if request.method == "OPTIONS":
            return Response(status_code=204)

        from starlette.responses import JSONResponse, StreamingResponse

        request_payload = await _read_json_object(services, request)
        chat_params = _sanitize_proxy_request_payload(services, request_payload)
        chat_params, cache_skip = _extract_cache_skip_flag(chat_params)
        model_name = str(chat_params.get("model", "") or "")

        try:
            active_cfg = getattr(runtime.gateway_cache, "config", None)
            if active_cfg is not None and getattr(active_cfg, "route_llm_enabled", False):
                provider_hint = str(request.headers.get("x-byte-provider", "") or "").strip().lower()
                if provider_hint in _PROVIDER_TIER_DEFAULTS:
                    cheap, strong = _PROVIDER_TIER_DEFAULTS[provider_hint]
                else:
                    cheap = str(getattr(active_cfg, "routing_cheap_model", "") or "")
                    strong = str(getattr(active_cfg, "routing_expensive_model", "") or "") or model_name
                threshold = float(getattr(active_cfg, "route_llm_threshold", 0.5) or 0.5)
                seed_path = str(getattr(active_cfg, "route_llm_seed_path", "") or "")
                last_user = _last_user_content(chat_params)
                if last_user and (cheap or strong):
                    from byte.router import (
                        route_decision as _rl_decide,  # pylint: disable=import-outside-toplevel
                    )
                    dec = _rl_decide(
                        last_user,
                        cheap_model=cheap,
                        strong_model=strong,
                        threshold=threshold,
                        seed_path=seed_path,
                        default_model=model_name,
                    )
                    safe_to_swap = True
                    if provider_hint and provider_hint in _PROVIDER_PREFIXES:
                        pfxs = _PROVIDER_PREFIXES[provider_hint]
                        target_ok = any(dec.selected_model.startswith(p) for p in pfxs)
                        orig_ok   = any(model_name.startswith(p) for p in pfxs) if model_name else True
                        if not (target_ok and orig_ok):
                            safe_to_swap = False
                    original_model = model_name
                    swapped = bool(
                        safe_to_swap and dec.selected_model and dec.selected_model != model_name
                    )
                    if swapped:
                        chat_params["model"] = dec.selected_model
                        model_name = dec.selected_model
                    request.state.byte_route_decision = {
                        "swapped": swapped,
                        "tier": dec.tier,
                        "from_model": original_model,
                        "to_model": dec.selected_model if swapped else original_model,
                        "score": float(getattr(dec, "score", 0.0) or 0.0),
                    }
                    try:
                        from byte.telemetry import (
                            bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
                        )
                        _bump("byte_router_total")
                        _bump(f"byte_router_{dec.tier}")
                        if not safe_to_swap:
                            _bump("byte_router_skipped_cross_provider")
                    except Exception:  # pragma: no cover - defensive
                        pass
        except Exception as _rl_err:  # pragma: no cover - defensive
            from byte.utils.log import byte_log as _bl  # pylint: disable=import-outside-toplevel
            _bl.debug("Byte Smart Router skipped: %s", _rl_err)

        request.state.byte_model_name = model_name
        control_plane = services.control_plane()
        scope = (
            control_plane.extract_scope(request.headers, chat_params)
            if control_plane is not None
            else None
        )
        intent = extract_request_intent(chat_params)
        if control_plane is not None and scope is not None:
            control_plane.record_intent(scope=scope, route_key=intent.route_key)
        if _chat_uses_server_credentials(services, request, model_name) and _admin_auth_required(services):
            _require_admin(services, request, "gateway.chat")
        is_stream = chat_params.get("stream", False)
        try:
            worker_selection = (
                control_plane.maybe_select_worker(scope=scope, request_payload=chat_params)
                if control_plane is not None and scope is not None and not is_stream
                else None
            )
            provider_mode = provider_mode_for_request(
                chat_params,
                worker_selected=worker_selection is not None,
            )
            if control_plane is not None and scope is not None:
                memory_resolution = control_plane.resolve_memory(
                    scope=scope,
                    request_payload=chat_params,
                    provider_mode=provider_mode,
                )
                chat_params = apply_memory_resolution(chat_params, memory_resolution)

            use_routed_gateway = _should_use_routed_gateway(services, model_name)
            request.state.byte_gateway_mode = "adaptive" if use_routed_gateway else "backend"
            handler = (
                byte_adapter.ChatCompletion.create
                if use_routed_gateway
                else services.default_backend.ChatCompletion.create
            )
            auth_kwargs = _resolve_gateway_auth_kwargs(
                services,
                request.headers,
                model_name,
                use_routed_gateway=use_routed_gateway,
            )
            if is_stream:
                import asyncio

                async def generate() -> Any:
                    yield ": ok\n\n"
                    queue: asyncio.Queue = asyncio.Queue()
                    _sentinel = object()
                    _loop = asyncio.get_running_loop()

                    def _enqueue(item: object) -> None:
                        _loop.call_soon_threadsafe(queue.put_nowait, item)

                    def _produce() -> None:
                        try:
                            _is_cache_hit = False
                            _last_dict_chunk: dict[str, Any] | None = None
                            _distill_req = bool(chat_params.get("byte_prompt_distillation_mode"))
                            for stream_response in handler(
                                cache_obj=runtime.gateway_cache,
                                cache_skip=cache_skip,
                                **auth_kwargs,
                                **chat_params,
                            ):
                                if stream_response == "[DONE]":
                                    _features_payload = json.dumps({
                                        "byte_features": _build_byte_features(
                                            _last_dict_chunk or {}, request,
                                            distill_requested=_distill_req,
                                        )
                                    })
                                    _enqueue(f"data: {_features_payload}\n\n")
                                    _enqueue("data: [DONE]\n\n")
                                    return
                                if isinstance(stream_response, dict):
                                    _last_dict_chunk = stream_response
                                    if stream_response.get("byte") is True:
                                        _is_cache_hit = True
                                _enqueue(f"data: {json.dumps(stream_response)}\n\n")
                                if _is_cache_hit:
                                    time.sleep(0.005)
                            _features_payload = json.dumps({
                                "byte_features": _build_byte_features(
                                    _last_dict_chunk or {}, request,
                                    distill_requested=_distill_req,
                                )
                            })
                            _enqueue(f"data: {_features_payload}\n\n")
                            _enqueue("data: [DONE]\n\n")
                        except Exception as stream_exc:
                            _enqueue(f"data: {json.dumps({'error': {'message': str(stream_exc), 'type': 'stream_error'}})}\n\n")
                            _enqueue("data: [DONE]\n\n")
                        finally:
                            _enqueue(_sentinel)

                    _loop.run_in_executor(None, _produce)

                    while True:
                        item = await queue.get()
                        if item is _sentinel:
                            break
                        yield item
                        await asyncio.sleep(0)

                _audit_event(
                    services,
                    request,
                    "chat.completions",
                    status="stream_started",
                    metadata={
                        "preview": sanitize_request_preview(chat_params),
                        "mode": request.state.byte_gateway_mode,
                        "cache_skip": bool(cache_skip),
                    },
                )
                telemetry_runtime = services.runtime_state().telemetry_runtime
                if telemetry_runtime is not None:
                    telemetry_runtime.record_chat_result(
                        mode=request.state.byte_gateway_mode,
                        cache_hit=None,
                        model_name=model_name,
                    )
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={
                        "X-Accel-Buffering": "no",
                        "Cache-Control": "no-cache, no-transform",
                        "Connection": "keep-alive",
                    },
                )

            started_at = time.perf_counter()
            if worker_selection is not None and control_plane is not None and scope is not None:
                try:
                    chat_response = await services.run_in_threadpool()(
                        control_plane.dispatch_to_worker,
                        worker=worker_selection,
                        scope=scope,
                        request_payload=chat_params,
                    )
                    request.state.byte_gateway_mode = f"control_plane:{worker_selection.source}"
                except Exception:
                    chat_response = await _call_chat_completion(
                        services,
                        handler,
                        cache_obj=runtime.gateway_cache,
                        cache_skip=cache_skip,
                        **auth_kwargs,
                        **chat_params,
                    )
            else:
                chat_response = await _call_chat_completion(
                    services,
                    handler,
                    cache_obj=runtime.gateway_cache,
                    cache_skip=cache_skip,
                    **auth_kwargs,
                    **chat_params,
                )
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            cache_hit = bool(chat_response.get("byte", False)) if isinstance(chat_response, dict) else False
            request.state.byte_cache_hit = cache_hit
            worker_id = ""
            if isinstance(chat_response, dict):
                worker_id = str((chat_response.get("byte_worker") or {}).get("worker_id", "") or "")
            if control_plane is not None and scope is not None:
                control_plane.store.record_cache_event(
                    scope=scope,
                    route_key=intent.route_key,
                    event_type="chat_completion",
                    reason=request.state.byte_gateway_mode,
                    cache_hit=cache_hit,
                    latency_ms=latency_ms,
                    worker_id=worker_id,
                    metadata={
                        "model": model_name,
                        "cache_skip": bool(cache_skip),
                        "scope_source": scope.source,
                    },
                )
                control_plane.remember_memory(
                    scope=scope,
                    request_payload=chat_params,
                    response_payload=chat_response if isinstance(chat_response, dict) else {"answer": str(chat_response)},
                    provider_mode=provider_mode,
                    worker_id=worker_id,
                )
                control_plane.maybe_schedule_replay(
                    scope=scope,
                    request_payload=chat_params,
                    response_payload=chat_response if isinstance(chat_response, dict) else {"answer": str(chat_response)},
                    feature="prompt_distillation"
                    if bool(chat_params.get("byte_prompt_distillation_mode"))
                    else "shadow",
                )
            _audit_event(
                services,
                request,
                "chat.completions",
                status="success",
                metadata={
                    "preview": sanitize_request_preview(chat_params),
                    "mode": request.state.byte_gateway_mode,
                    "cache_skip": bool(cache_skip),
                    "cache_hit": cache_hit,
                },
            )
            telemetry_runtime = services.runtime_state().telemetry_runtime
            if telemetry_runtime is not None:
                telemetry_runtime.record_chat_result(
                    mode=request.state.byte_gateway_mode,
                    cache_hit=cache_hit,
                    model_name=model_name,
                )
            if isinstance(chat_response, dict):
                chat_response["byte_features"] = _build_byte_features(
                    chat_response, request,
                    distill_requested=bool(chat_params.get("byte_prompt_distillation_mode")),
                )
            return JSONResponse(content=chat_response)
        except Exception as exc:  # pylint: disable=W0703
            _raise_route_error(
                services,
                request,
                "chat.completions",
                exc,
                public_detail="Byte gateway chat request failed.",
            )


__all__ = ["register_chat_routes"]
