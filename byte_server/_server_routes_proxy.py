"""Proxy routes for non-chat gateway endpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

import byte.adapter as byte_adapter
from byte.security import sanitize_request_preview, validate_declared_content_length
from byte.utils.error import CacheError
from byte_server._server_gateway import (
    _call_chat_completion,
    _resolve_gateway_auth_kwargs,
    _resolve_gateway_handler,
    _response_cache_hit,
    _speech_media_type,
)
from byte_server._server_security import (
    _admin_auth_required,
    _audit_event,
    _chat_uses_server_credentials,
    _raise_route_error,
    _read_json_object,
    _read_upload_bytes,
    _require_admin,
    _sanitize_proxy_request_payload,
    _security_max_upload_bytes,
)
from byte_server._server_state import ServerServices


def _coerce_form_value(value) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


async def _read_proxy_form_payload(services: ServerServices, request: Request) -> dict:
    try:
        raw_length = str(request.headers.get("content-length", "") or "").strip()
        validate_declared_content_length(
            int(raw_length) if raw_length else None,
            limit=_security_max_upload_bytes(services),
            label="Upload request body",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc
    except CacheError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    form = await request.form()
    payload = {}
    for key, value in form.multi_items():
        payload[key] = value if hasattr(value, "filename") else _coerce_form_value(value)
    return payload


async def _run_proxy_gateway_request(
    services: ServerServices,
    request: Request,
    *,
    route_name: str,
    public_detail: str,
    model_name: str,
    cache_skip: bool,
    backend_handler,
    routed_handler,
    request_payload: dict[str, Any],
    response_builder: Callable[[Any], Response],
    audit_preview: dict[str, Any],
    extra_call_kwargs: dict[str, Any] | None = None,
) -> Response:
    """Execute one non-chat gateway route behind a single route-error boundary."""

    runtime = services.runtime_state()
    if runtime.gateway_cache is None:
        raise HTTPException(status_code=500, detail="The Byte gateway is not enabled on this server.")
    try:  # route normalization boundary
        use_routed_gateway, handler = _resolve_gateway_handler(
            services,
            model_name,
            backend_handler=backend_handler,
            routed_handler=routed_handler,
        )
        request.state.byte_gateway_mode = "adaptive" if use_routed_gateway else "backend"
        auth_kwargs = _resolve_gateway_auth_kwargs(
            services,
            request.headers,
            model_name,
            use_routed_gateway=use_routed_gateway,
        )
        response_payload = await _call_chat_completion(
            services,
            handler,
            cache_obj=runtime.gateway_cache,
            cache_skip=cache_skip,
            **auth_kwargs,
            **(extra_call_kwargs or {}),
            **request_payload,
        )
        request.state.byte_cache_hit = _response_cache_hit(response_payload)
        _audit_event(
            services,
            request,
            route_name,
            status="success",
            metadata={
                "preview": sanitize_request_preview(audit_preview),
                "mode": request.state.byte_gateway_mode,
                "cache_skip": bool(cache_skip),
                "cache_hit": bool(request.state.byte_cache_hit),
            },
        )
        return response_builder(response_payload)
    except Exception as exc:
        _raise_route_error(services, request, route_name, exc, public_detail=public_detail)


def register_proxy_routes(app: FastAPI, services: ServerServices) -> None:
    @app.api_route(f"{services.gateway_root}/images", methods=["POST", "OPTIONS"])
    async def image_generations(request: Request) -> Any:
        if request.method == "OPTIONS":
            return Response(status_code=204)
        image_params = _sanitize_proxy_request_payload(services, await _read_json_object(services, request))
        cache_skip = bool(image_params.pop("cache_skip", False))
        model_name = str(image_params.get("model", "") or "")
        request.state.byte_model_name = model_name
        if _chat_uses_server_credentials(services, request, model_name) and _admin_auth_required(services):
            _require_admin(services, request, "gateway.images")
        return await _run_proxy_gateway_request(
            services,
            request,
            route_name="images.generations",
            public_detail="Byte gateway image request failed.",
            model_name=model_name,
            cache_skip=cache_skip,
            backend_handler=services.default_backend.Image.create,
            routed_handler=byte_adapter.Image.create,
            request_payload=image_params,
            response_builder=lambda payload: JSONResponse(content=payload),
            audit_preview=image_params,
        )

    @app.api_route(f"{services.gateway_root}/moderations", methods=["POST", "OPTIONS"])
    async def moderations(request: Request) -> Any:
        if request.method == "OPTIONS":
            return Response(status_code=204)
        moderation_params = _sanitize_proxy_request_payload(services, await _read_json_object(services, request))
        cache_skip = bool(moderation_params.pop("cache_skip", False))
        model_name = str(moderation_params.get("model", "") or "")
        request.state.byte_model_name = model_name
        if _chat_uses_server_credentials(services, request, model_name) and _admin_auth_required(services):
            _require_admin(services, request, "gateway.moderations")
        return await _run_proxy_gateway_request(
            services,
            request,
            route_name="moderations",
            public_detail="Byte gateway moderation request failed.",
            model_name=model_name,
            cache_skip=cache_skip,
            backend_handler=services.default_backend.Moderation.create,
            routed_handler=byte_adapter.Moderation.create,
            request_payload=moderation_params,
            response_builder=lambda payload: JSONResponse(content=payload),
            audit_preview=moderation_params,
        )

    @app.api_route(f"{services.gateway_root}/audio/speech", methods=["POST", "OPTIONS"])
    async def audio_speech(request: Request) -> Any:
        if request.method == "OPTIONS":
            return Response(status_code=204)
        speech_params = _sanitize_proxy_request_payload(services, await _read_json_object(services, request))
        cache_skip = bool(speech_params.pop("cache_skip", False))
        model_name = str(speech_params.get("model", "") or "")
        response_format = str(speech_params.get("response_format", "mp3") or "mp3")
        request.state.byte_model_name = model_name
        if _chat_uses_server_credentials(services, request, model_name) and _admin_auth_required(services):
            _require_admin(services, request, "gateway.speech")
        return await _run_proxy_gateway_request(
            services,
            request,
            route_name="audio.speech",
            public_detail="Byte gateway speech request failed.",
            model_name=model_name,
            cache_skip=cache_skip,
            backend_handler=services.default_backend.Speech.create,
            routed_handler=byte_adapter.Speech.create,
            request_payload=speech_params,
            response_builder=lambda payload: Response(
                content=payload.get("audio", b"") if isinstance(payload, dict) else payload,
                media_type=_speech_media_type(response_format),
            ),
            audit_preview={k: v for k, v in speech_params.items() if k != "input"},
        )

    @app.api_route(f"{services.gateway_root}/audio/transcriptions", methods=["POST", "OPTIONS"])
    async def audio_transcriptions(request: Request) -> Any:
        if request.method == "OPTIONS":
            return Response(status_code=204)
        request_payload = await _read_proxy_form_payload(services, request)
        upload = request_payload.pop("file", None)
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="audio transcription requests require a file upload.")
        audio_params = _sanitize_proxy_request_payload(services, request_payload)
        cache_skip = bool(audio_params.pop("cache_skip", False))
        model_name = str(audio_params.get("model", "") or "")
        request.state.byte_model_name = model_name
        if _chat_uses_server_credentials(services, request, model_name) and _admin_auth_required(services):
            _require_admin(services, request, "gateway.audio.transcriptions")
        file_payload = {"name": str(getattr(upload, "filename", "") or "upload.bin"), "bytes": await _read_upload_bytes(services, upload, label="Audio transcription upload")}
        return await _run_proxy_gateway_request(
            services,
            request,
            route_name="audio.transcriptions",
            public_detail="Byte gateway transcription request failed.",
            model_name=model_name,
            cache_skip=cache_skip,
            backend_handler=services.default_backend.Audio.transcribe,
            routed_handler=byte_adapter.Audio.transcribe,
            request_payload=audio_params,
            response_builder=lambda payload: JSONResponse(content=payload),
            audit_preview=audio_params,
            extra_call_kwargs={"file": file_payload},
        )

    @app.api_route(f"{services.gateway_root}/audio/translations", methods=["POST", "OPTIONS"])
    async def audio_translations(request: Request) -> Any:
        if request.method == "OPTIONS":
            return Response(status_code=204)
        request_payload = await _read_proxy_form_payload(services, request)
        upload = request_payload.pop("file", None)
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="audio translation requests require a file upload.")
        audio_params = _sanitize_proxy_request_payload(services, request_payload)
        cache_skip = bool(audio_params.pop("cache_skip", False))
        model_name = str(audio_params.get("model", "") or "")
        request.state.byte_model_name = model_name
        if _chat_uses_server_credentials(services, request, model_name) and _admin_auth_required(services):
            _require_admin(services, request, "gateway.audio.translations")
        file_payload = {"name": str(getattr(upload, "filename", "") or "upload.bin"), "bytes": await _read_upload_bytes(services, upload, label="Audio translation upload")}
        return await _run_proxy_gateway_request(
            services,
            request,
            route_name="audio.translations",
            public_detail="Byte gateway translation request failed.",
            model_name=model_name,
            cache_skip=cache_skip,
            backend_handler=services.default_backend.Audio.translate,
            routed_handler=byte_adapter.Audio.translate,
            request_payload=audio_params,
            response_builder=lambda payload: JSONResponse(content=payload),
            audit_preview=audio_params,
            extra_call_kwargs={"file": file_payload},
        )


__all__ = ["register_proxy_routes"]
