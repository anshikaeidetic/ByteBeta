"""Private inference worker service for Byte control-plane routing."""

from __future__ import annotations

import argparse
import os
import time
from threading import Lock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import byte.adapter as byte_adapter
from byte import INTERNAL_AUTH_HEADER, PRODUCT_NAME, Cache, Config, __version__
from byte_server._control_plane import request_text
from byte_server._server_gateway import _init_gateway_cache
from byte_server.models import (
    InferenceGenerateRequest,
    InferencePrefillRequest,
    KVEventData,
    WorkerHeartbeat,
)


def _prompt_digest(payload: dict[str, Any]) -> str:
    from hashlib import sha256

    return sha256(request_text(payload).encode("utf-8")).hexdigest()[:16]


def _compression_profile(payload: dict[str, Any]) -> str:
    codec = str(payload.get("byte_kv_codec") or payload.get("kv_codec") or "disabled")
    bits = int(payload.get("byte_kv_bits") or payload.get("kv_bits") or 8)
    h2o = int(bool(payload.get("byte_h2o_enabled") or payload.get("h2o_enabled") or False))
    return f"{codec}:{bits}:h2o={h2o}"


class InferenceWorkerRuntime:
    def __init__(
        self,
        *,
        worker_id: str,
        cache_dir: str,
        model_inventory: list[str] | None = None,
        free_vram_gb: float = 0.0,
    ) -> None:
        self.worker_id = str(worker_id or "byte-worker")
        self.cache_dir = str(cache_dir or "byte_worker_cache")
        self.model_inventory = [
            str(item).strip() for item in (model_inventory or []) if str(item).strip()
        ]
        self.free_vram_gb = float(free_vram_gb or 0.0)
        self.queue_depth = 0
        self.requests = 0
        self.prefills = 0
        self.kv_events: list[dict[str, Any]] = []
        self._lock = Lock()
        self.cache = _init_gateway_cache(
            mode="exact",
            cache_dir=self.cache_dir,
            cache_obj=Cache(),
            config=Config(enable_token_counter=False, model_routing=False),
        )

    def heartbeat(self) -> dict[str, Any]:
        with self._lock:
            return WorkerHeartbeat(
                worker_id=self.worker_id,
                status="ready",
                queue_depth=self.queue_depth,
                free_vram_gb=self.free_vram_gb,
                model_inventory=self.model_inventory,
                health_score=max(0.05, min(1.0, 1.0 - min(self.queue_depth / 20.0, 0.5))),
                metadata={
                    "requests": self.requests,
                    "prefills": self.prefills,
                    "kv_events": len(self.kv_events),
                },
                last_heartbeat=time.time(),
            ).model_dump()

    def prefill(self, payload: InferencePrefillRequest) -> dict[str, Any]:
        with self._lock:
            self.prefills += 1
        return {
            "worker_id": self.worker_id,
            "scope": payload.scope.model_dump(),
            "model_name": str(payload.request.get("model", "") or ""),
            "prompt_digest": _prompt_digest(payload.request),
            "compression_profile": _compression_profile(payload.request),
            "accepted": True,
        }

    def record_kv_event(self, payload: KVEventData) -> dict[str, Any]:
        event = payload.model_dump()
        event["created_at"] = time.time()
        with self._lock:
            self.kv_events.append(event)
            self.kv_events = self.kv_events[-100:]
        return event

    def generate(self, payload: InferenceGenerateRequest) -> dict[str, Any]:
        request_payload = dict(payload.request or {})
        if bool(request_payload.get("stream", False)):
            raise HTTPException(
                status_code=400,
                detail="byte_inference currently supports non-streaming generate calls only.",
            )
        with self._lock:
            self.queue_depth += 1
            self.requests += 1
        try:
            response = byte_adapter.ChatCompletion.create(
                cache_obj=self.cache,
                cache_skip=True,
                **request_payload,
            )
            if not isinstance(response, dict):
                raise HTTPException(status_code=502, detail="byte_inference expected a dict response.")
            response["byte_worker"] = {
                "worker_id": self.worker_id,
                "scope": payload.scope.model_dump(),
                "selection_source": payload.selection_source,
                "prompt_digest": _prompt_digest(request_payload),
                "compression_profile": _compression_profile(request_payload),
                "model_inventory": list(self.model_inventory),
            }
            return response
        finally:
            with self._lock:
                self.queue_depth = max(0, self.queue_depth - 1)


def _normalize_internal_token(value: str | None) -> str:
    return str(value or "").strip()


def _require_internal_auth(request: Request) -> None:
    if not _internal_auth_token:
        return
    if request.headers.get(INTERNAL_AUTH_HEADER) == _internal_auth_token:
        return
    raise HTTPException(status_code=403, detail="Byte internal authentication failed.")


runtime = InferenceWorkerRuntime(
    worker_id=os.getenv("BYTE_WORKER_ID", "byte-worker"),
    cache_dir=os.getenv("BYTE_WORKER_CACHE_DIR", "byte_worker_cache"),
    model_inventory=[
        item.strip()
        for item in str(os.getenv("BYTE_WORKER_MODELS", "") or "").split(",")
        if item.strip()
    ],
    free_vram_gb=float(os.getenv("BYTE_WORKER_FREE_VRAM_GB", "0") or 0.0),
)
_internal_auth_token = _normalize_internal_token(os.getenv("BYTE_INTERNAL_TOKEN", ""))
app = FastAPI(title=f"{PRODUCT_NAME} Inference Worker", version=__version__)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "service": "byte_inference"}


@app.get("/internal/v1/runtime")
async def runtime_snapshot(request: Request) -> dict[str, Any]:
    _require_internal_auth(request)
    return runtime.heartbeat()


@app.post("/internal/v1/heartbeat")
async def heartbeat(request: Request) -> JSONResponse:
    _require_internal_auth(request)
    return JSONResponse(content=runtime.heartbeat())


@app.post("/internal/v1/prefill")
async def prefill(request: Request, payload: InferencePrefillRequest) -> JSONResponse:
    _require_internal_auth(request)
    return JSONResponse(content=runtime.prefill(payload))


@app.post("/internal/v1/events/kv")
async def kv_event(request: Request, payload: KVEventData) -> JSONResponse:
    _require_internal_auth(request)
    return JSONResponse(content=runtime.record_kv_event(payload))


@app.post("/internal/v1/generate")
async def generate(request: Request, payload: InferenceGenerateRequest) -> JSONResponse:
    _require_internal_auth(request)
    return JSONResponse(content=runtime.generate(payload))


def create_app() -> FastAPI:
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--worker-id", default=os.getenv("BYTE_WORKER_ID", "byte-worker"))
    parser.add_argument("--cache-dir", default=os.getenv("BYTE_WORKER_CACHE_DIR", "byte_worker_cache"))
    parser.add_argument("--models", default=os.getenv("BYTE_WORKER_MODELS", ""))
    parser.add_argument("--free-vram-gb", type=float, default=float(os.getenv("BYTE_WORKER_FREE_VRAM_GB", "0") or 0.0))
    parser.add_argument(
        "--internal-auth-token",
        default=os.getenv("BYTE_INTERNAL_TOKEN", ""),
    )
    args = parser.parse_args()

    global runtime, _internal_auth_token
    runtime = InferenceWorkerRuntime(
        worker_id=str(args.worker_id),
        cache_dir=str(args.cache_dir),
        model_inventory=[item.strip() for item in str(args.models or "").split(",") if item.strip()],
        free_vram_gb=float(args.free_vram_gb or 0.0),
    )
    _internal_auth_token = _normalize_internal_token(args.internal_auth_token)
    uvicorn.run(app, host=args.host, port=args.port)


__all__ = [
    "InferenceWorkerRuntime",
    "_internal_auth_token",
    "app",
    "create_app",
    "main",
    "runtime",
]
