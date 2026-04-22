"""Administrative, cache, and health routes for the Byte server."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from byte import PRODUCT_SHORT_NAME, __version__
from byte.adapter.api import (
    export_memory_artifact,
    export_memory_snapshot,
    get,
    import_memory_artifact,
    import_memory_snapshot,
    put,
)
from byte.security import ensure_byte_limit
from byte.utils.error import CacheError
from byte_server._server_security import (
    _admin_auth_required,
    _audit_event,
    _cache_file_disabled,
    _prometheus_metrics_payload,
    _readiness_payload,
    _require_admin,
    _resolve_artifact_path,
    _security_max_archive_bytes,
    _security_max_archive_members,
    short_digest,
)
from byte_server._server_state import ServerServices
from byte_server.models import (
    CacheData,
    FeedbackData,
    MemoryArtifactExportData,
    MemoryArtifactImportData,
    MemoryImportData,
    WarmData,
)


def register_cache_routes(app: FastAPI, services: ServerServices) -> None:
    _demo_path = Path(os.environ.get("BYTE_DEMO_PATH", "") or "demo.html")
    if not _demo_path.is_absolute():
        _demo_path = Path.cwd() / _demo_path

    @app.get("/", include_in_schema=False)
    @app.get("/demo", include_in_schema=False)
    @app.get("/demo.html", include_in_schema=False)
    async def root() -> Response:
        if _demo_path.is_file():
            return FileResponse(str(_demo_path), media_type="text/html")
        return Response(
            content=f"{PRODUCT_SHORT_NAME} v{__version__} — gateway is running",
            media_type="text/plain",
        )

    async def _health_payload() -> dict:
        ready, readiness = _readiness_payload(services)
        return {
            "status": "ok",
            "service": PRODUCT_SHORT_NAME,
            "version": __version__,
            "ready": ready,
            "readiness": readiness,
        }

    @app.get("/healthz")
    async def healthz() -> dict:
        return await _health_payload()

    # Google Frontend intercepts /healthz on some hosting platforms (e.g. Cloud Run)
    # before it reaches the container, so /ping is exposed as a non-reserved alias.
    @app.get("/ping")
    async def ping() -> dict:
        return await _health_payload()

    @app.get("/readyz")
    async def readyz() -> Response:
        ready, payload = _readiness_payload(services)
        return Response(
            content=json.dumps(payload),
            status_code=200 if ready else 503,
            media_type="application/json",
        )

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        if _admin_auth_required(services):
            _require_admin(services, request, "observability.metrics")
        _audit_event(services, request, "observability.metrics", status="success")
        return Response(
            content=_prometheus_metrics_payload(services),
            media_type=services.prometheus_content_type,
        )

    @app.post("/put")
    async def put_cache(request: Request, cache_data: CacheData) -> str:
        _require_admin(services, request, "cache.put")
        put(cache_data.prompt, cache_data.answer)
        _audit_event(
            services,
            request,
            "cache.put",
            status="success",
            metadata={"query_digest": short_digest(cache_data.prompt)},
        )
        return "successfully update the cache"

    @app.post("/get")
    async def get_cache(request: Request, cache_data: CacheData) -> CacheData:
        _require_admin(services, request, "cache.get")
        result = get(cache_data.prompt)
        _audit_event(
            services,
            request,
            "cache.get",
            status="success",
            metadata={"query_digest": short_digest(cache_data.prompt)},
        )
        return CacheData(prompt=cache_data.prompt, answer=result)

    @app.post("/flush")
    async def flush_cache(request: Request) -> str:
        _require_admin(services, request, "cache.flush")
        services.active_cache().flush()
        _audit_event(services, request, "cache.flush", status="success")
        return "successfully flush the cache"

    @app.post("/invalidate")
    async def invalidate_cache(request: Request, cache_data: CacheData) -> dict:
        _require_admin(services, request, "cache.invalidate")
        deleted = services.active_cache().invalidate_by_query(cache_data.prompt)
        _audit_event(
            services,
            request,
            "cache.invalidate",
            status="success",
            metadata={"query_digest": short_digest(cache_data.prompt), "deleted": bool(deleted)},
        )
        response = {"deleted": deleted, "query_digest": short_digest(cache_data.prompt)}
        if not getattr(services.active_cache().config, "security_mode", False):
            response["query"] = cache_data.prompt
        return response

    @app.post("/clear")
    async def clear_cache(request: Request) -> str:
        _require_admin(services, request, "cache.clear")
        services.active_cache().clear()
        _audit_event(services, request, "cache.clear", status="success")
        return "successfully cleared all cache entries"

    @app.get("/cache/mode")
    async def get_cache_mode() -> dict:
        runtime = services.runtime_state()
        return {
            "mode": getattr(runtime, "gateway_cache_mode", "") or "",
            "available": ["exact", "normalized", "semantic", "hybrid"],
        }

    @app.post("/cache/mode")
    async def set_cache_mode(request: Request) -> dict:
        _require_admin(services, request, "cache.mode")
        try:
            body = await request.json()
        except Exception:
            body = {}
        requested = str((body or {}).get("mode", "")).lower().strip()
        allowed = {"exact", "normalized", "semantic", "hybrid"}
        if requested not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid mode '{requested}'. Allowed: {sorted(allowed)}")

        runtime = services.runtime_state()
        cache_obj = runtime.gateway_cache
        if cache_obj is None:
            raise HTTPException(status_code=409, detail="Gateway cache is not enabled.")

        # Re-initialize the existing cache object's internals with the new mode.
        # Importing here avoids a circular import at module load.
        from byte_server._server_gateway import _init_gateway_cache as _reinit_gateway_cache

        try:
            _reinit_gateway_cache(mode=requested, cache_dir="byte_gateway_cache", cache_obj=cache_obj)
            # Flush stale entries so the new similarity evaluator starts clean.
            try:
                cache_obj.clear()
            except Exception:
                pass
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Mode change failed: {exc}") from exc

        # Update the module-level global so sync_runtime_state propagates it.
        import sys as _sys
        _srv = _sys.modules.get("byte_server.server")
        if _srv is not None:
            _srv.gateway_cache_mode = requested

        _audit_event(
            services,
            request,
            "cache.mode",
            status="success",
            metadata={"mode": requested},
        )
        return {"mode": requested, "status": "ok"}

    @app.get("/stats")
    async def cache_stats(request: Request) -> dict:
        _require_admin(services, request, "cache.stats")
        _audit_event(services, request, "cache.stats", status="success")
        return services.active_cache().cost_summary()

    @app.post("/warm")
    async def warm_cache(request: Request, warm_data: WarmData) -> dict:
        _require_admin(services, request, "cache.warm")
        result = services.active_cache().warm(warm_data.data)
        _audit_event(
            services,
            request,
            "cache.warm",
            status="success",
            metadata={"items": len(warm_data.data or [])},
        )
        return result

    @app.post("/feedback")
    async def cache_feedback(request: Request, fb: FeedbackData) -> dict:
        _require_admin(services, request, "cache.feedback")
        result = services.active_cache().feedback(fb.query, fb.thumbs_up)
        _audit_event(
            services,
            request,
            "cache.feedback",
            status="success",
            metadata={"query_digest": short_digest(fb.query), "thumbs_up": bool(fb.thumbs_up)},
        )
        return result

    @app.get("/quality")
    async def cache_quality(request: Request) -> dict:
        _require_admin(services, request, "cache.quality")
        _audit_event(services, request, "cache.quality", status="success")
        return services.active_cache().quality_stats()

    @app.get("/memory")
    async def memory_snapshot(request: Request, limit: int = 20) -> dict:
        _require_admin(services, request, "memory.snapshot")
        result = export_memory_snapshot(cache_obj=services.active_cache(), tool_result_limit=limit)
        _audit_event(services, request, "memory.snapshot", status="success", metadata={"limit": limit})
        return result

    @app.get("/memory/recent")
    async def memory_recent(request: Request, limit: int = 10) -> dict:
        _require_admin(services, request, "memory.recent")
        result = {
            "entries": services.active_cache().recent_interactions(limit=limit),
            "stats": services.active_cache().ai_memory_stats(),
        }
        _audit_event(services, request, "memory.recent", status="success", metadata={"limit": limit})
        return result

    @app.post("/memory/import")
    async def memory_import(request: Request, payload: MemoryImportData) -> dict:
        _require_admin(services, request, "memory.import")
        result = import_memory_snapshot(payload.snapshot, cache_obj=services.active_cache())
        _audit_event(services, request, "memory.import", status="success")
        return result

    @app.post("/memory/export_artifact")
    async def memory_export_artifact(request: Request, payload: MemoryArtifactExportData) -> dict:
        _require_admin(services, request, "memory.export_artifact")
        target_path = _resolve_artifact_path(services, payload.path)
        result = export_memory_artifact(
            target_path,
            cache_obj=services.active_cache(),
            format=payload.format,
            tool_result_limit=payload.limit,
        )
        _audit_event(
            services,
            request,
            "memory.export_artifact",
            status="success",
            metadata={"path_digest": short_digest(target_path), "format": payload.format or ""},
        )
        return result

    @app.post("/memory/import_artifact")
    async def memory_import_artifact(request: Request, payload: MemoryArtifactImportData) -> dict:
        _require_admin(services, request, "memory.import_artifact")
        target_path = _resolve_artifact_path(services, payload.path)
        result = import_memory_artifact(
            target_path,
            cache_obj=services.active_cache(),
            format=payload.format,
        )
        _audit_event(
            services,
            request,
            "memory.import_artifact",
            status="success",
            metadata={"path_digest": short_digest(target_path), "format": payload.format or ""},
        )
        return result

    @app.get("/cache_file")
    async def get_cache_file(request: Request, key: str = "") -> Response:
        _require_admin(services, request, "cache.download")
        runtime = services.runtime_state()
        if _cache_file_disabled(services):
            _audit_event(services, request, "cache.download", status="denied", metadata={"reason": "cache_file_disabled"})
            raise HTTPException(status_code=403, detail="The cache_file endpoint is disabled by ByteAI Cache security policy.")
        if runtime.cache_dir == "":
            raise HTTPException(status_code=403, detail="the server.cache_dir was not specified when the service was initialized")
        if runtime.cache_file_key == "":
            raise HTTPException(status_code=403, detail="the cache file can't be downloaded because the cache-file-key was not specified")
        if runtime.cache_file_key != key:
            raise HTTPException(status_code=403, detail="the cache file key is wrong")
        archive = io.BytesIO()
        file_count = 0
        total_source_bytes = 0
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(runtime.cache_dir):
                for file in files:
                    source_path = os.path.join(root, file)
                    file_count += 1
                    if file_count > _security_max_archive_members(services):
                        raise HTTPException(status_code=413, detail="Cache download archive exceeds Byte's configured file-count limit.")
                    total_source_bytes += os.path.getsize(source_path)
                    try:
                        ensure_byte_limit(total_source_bytes, limit=_security_max_archive_bytes(services), label="Cache download archive")
                    except CacheError as exc:
                        raise HTTPException(status_code=413, detail=str(exc)) from exc
                    zipf.write(source_path, arcname=os.path.relpath(source_path, start=runtime.cache_dir))
        archive_bytes = archive.getvalue()
        try:
            ensure_byte_limit(len(archive_bytes), limit=_security_max_archive_bytes(services), label="Cache download archive")
        except CacheError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        filename = f"{Path(runtime.cache_dir).name or 'byte_cache'}.zip"
        _audit_event(services, request, "cache.download", status="success", metadata={"path_digest": short_digest(filename)})
        return Response(
            content=archive_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


__all__ = ["register_cache_routes"]
