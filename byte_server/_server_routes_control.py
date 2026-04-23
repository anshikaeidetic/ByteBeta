"""Control-plane admin routes for Byte server."""

from __future__ import annotations

from dataclasses import fields as _dc_fields
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from byte._config_sections import _FIELD_TO_SECTION, _SECTION_ATTRS
from byte_server._server_security import _audit_event, _raise_route_error, _require_admin
from byte_server._server_state import ServerServices
from byte_server.models import ControlPlaneSettingsUpdate

# Metadata for Byte-native features exposed via /features endpoint.
_PAPERS_REGISTRY = [
    {"id": "byte-prompt-cache",       "title": "Byte Prompt Recycler — detects and reuses repeated prompt attention states",                "config_toggle": "prompt_module_mode",                 "status": "implemented"},
    {"id": "byte-prompt-distill",     "title": "Byte Prompt Slimmer — query-aware prompt compression saving 50-80% tokens",                "config_toggle": "prompt_distillation_mode",           "status": "implemented"},
    {"id": "byte-distill-core",       "title": "Byte Auto Compressor — general-purpose task-agnostic compression engine",                  "config_toggle": "prompt_distillation_backend",        "status": "implemented"},
    {"id": "byte-context-compiler",   "title": "Byte Context Trimmer — removes irrelevant context for fewer tokens",                       "config_toggle": "context_compiler",                   "status": "implemented"},
    {"id": "byte-retrieval-compress", "title": "Byte Document Filter — RAG context pruning with selective augmentation",                   "config_toggle": "context_compiler",                   "status": "implemented"},
    {"id": "byte-vcache",             "title": "Byte Confidence Cache — verified semantic caching with error-rate guarantee",               "config_toggle": "vcache_enabled",                     "status": "implemented"},
    {"id": "byte-prompt-intern",      "title": "Byte Pattern Learner — internalizes recurrent prompts into model weights",                 "config_toggle": "prompt_distillation_module_mode",    "status": "implemented"},
    {"id": "byte-kv-qjl",             "title": "Byte Memory Saver (QJL) — 1-bit quantized compression for 4-8x memory reduction",         "config_toggle": "kv_codec",                           "status": "implemented"},
    {"id": "byte-kv-polar",           "title": "Byte Precision Compressor — rotational-symmetry high-fidelity quantization",               "config_toggle": "kv_codec",                           "status": "implemented"},
    {"id": "byte-kv-turbo",           "title": "Byte Realtime Compressor — online activation compression during inference",                "config_toggle": "kv_codec",                           "status": "implemented"},
    {"id": "byte-kv-taxonomy",        "title": "Byte Compression Guide — reference framework for compression strategy",                    "config_toggle": None,                                 "status": "taxonomy"},
    {"id": "byte-h2o-eviction",       "title": "Byte Cache Housekeeper — smart eviction keeping high-value entries",                       "config_toggle": "h2o_enabled",                        "status": "implemented"},
    {"id": "byte-intent-context",     "title": "Byte Intent Detector — intent-aware caching and context filtering",                        "config_toggle": "dual_threshold_reference_mode",      "status": "implemented"},
    {"id": "byte-smart-router",       "title": "Byte Cost Router — routes queries to optimal cheap/strong models",                         "config_toggle": "route_llm_enabled",                  "status": "implemented"},
    {"id": "byte-cost-eviction",      "title": "Byte Value-Based Eviction — quality-weighted cache management",                            "config_toggle": "eviction_policy",                    "status": "implemented"},
    {"id": "byte-lsh-prefilter",      "title": "Byte Duplicate Spotter — near-instant duplicate detection via LSH",                        "config_toggle": "lsh_prefilter_enabled",              "status": "implemented"},
    {"id": "byte-cascade",            "title": "Byte Cascade — confidence-gated escalation and stale-entry auto-invalidation",             "config_toggle": "cascade_escalation_enabled",         "status": "implemented"},
    {"id": "byte-reasoning-reuse",    "title": "Byte Reasoning Reuse — deterministic shortcut engine and verified-memory lookup",          "config_toggle": "reasoning_reuse",                    "status": "implemented"},
    {"id": "byte-quality-guard",      "title": "Byte Quality Guard — multi-signal response scoring and admission gate",                    "config_toggle": "response_repair",                    "status": "implemented"},
]


_PRESETS: dict[str, dict] = {
    "off": {
        "vcache_enabled": False,
        "adaptive_threshold": False,
        "dual_threshold_reference_mode": False,
        "llm_equivalence_enabled": False,
        "intent_context_filtering_enabled": False,
        "context_compiler": False,
        "prompt_distillation": False,
        "h2o_enabled": False,
        "kv_codec": "disabled",
        "budget_strategy": "balanced",
        # 2026 additions off by default
        "route_llm_enabled": False,
        "eviction_policy": "LRU",
        "lsh_prefilter_enabled": False,
    },
    "balanced": {
        "vcache_enabled": True, "vcache_delta": 0.05,
        "adaptive_threshold": True, "target_hit_rate": 0.65,
        "context_compiler": True,
        "prompt_distillation": True, "prompt_distillation_mode": "guarded",
        "dual_threshold_reference_mode": True,
        "intent_context_filtering_enabled": True, "intent_context_budget_ratio": 0.6,
        "cache_latency_guard": True,
        "budget_strategy": "balanced",
        "llm_equivalence_enabled": False, "h2o_enabled": False, "kv_codec": "disabled",
        # Byte Router  — cheap/strong router with a balanced 0.5 threshold
        "route_llm_enabled": True, "route_llm_threshold": 0.5,
        # LSH prefilter 
        "lsh_prefilter_enabled": True, "lsh_num_perm": 128, "lsh_threshold": 0.6,
        # LRU stays for Balanced — cost-aware eviction only kicks in for Savings/Accuracy
        "eviction_policy": "LRU",
    },
    "savings": {
        "vcache_enabled": True, "vcache_delta": 0.08,
        "adaptive_threshold": True, "target_hit_rate": 0.75,
        "context_compiler": True,
        "prompt_distillation": True, "prompt_distillation_mode": "enabled",
        "prompt_distillation_budget_ratio": 0.45,
        "dual_threshold_reference_mode": True,
        "intent_context_filtering_enabled": True, "intent_context_budget_ratio": 0.5,
        "cache_latency_guard": True,
        "budget_strategy": "lowest_cost",
        "llm_equivalence_enabled": False, "h2o_enabled": False, "kv_codec": "disabled",
        # Byte Router — aggressive cheap routing with a low threshold
        "route_llm_enabled": True, "route_llm_threshold": 0.35,
        # Cost-aware eviction keeps the highest-value entries longer
        "eviction_policy": "COST_AWARE",
        # LSH prefilter at a looser threshold to catch more near-duplicates
        "lsh_prefilter_enabled": True, "lsh_num_perm": 128, "lsh_threshold": 0.55,
    },
    "accuracy": {
        "vcache_enabled": True, "vcache_delta": 0.02, "vcache_cold_fallback_threshold": 0.85,
        "adaptive_threshold": True, "target_hit_rate": 0.50,
        "context_compiler": True,
        "prompt_distillation": True, "prompt_distillation_mode": "guarded",
        "llm_equivalence_enabled": True,
        "dual_threshold_reference_mode": False,
        "intent_context_filtering_enabled": True, "intent_context_budget_ratio": 0.65,
        "cache_latency_guard": True,
        "budget_strategy": "quality_first",
        "evidence_verification": True,
        "h2o_enabled": False, "kv_codec": "disabled",
        # Byte Router — conservative threshold keeps more queries on the strong model
        "route_llm_enabled": True, "route_llm_threshold": 0.65,
        # Cost-aware eviction + strict LSH (only clear near-duplicates short-circuit)
        "eviction_policy": "COST_AWARE",
        "lsh_prefilter_enabled": True, "lsh_num_perm": 256, "lsh_threshold": 0.75,
    },
}


def _control_plane(services: ServerServices) -> Any:
    runtime = services.control_plane()
    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="The Byte control plane is not configured on this server.",
        )
    return runtime


def register_control_routes(app: FastAPI, services: ServerServices) -> None:
    @app.get("/byte/control/cache/inspect")
    async def cache_inspect(request: Request, limit: int = 50) -> Any:
        _require_admin(services, request, "control.inspect")
        runtime = _control_plane(services)
        payload = {
            "events": runtime.store.list_cache_events(limit=limit),
            "runtime": runtime.inspect(),
        }
        _audit_event(
            services,
            request,
            "control.inspect",
            status="success",
            metadata={"limit": limit},
        )
        return JSONResponse(content=payload)

    @app.get("/byte/control/replays")
    async def control_replays(request: Request, limit: int = 50) -> Any:
        _require_admin(services, request, "control.replays")
        runtime = _control_plane(services)
        payload = {"replays": runtime.store.list_replays(limit=limit)}
        _audit_event(
            services,
            request,
            "control.replays",
            status="success",
            metadata={"limit": limit},
        )
        return JSONResponse(content=payload)

    @app.get("/byte/control/recommendations")
    async def control_recommendations(request: Request) -> Any:
        _require_admin(services, request, "control.recommendations")
        runtime = _control_plane(services)
        payload = {"recommendations": runtime.store.list_recommendations()}
        _audit_event(services, request, "control.recommendations", status="success")
        return JSONResponse(content=payload)

    @app.get("/byte/control/intent-graph")
    async def control_intent_graph(request: Request, limit: int = 25) -> Any:
        _require_admin(services, request, "control.intent_graph")
        runtime = _control_plane(services)
        payload = runtime.store.intent_graph_snapshot(limit=limit)
        _audit_event(
            services,
            request,
            "control.intent_graph",
            status="success",
            metadata={"limit": limit},
        )
        return JSONResponse(content=payload)

    @app.get("/byte/control/roi")
    async def control_roi(request: Request) -> Any:
        _require_admin(services, request, "control.roi")
        runtime = _control_plane(services)
        payload = runtime.store.roi_snapshot()
        _audit_event(services, request, "control.roi", status="success")
        return JSONResponse(content=payload)

    @app.post("/byte/control/settings/features")
    async def control_feature_settings(request: Request, payload: ControlPlaneSettingsUpdate) -> Any:
        _require_admin(services, request, "control.settings")
        runtime = _control_plane(services)
        try:
            updated = runtime.update_feature_flags(
                {key: value for key, value in payload.model_dump().items() if value is not None}
            )
            runtime.refresh_workers()
            _audit_event(services, request, "control.settings", status="success")
            return JSONResponse(content={"settings": updated})
        except Exception as exc:  # pragma: no cover - defensive
            _raise_route_error(
                services,
                request,
                "control.settings",
                exc,
                public_detail="Updating Byte control-plane settings failed.",
            )


def register_config_routes(app: FastAPI, services: ServerServices) -> None:
    """Register GET /config, PATCH /config, GET /papers, DELETE /cache/entry."""

    @app.get("/config")
    async def get_config(request: Request) -> Any:
        _require_admin(services, request, "config.read")
        cache_obj = services.active_cache()
        cfg = getattr(cache_obj, "config", None)
        if cfg is None:
            raise HTTPException(status_code=503, detail="Cache config not available.")
        if callable(getattr(cfg, "to_flat_dict", None)):
            flat = cfg.to_flat_dict()
        else:
            flat = {}
            for section_attr in _SECTION_ATTRS.values():
                section = object.__getattribute__(cfg, section_attr)
                for f in _dc_fields(section):
                    flat[f.name] = getattr(section, f.name)
        _audit_event(services, request, "config.read", status="success")
        return JSONResponse(content=flat)

    @app.patch("/config")
    async def patch_config(request: Request) -> Any:
        _require_admin(services, request, "config.patch")
        body = await request.json()
        if not isinstance(body, dict) or not body:
            raise HTTPException(status_code=400, detail="Request body must be a non-empty JSON object.")
        cache_obj = services.active_cache()
        cfg = getattr(cache_obj, "config", None)
        if cfg is None:
            raise HTTPException(status_code=503, detail="Cache config not available.")

        unknown = [k for k in body if k not in _FIELD_TO_SECTION]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown config fields: {unknown}")

        updated: dict[str, Any] = {}
        try:
            for field_name, value in body.items():
                setattr(cfg, field_name, value)
                updated[field_name] = value
        except Exception as exc:  # pylint: disable=W0703
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Propagate to control plane workers if available
        runtime = services.runtime_state().control_plane_runtime
        if runtime is not None:
            try:
                runtime.update_feature_flags(updated)
                runtime.refresh_workers()
            except Exception:  # pylint: disable=W0703
                pass

        _audit_event(services, request, "config.patch", status="success", metadata={"fields": list(updated)})
        return JSONResponse(content={"updated": updated})

    @app.get("/papers")
    async def get_papers(request: Request) -> Any:
        return JSONResponse(content={"features": _PAPERS_REGISTRY, "total": len(_PAPERS_REGISTRY)})

    @app.get("/features")
    async def get_features(request: Request) -> Any:
        return JSONResponse(content={"features": _PAPERS_REGISTRY, "total": len(_PAPERS_REGISTRY)})

    @app.delete("/cache/entry")
    async def delete_cache_entry(request: Request) -> Any:
        _require_admin(services, request, "cache.invalidate")
        body = await request.json()
        query = body.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="'query' field is required.")
        cache_obj = services.active_cache()
        try:
            removed = cache_obj.invalidate_by_query(query)
        except Exception as exc:  # pylint: disable=W0703
            _raise_route_error(services, request, "cache.invalidate", exc, public_detail="Cache entry deletion failed.")
        _audit_event(services, request, "cache.invalidate", status="success", metadata={"query": query[:120]})
        return JSONResponse(content={"removed": removed, "query": query})

    @app.get("/config/presets")
    async def get_presets(request: Request) -> Any:
        return JSONResponse(content={
            "presets": [
                {"id": "off",      "label": "Off",         "description": "No automatic optimizations."},
                {"id": "balanced", "label": "Balanced",     "description": "Best overall: savings, speed, and accuracy."},
                {"id": "savings",  "label": "Max Savings",  "description": "Maximize cache hit rate and token reduction."},
                {"id": "accuracy", "label": "Max Accuracy", "description": "Minimize false cache hits with LLM-verified matches."},
            ]
        })

    @app.post("/config/preset")
    async def apply_preset(request: Request) -> Any:
        _require_admin(services, request, "config.preset")
        body = await request.json()
        preset_id = body.get("preset", "")
        if preset_id not in _PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown preset '{preset_id}'. Available: {list(_PRESETS)}")
        fields = _PRESETS[preset_id]
        cache_obj = services.active_cache()
        cfg = getattr(cache_obj, "config", None)
        if cfg is None:
            raise HTTPException(status_code=503, detail="Cache config not available.")
        for field_name, value in fields.items():
            try:
                setattr(cfg, field_name, value)
            except Exception:  # pylint: disable=W0703
                pass
        runtime = services.runtime_state().control_plane_runtime
        if runtime is not None:
            try:
                runtime.update_feature_flags(fields)
                runtime.refresh_workers()
            except Exception:  # pylint: disable=W0703
                pass
        _audit_event(services, request, "config.preset", status="success", metadata={"preset": preset_id})
        return JSONResponse(content={"preset": preset_id, "applied": fields})


__all__ = ["register_config_routes", "register_control_routes"]
