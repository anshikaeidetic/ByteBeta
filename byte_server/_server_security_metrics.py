"""Readiness and metrics payload helpers for the Byte server."""

from __future__ import annotations

from typing import Any

from byte import PRODUCT_SHORT_NAME, __version__
from byte_server._server_state import ServerServices

from ._server_security_auth import _flag


def _readiness_payload(services: ServerServices) -> tuple[bool, dict[str, Any]]:
    def state(cache_obj: Any | None) -> dict[str, Any]:
        data_manager = getattr(cache_obj, "data_manager", None) if cache_obj is not None else None
        similarity = getattr(cache_obj, "similarity_evaluation", None) if cache_obj is not None else None
        return {
            "initialized": bool(cache_obj is not None and getattr(cache_obj, "has_init", False)),
            "data_manager_ready": data_manager is not None,
            "similarity_ready": similarity is not None,
            "embedding_ready": bool(cache_obj is not None and callable(getattr(cache_obj, "embedding_func", None))),
            "data_manager_type": type(data_manager).__name__ if data_manager is not None else "",
            "similarity_type": type(similarity).__name__ if similarity is not None else "",
        }

    runtime = services.runtime_state()
    active_state = state(services.active_cache())
    ready = bool(
        active_state["initialized"]
        and active_state["data_manager_ready"]
        and active_state["similarity_ready"]
        and active_state["embedding_ready"]
    )
    payload = {
        "status": "ready" if ready else "not_ready",
        "service": PRODUCT_SHORT_NAME,
        "version": __version__,
        "gateway_enabled": bool(runtime.gateway_cache is not None),
        "active_cache": active_state,
        "base_cache": state(services.base_cache),
    }
    if runtime.gateway_cache is not None:
        payload["gateway_cache"] = state(runtime.gateway_cache)
    return bool(ready), payload


def _prometheus_metrics_payload(services: ServerServices) -> str:
    ready, readiness = _readiness_payload(services)
    active = services.active_cache()
    report = getattr(active, "report", None)
    quality = active.quality_stats() if active is not None else {}
    router = services.router_registry_summary() or {}
    h2o = services.h2o_runtime_stats()
    control_plane = services.control_plane()
    control_roi = control_plane.store.roi_snapshot() if control_plane is not None else {}
    total_requests = int(getattr(getattr(report, "op_pre", None), "count", 0) or 0) if report else 0
    cache_hits = int(getattr(report, "hint_cache_count", 0) or 0) if report else 0
    cache_misses = max(0, total_requests - cache_hits)
    hit_ratio = (cache_hits / total_requests) if total_requests else 0.0

    # Research counters (process-wide, incremented by subsystems at runtime)
    try:
        from byte.telemetry import (
            research_metrics_snapshot,  # pylint: disable=import-outside-toplevel
        )
        research = research_metrics_snapshot()
    except Exception:  # pragma: no cover - defensive
        research = {}

    # vCache gauges — read directly from the active evaluator if it's a VCacheEvaluation,
    # or the underlying GuardedSimilarityEvaluation wrapping one.
    vcache_error_rate = 0.0
    vcache_cold = 0
    sim_eval = getattr(active, "similarity_evaluation", None) if active is not None else None
    for candidate in (sim_eval, getattr(sim_eval, "base_evaluation", None), getattr(sim_eval, "base", None)):
        if candidate is None:
            continue
        cold_fn = getattr(candidate, "cold_count", None)
        err_fn  = getattr(candidate, "empirical_error_rate", None)
        if callable(cold_fn) and callable(err_fn):
            try:
                vcache_cold = int(cold_fn() or 0)
                vcache_error_rate = float(err_fn() or 0.0)
            except Exception:  # pragma: no cover - defensive
                pass
            break

    # Budget savings (tokens + usd) from the report/budget tracker
    tokens_saved_total = 0
    dollars_saved_total = 0.0
    try:
        cost = active.cost_summary() if active is not None else {}
        budget = cost.get("budget") or {}
        for model_stats in (budget.get("per_model") or {}).values():
            tokens_saved_total  += int(model_stats.get("tokens_saved", 0) or 0)
            dollars_saved_total += float(model_stats.get("saved_usd", 0.0) or 0.0)
    except Exception:  # pragma: no cover - defensive
        pass

    lines = [
        "byteai_cache_up 1",
        f"byteai_cache_ready {_flag(ready)}",
        f"byteai_cache_gateway_enabled {_flag(readiness['gateway_enabled'])}",
        f"byteai_cache_requests_total {total_requests}",
        f"byteai_cache_hits_total {cache_hits}",
        f"byteai_cache_misses_total {cache_misses}",
        f"byteai_cache_hit_ratio {hit_ratio:.6f}",
        f"byteai_cache_quality_total_scored {int(quality.get('total_scored', 0) or 0)}",
        f"byteai_cache_quality_average_score {float(quality.get('avg_quality_score', 0.0) or 0.0):.6f}",
        f"byteai_cache_quality_low_total {int(quality.get('low_quality_count', 0) or 0)}",
        f"byteai_cache_router_aliases {len(router.get('aliases', {}))}",
        f"byteai_cache_tokens_saved_total {tokens_saved_total}",
        f"byteai_cache_dollars_saved_total {dollars_saved_total:.6f}",
        f"byteai_h2o_requests_total {int(h2o.get('requests', 0) or 0)}",
        f"byteai_h2o_applied_total {int(h2o.get('applied', 0) or 0)}",
        f"byteai_h2o_fallbacks_total {int(h2o.get('fallbacks', 0) or 0)}",
        f"byteai_h2o_evictions_total {int(h2o.get('evictions', 0) or 0)}",
        f"byteai_h2o_avg_retained_fraction {float(h2o.get('avg_retained_fraction', 1.0) or 1.0):.6f}",
        f"byteai_control_plane_workers_total {int(control_roi.get('active_workers', 0) or 0)}",
        f"byteai_control_plane_routed_total {int(control_roi.get('worker_routed_requests', 0) or 0)}",
        f"byteai_control_plane_recommendations_total {int(control_roi.get('recommendations', 0) or 0)}",
        f"byteai_control_plane_projected_savings {float(control_roi.get('projected_savings', 0.0) or 0.0):.6f}",
        # ── Research paper counters / gauges (wired by subsystems at runtime) ──
        f"byteai_dual_threshold_reference_hits_total {int(research.get('dual_threshold_reference_hits', 0) or 0)}",
        f"byteai_intent_context_tokens_saved_total {int(research.get('intent_context_tokens_saved', 0) or 0)}",
        f"byteai_llm_equivalence_calls_total {int(research.get('llm_equivalence_calls', 0) or 0)}",
        f"byteai_vcache_threshold_updates_total {int(research.get('vcache_threshold_updates', 0) or 0)}",
        f"byteai_vcache_error_rate_gauge {vcache_error_rate:.6f}",
        f"byteai_vcache_cold_embeddings_total {vcache_cold}",
        f"byteai_recomp_augmentation_hits_total {int(research.get('recomp_augmentation_hits', 0) or 0)}",
        f"byteai_prompt_distillation_calls_total {int(research.get('prompt_distillation_calls', 0) or 0)}",
        f"byteai_prompt_distillation_tokens_saved_total {int(research.get('prompt_distillation_tokens_saved', 0) or 0)}",
        # Byte Router 
        f"byteai_route_llm_decisions_total {int(research.get('route_llm_total', 0) or 0)}",
        f"byteai_route_llm_cheap_selections_total {int(research.get('route_llm_cheap', 0) or 0)}",
        f"byteai_route_llm_strong_selections_total {int(research.get('route_llm_strong', 0) or 0)}",
        # Cost-Aware Eviction 
        f"byteai_eviction_cost_aware_evictions_total {int(research.get('eviction_cost_aware_evictions', 0) or 0)}",
        f"byteai_eviction_cost_aware_savings_total {int(research.get('eviction_cost_aware_savings', 0) or 0)}",
        # LSH Prefilter 
        f"byteai_lsh_prefilter_lookups_total {int(research.get('lsh_prefilter_lookups', 0) or 0)}",
        f"byteai_lsh_prefilter_tier0_hits_total {int(research.get('lsh_prefilter_tier0_hits', 0) or 0)}",
        f"byteai_lsh_prefilter_skipped_searches_total {int(research.get('lsh_prefilter_skipped_searches', 0) or 0)}",
    ]
    if report is not None:
        for operation in ("pre", "embedding", "search", "data", "evaluation", "post", "llm", "save"):
            counter = getattr(report, f"op_{operation}", None)
            if counter is None:
                continue
            lines.append(
                f'byteai_cache_operation_total{{operation="{operation}"}} {int(getattr(counter, "count", 0) or 0)}'
            )
    return "\n".join(lines) + "\n"
