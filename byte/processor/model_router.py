
"""Compatibility facade for Byte model routing."""

from __future__ import annotations

from byte.processor._model_router_policy import route_request_model
from byte.processor._model_router_tracker import (
    RoutePerformanceTracker,
    clear_route_performance,
    record_route_outcome,
    route_performance_stats,
)
from byte.processor._model_router_types import ModelRouteDecision

__all__ = [
    "ModelRouteDecision",
    "RoutePerformanceTracker",
    "clear_route_performance",
    "record_route_outcome",
    "route_performance_stats",
    "route_request_model",
]
