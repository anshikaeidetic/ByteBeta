"""Routed provider runtime facade for Byte adapters."""

from byte.adapter._router_execution import (
    _aroute_targets as _aroute_targets,
)
from byte.adapter._router_execution import (
    _attempt_target_async as _attempt_target_async,
)
from byte.adapter._router_execution import (
    _attempt_target_sync as _attempt_target_sync,
)
from byte.adapter._router_execution import (
    _route_surface as _route_surface,
)
from byte.adapter._router_execution import (
    _route_targets as _route_targets,
)
from byte.adapter._router_execution import (
    _surface_callable as _surface_callable,
)
from byte.adapter._router_execution import (
    route_completion,
)
from byte.adapter._router_registry import (
    RouteTarget,
    clear_model_aliases,
    clear_route_runtime_stats,
    model_aliases,
    register_model_alias,
    route_runtime_stats,
)
from byte.adapter._router_registry import (
    _RouterRegistry as _RouterRegistry,
)
from byte.adapter._router_resolution import (
    _coerce_targets as _coerce_targets,
)
from byte.adapter._router_resolution import (
    _enrich_target as _enrich_target,
)
from byte.adapter._router_resolution import (
    _provider_request_kwargs as _provider_request_kwargs,
)
from byte.adapter._router_resolution import (
    _supports_surface as _supports_surface,
)
from byte.adapter._router_resolution import (
    _target_from_string as _target_from_string,
)
from byte.adapter._router_resolution import (
    resolve_model_name_for_provider,
    resolve_provider_model,
)
from byte.adapter._router_selection import (
    _annotate_response as _annotate_response,
)
from byte.adapter._router_selection import (
    _flatten_targets as _flatten_targets,
)
from byte.adapter._router_selection import (
    _is_retryable_error as _is_retryable_error,
)
from byte.adapter._router_selection import (
    _order_targets as _order_targets,
)

__all__ = [
    "RouteTarget",
    "clear_model_aliases",
    "clear_route_runtime_stats",
    "model_aliases",
    "register_model_alias",
    "resolve_model_name_for_provider",
    "resolve_provider_model",
    "route_completion",
    "route_runtime_stats",
]
