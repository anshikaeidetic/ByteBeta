"""Canonical Byte adapter surface."""

from byte.adapter.api import (
    clear_router_alias_registry as clear_routes,
)
from byte.adapter.api import (
    export_memory_artifact,
    export_memory_snapshot,
    import_memory_artifact,
    import_memory_snapshot,
)
from byte.adapter.api import (
    register_router_alias as register_route,
)
from byte.adapter.api import (
    router_registry_summary as route_summary,
)
from byte.adapter.unified import Audio, ChatCompletion, Completion, Image, Moderation, Speech

__all__ = [
    "Audio",
    "ChatCompletion",
    "Completion",
    "Image",
    "Moderation",
    "Speech",
    "clear_routes",
    "export_memory_artifact",
    "export_memory_snapshot",
    "import_memory_artifact",
    "import_memory_snapshot",
    "register_route",
    "route_summary",
]
