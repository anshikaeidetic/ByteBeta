
"""Typed route-decision models for Byte model routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelRouteDecision:
    original_model: str
    selected_model: str
    tier: str
    reason: str
    category: str
    route_key: str
    prompt_chars: int
    message_count: int
    has_tools: bool
    applied: bool
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

__all__ = ["ModelRouteDecision"]
