"""Byte router — per-query cheap/strong model selection.

Public API:
    ByteRouterScorer  — score a query in [0, 1] (0 = definitely cheap, 1 = definitely strong)
    route_decision    — high-level helper returning (selected_model, score, reason)
"""

from byte.router.route_llm import (
    ByteRouterDecision,
    ByteRouterScorer,
    route_decision,
)

# Back-compat aliases so legacy imports keep working.
RouteLLMDecision = ByteRouterDecision
RouteLLMScorer = ByteRouterScorer

__all__ = [
    "ByteRouterDecision",
    "ByteRouterScorer",
    "RouteLLMDecision",
    "RouteLLMScorer",
    "route_decision",
]
