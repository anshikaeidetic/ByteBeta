"""Byte router — learned per-query cheap/strong model selection.

Public API:
    RouteLLMScorer  — score a query in [0, 1] (0 = definitely cheap, 1 = definitely strong)
    route_decision  — high-level helper returning (selected_model, score, reason)
"""

from byte.router.route_llm import (
    RouteLLMDecision,
    RouteLLMScorer,
    route_decision,
)

__all__ = [
    "RouteLLMDecision",
    "RouteLLMScorer",
    "route_decision",
]
