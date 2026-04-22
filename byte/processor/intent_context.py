"""Intent-driven context filtering for token-efficient prompt assembly.

Implements arXiv 2601.11687 (Harmohit Singh, January 2026, "Semantic Caching
and Intent-Driven Context Optimization for Multi-Agent Natural Language to
Code Systems"). Reduces context window token consumption 40-60% by filtering
to intent-relevant schema elements and document chunks.
"""

from __future__ import annotations

from typing import Any

from byte.utils.log import byte_log

# Intent labels
INTENT_LOOKUP = "lookup"
INTENT_AGGREGATION = "aggregation"
INTENT_JOIN = "join"
INTENT_COMPARISON = "comparison"
INTENT_GENERATION = "generation"

_INTENT_KEYWORDS: dict[str, list[str]] = {
    INTENT_LOOKUP: [
        "find", "get", "fetch", "retrieve", "show", "list", "select", "what is",
        "which", "who", "where", "when",
    ],
    INTENT_AGGREGATION: [
        "count", "sum", "average", "avg", "total", "max", "min", "group by",
        "how many", "how much", "aggregate",
    ],
    INTENT_JOIN: [
        "join", "merge", "combine", "link", "relate", "relationship", "with",
        "along with", "together with",
    ],
    INTENT_COMPARISON: [
        "compare", "difference", "versus", "vs", "better", "worse", "higher", "lower",
        "more than", "less than", "between",
    ],
    INTENT_GENERATION: [
        "generate", "create", "write", "build", "implement", "code", "function",
        "class", "script", "template",
    ],
}

# Intent-to-relevant-field heuristics: which message roles/keywords matter most per intent
_INTENT_RELEVANCE_BOOSTS: dict[str, list[str]] = {
    INTENT_LOOKUP: ["schema", "table", "column", "field", "index"],
    INTENT_AGGREGATION: ["schema", "table", "column", "numeric", "count", "sum"],
    INTENT_JOIN: ["schema", "table", "foreign key", "relation", "join"],
    INTENT_COMPARISON: ["metric", "value", "column", "table", "benchmark"],
    INTENT_GENERATION: ["example", "template", "function", "class", "import"],
}


def classify_intent(query: str) -> str:
    """Classify query intent using keyword heuristics.

    Returns one of: lookup, aggregation, join, comparison, generation.
    """
    query_lower = query.lower()
    scores: dict[str, int] = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        scores[intent] = sum(1 for kw in keywords if kw in query_lower)
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return INTENT_GENERATION  # default to generation for open-ended queries
    return best


def _score_message_relevance(message: dict[str, Any], intent: str) -> float:
    """Score how relevant a single message is for the given intent (0.0 – 1.0)."""
    boost_words = _INTENT_RELEVANCE_BOOSTS.get(intent, [])
    content = ""
    if isinstance(message.get("content"), str):
        content = message["content"].lower()
    elif isinstance(message.get("content"), list):
        content = " ".join(
            part.get("text", "") for part in message["content"] if isinstance(part, dict)
        ).lower()

    if not content:
        return 0.3  # neutral score for empty/unknown

    matches = sum(1 for word in boost_words if word in content)
    # Longer, more relevant messages score higher; cap at 1.0
    length_factor = min(1.0, len(content) / 500)
    return min(1.0, 0.2 + 0.5 * (matches / max(len(boost_words), 1)) + 0.3 * length_factor)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def filter_context(
    messages: list[dict[str, Any]],
    intent_label: str,
    budget_ratio: float = 0.6,
) -> tuple[list[dict[str, Any]], int]:
    """Filter message list to the top `budget_ratio` fraction by intent relevance.

    Returns (filtered_messages, tokens_removed).
    """
    if not messages or budget_ratio >= 1.0:
        return messages, 0

    # Always preserve the last user message (the actual query)
    keep_indices = {len(messages) - 1}

    scored: list[tuple[int, float, int]] = []  # (index, score, tokens)
    for i, msg in enumerate(messages):
        if i in keep_indices:
            continue
        content_str = ""
        if isinstance(msg.get("content"), str):
            content_str = msg["content"]
        elif isinstance(msg.get("content"), list):
            content_str = " ".join(
                part.get("text", "") for part in msg["content"] if isinstance(part, dict)
            )
        score = _score_message_relevance(msg, intent_label)
        tokens = _estimate_tokens(content_str)
        scored.append((i, score, tokens))

    # Sort by score descending; keep top budget_ratio fraction
    scored.sort(key=lambda x: x[1], reverse=True)
    keep_count = max(1, int(len(scored) * budget_ratio))
    kept_indices = {idx for idx, _, _ in scored[:keep_count]} | keep_indices

    tokens_removed = sum(
        tokens for idx, _, tokens in scored if idx not in kept_indices
    )

    filtered = [msg for i, msg in enumerate(messages) if i in kept_indices]
    # Preserve original order
    filtered.sort(key=lambda m: messages.index(m) if m in messages else 0)

    return filtered, tokens_removed


class IntentDrivenContextFilter:
    """Intent-driven context pruning (arXiv 2601.11687).

    Classifies incoming query intent (lookup/aggregation/join/comparison/generation)
    and filters the context window to only include schema elements, tool definitions,
    or document chunks relevant to that intent. Reduces token consumption 40-60%.

    Integration: called from the pre-context processing stage when
    intent_context_filtering_enabled=True.
    """

    def __init__(
        self,
        budget_ratio: float = 0.6,
        cache_intent_labels: bool = True,
    ) -> None:
        self.budget_ratio = budget_ratio
        self.cache_intent_labels = cache_intent_labels

    def apply(self, kwargs: dict[str, Any], context: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Apply intent-driven filtering to the request kwargs.

        Returns (updated_kwargs, tokens_removed).
        """
        messages = kwargs.get("messages")
        if not messages or not isinstance(messages, list):
            return kwargs, 0

        # Classify intent from the last user message
        last_user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_query = content
                elif isinstance(content, list):
                    last_user_query = " ".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                break

        if not last_user_query:
            return kwargs, 0

        intent = classify_intent(last_user_query)

        if self.cache_intent_labels:
            context["_byte_intent_label"] = intent

        filtered_messages, tokens_removed = filter_context(
            messages, intent, self.budget_ratio
        )

        if tokens_removed > 0:
            byte_log.debug(
                "IntentDrivenContextFilter: intent=%s, removed ~%d tokens",
                intent,
                tokens_removed,
            )
            try:
                from byte.telemetry import (
                    bump_research_counter as _bump,  # pylint: disable=import-outside-toplevel
                )
                _bump("intent_context_tokens_saved", int(tokens_removed))
            except Exception:  # pragma: no cover - defensive
                pass
            updated = dict(kwargs)
            updated["messages"] = filtered_messages
            return updated, tokens_removed

        return kwargs, 0
