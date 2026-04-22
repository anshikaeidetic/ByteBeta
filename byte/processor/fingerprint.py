"""Conversation-Aware Semantic Fingerprinting.

Hashes the last N turns of conversation into a weighted semantic
fingerprint that becomes part of the cache key. Two identical
"Tell me more" requests with different conversation histories
correctly cache-miss.
"""

import hashlib
import json
from typing import Any


class ConversationFingerprinter:
    """Generates context-aware fingerprints for multi-turn conversations.

    Instead of treating each request independently, the fingerprinter
    creates a weighted hash of recent conversation turns so semantically
    identical surface queries with different histories produce different
    cache keys.

    :param window_size: number of recent turns to include (default 3)
    :param decay_factor: exponential weight decay for older turns (default 0.5)
    """

    def __init__(self, window_size: int = 3, decay_factor: float = 0.5) -> None:
        self.window_size = max(1, window_size)
        self.decay_factor = decay_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fingerprint(self, messages: list[dict[str, Any]]) -> str:
        """Return a deterministic fingerprint string for a conversation.

        The fingerprint encodes:
        - The most recent user message (weight=1.0)
        - Previous turns with exponentially decaying weights
        - Role information (system/user/assistant) so identical text
          from different roles doesn't collide

        :param messages: list of message dicts, each with "role" and "content"
        :return: hex digest string
        """
        if not messages:
            return ""

        # Take the last `window_size` messages
        window = messages[-self.window_size :]

        parts: list[str] = []
        n = len(window)
        for i, msg in enumerate(window):
            # Weight: newest message (last) gets 1.0, older ones decay
            age = n - 1 - i  # 0 for newest
            weight = self.decay_factor**age
            role = msg.get("role", "user")
            content = self._normalise(msg.get("content", ""))
            # Encode weight into the hash input so different weights
            # produce different fingerprints even for same content
            parts.append(f"{weight:.4f}|{role}|{content}")

        combined = "\n".join(parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def context_key(self, messages: list[dict[str, Any]]) -> str:
        """Return a short context key suitable for cache key composition.

        This is a truncated version of the fingerprint (16 hex chars)
        designed to be concatenated with the main cache key.

        :param messages: conversation message list
        :return: 16-char hex string, or "" for single-turn
        """
        if not messages or len(messages) <= 1:
            return ""
        fp = self.fingerprint(messages)
        return fp[:16]

    def enrich_pre_embedding(
        self,
        current_query: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """Combine current query with a context fingerprint.

        Returns a composite string that can be used as pre-embedding data.
        For single-turn requests, returns the query unchanged.

        :param current_query: the current user query text
        :param messages: full conversation message list
        :return: enriched query string
        """
        ctx = self.context_key(messages)
        if not ctx:
            return current_query
        return f"{current_query}||ctx:{ctx}"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Lightweight normalisation: lowercase, strip whitespace."""
        if not isinstance(text, str):
            text = str(text)
        return text.strip().lower()


def selective_payload_fingerprint(
    payload: dict[str, Any],
    fields: list[str],
    *,
    max_chars: int = 2048,
) -> str:
    """Build a stable fingerprint from selected request fields."""
    if not payload or not fields:
        return ""

    selected = {}
    for field in fields:
        value = payload.get(field)
        if value is not None:
            selected[str(field)] = _fingerprint_value(value, max_chars=max_chars)

    if not selected:
        return ""

    normalized = json.dumps(selected, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _fingerprint_value(value: Any, *, max_chars: int) -> dict[str, Any]:
    normalized = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    preview = normalized if len(normalized) <= max_chars else normalized[:max_chars]
    return {
        "digest": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "size": len(normalized),
        "preview": preview,
    }
