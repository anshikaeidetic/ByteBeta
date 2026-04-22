"""Per-cache adapter runtime state.

The adapter layer used to keep process-wide singletons in ``adapter.py``.
That made multi-cache and multi-tenant use in one Python process share mutable
state unexpectedly. This module scopes runtime state to each cache instance.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from typing import Any

import cachetools

from byte.processor.adaptive_threshold import AdaptiveThreshold
from byte.processor.budget import BudgetTracker
from byte.processor.coalesce import RequestCoalescer
from byte.processor.fingerprint import ConversationFingerprinter
from byte.processor.quality import QualityScorer


@dataclass
class AdapterRuntimeState:
    embedding_cache: cachetools.LRUCache = field(
        default_factory=lambda: cachetools.LRUCache(maxsize=1)
    )
    embedding_cache_size: int = 1
    coalescer: RequestCoalescer = field(default_factory=lambda: RequestCoalescer(enabled=True))
    adaptive_threshold: AdaptiveThreshold | None = None
    adaptive_threshold_signature: tuple[float, float] | None = None
    budget_tracker: BudgetTracker = field(default_factory=BudgetTracker)
    fingerprinter: ConversationFingerprinter = field(
        default_factory=lambda: ConversationFingerprinter(window_size=4)
    )
    fingerprinter_window_size: int = 4
    quality_scorer: QualityScorer = field(default_factory=QualityScorer)


_STATE_LOCK = threading.RLock()
_RUNTIME_STATES: weakref.WeakKeyDictionary[Any, AdapterRuntimeState] = weakref.WeakKeyDictionary()
_DEFAULT_STATE = AdapterRuntimeState()


def _runtime_owner(chat_cache: Any) -> Any:
    owner = getattr(chat_cache, "__byte_cache_owner__", None)
    if callable(owner):
        try:
            return owner()
        except Exception:  # pylint: disable=W0703
            return chat_cache
    return chat_cache


def get_runtime_state(chat_cache: Any | None = None) -> AdapterRuntimeState:
    """Return the runtime state bound to a cache instance."""
    if chat_cache is None:
        return _DEFAULT_STATE
    chat_cache = _runtime_owner(chat_cache)
    with _STATE_LOCK:
        state = _RUNTIME_STATES.get(chat_cache)
        if state is None:
            state = AdapterRuntimeState()
            _RUNTIME_STATES[chat_cache] = state
        return state


def get_embedding_cache(chat_cache: Any, max_size: int) -> cachetools.LRUCache:
    state = get_runtime_state(chat_cache)
    max_size = max(1, int(max_size or 1))
    if state.embedding_cache.maxsize != max_size or state.embedding_cache_size != max_size:
        state.embedding_cache = cachetools.LRUCache(maxsize=max_size)
        state.embedding_cache_size = max_size
    return state.embedding_cache


def get_coalescer(chat_cache: Any) -> RequestCoalescer:
    return get_runtime_state(chat_cache).coalescer


def get_adaptive_threshold(chat_cache: Any) -> AdaptiveThreshold:
    state = get_runtime_state(chat_cache)
    config = getattr(chat_cache, "config", None)
    signature = (
        float(getattr(config, "similarity_threshold", 0.0) or 0.0),
        float(getattr(config, "target_hit_rate", 0.0) or 0.0),
    )
    if state.adaptive_threshold is None or state.adaptive_threshold_signature != signature:
        state.adaptive_threshold = AdaptiveThreshold(
            base_threshold=signature[0],
            target_hit_rate=signature[1],
        )
        state.adaptive_threshold_signature = signature
    return state.adaptive_threshold


def get_budget_tracker(chat_cache: Any | None = None) -> BudgetTracker:
    return get_runtime_state(chat_cache).budget_tracker


def get_quality_scorer(chat_cache: Any | None = None) -> QualityScorer:
    return get_runtime_state(chat_cache).quality_scorer


def get_fingerprinter(chat_cache: Any, window_size: int) -> ConversationFingerprinter:
    state = get_runtime_state(chat_cache)
    window_size = max(1, int(window_size or 1))
    if (
        state.fingerprinter.window_size != window_size
        or state.fingerprinter_window_size != window_size
    ):
        state.fingerprinter = ConversationFingerprinter(window_size=window_size)
        state.fingerprinter_window_size = window_size
    return state.fingerprinter
