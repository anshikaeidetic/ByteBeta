"""Budget guard -- per-model cost tracking and pricing helpers.

Tracks token usage per model, calculates actual API costs using a built-in
pricing table, and exposes estimation helpers for router decisions.
"""

from collections import defaultdict
from typing import Any

# Pricing per 1M tokens (input, output) in USD.
# These defaults are intentionally conservative and can be overridden per app.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # Meta and Groq/OpenRouter common models
    "llama-3.1-70b": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.59, "output": 0.79},
    # Mistral
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-small-latest": {"input": 0.20, "output": 0.60},
    # Embeddings
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
}


def normalize_model_for_pricing(
    model: str,
    pricing: dict[str, dict[str, float]] | None = None,
) -> str:
    """Normalize a model name for pricing lookup."""
    table = pricing or MODEL_PRICING
    normalized = str(model or "").strip().lower()
    if not normalized:
        return ""
    if normalized in table:
        return normalized
    if "/" in normalized:
        provider, _, raw_model = normalized.partition("/")
        qualified = f"{provider}/{raw_model}"
        if qualified in table:
            return qualified
        normalized = raw_model
        if normalized in table:
            return normalized
    for known in sorted(table.keys(), key=len, reverse=True):
        known_lower = known.lower()
        if normalized.startswith(known_lower):
            return known
        if "/" in known_lower and normalized.startswith(known_lower.split("/", 1)[1]):
            return known
    return normalized


def get_model_pricing(
    model: str,
    *,
    pricing_overrides: dict[str, dict[str, float]] | None = None,
) -> dict[str, float] | None:
    """Return pricing metadata for a model, if known."""
    table = dict(MODEL_PRICING)
    if pricing_overrides:
        table.update(pricing_overrides)
    key = normalize_model_for_pricing(model, table)
    return table.get(key)


def estimate_tokens_for_request(
    request_kwargs: dict[str, Any] | None,
    *,
    fallback_completion_tokens: int = 256,
) -> dict[str, int]:
    """Approximate token usage for a provider-agnostic request."""
    request_kwargs = request_kwargs or {}
    prompt_chars = 0
    messages = request_kwargs.get("messages") or []
    if messages:
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        prompt_chars += len(
                            str(item.get("text", "") or item.get("content", "") or "")
                        )
                    else:
                        prompt_chars += len(str(item or ""))
            else:
                prompt_chars += len(str(content or ""))
    elif request_kwargs.get("prompt") is not None:
        prompt_chars = len(str(request_kwargs.get("prompt") or ""))
    elif request_kwargs.get("input") is not None:
        prompt_chars = len(str(request_kwargs.get("input") or ""))

    prompt_tokens = max(1, int(round(prompt_chars / 4.0))) if prompt_chars else 1
    completion_tokens = int(
        request_kwargs.get("max_tokens")
        or request_kwargs.get("max_output_tokens")
        or fallback_completion_tokens
        or 0
    )
    if completion_tokens <= 0:
        completion_tokens = int(fallback_completion_tokens or 1)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": max(1, completion_tokens),
    }


def estimate_request_cost(
    model: str,
    request_kwargs: dict[str, Any] | None = None,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    pricing_overrides: dict[str, dict[str, float]] | None = None,
) -> float | None:
    """Estimate request cost in USD from explicit or derived token counts."""
    pricing = get_model_pricing(model, pricing_overrides=pricing_overrides)
    if pricing is None:
        return None
    if prompt_tokens is None or completion_tokens is None:
        estimated = estimate_tokens_for_request(request_kwargs)
        prompt_tokens = estimated["prompt_tokens"] if prompt_tokens is None else prompt_tokens
        completion_tokens = (
            estimated["completion_tokens"] if completion_tokens is None else completion_tokens
        )
    input_cost = (int(prompt_tokens or 0) / 1_000_000) * float(pricing["input"])
    output_cost = (int(completion_tokens or 0) / 1_000_000) * float(pricing["output"])
    return round(input_cost + output_cost, 8)


class BudgetTracker:
    """Per-model cost tracker with savings calculation."""

    def __init__(self, custom_pricing: dict[str, dict[str, float]] | None = None) -> None:
        self._pricing = {**MODEL_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)

        self._usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        self._saved: dict[str, dict[str, int]] = defaultdict(
            lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )

    def record_usage(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        model_key = self._normalize_model(model)
        self._usage[model_key]["prompt_tokens"] += prompt_tokens
        self._usage[model_key]["completion_tokens"] += completion_tokens
        self._usage[model_key]["calls"] += 1

    def record_cache_hit(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        model_key = self._normalize_model(model)
        self._saved[model_key]["prompt_tokens"] += prompt_tokens
        self._saved[model_key]["completion_tokens"] += completion_tokens
        self._saved[model_key]["calls"] += 1

    def _cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self._pricing.get(model)
        if not pricing:
            return 0.0
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def _normalize_model(self, model: str) -> str:
        return normalize_model_for_pricing(model, self._pricing)

    def summary(self) -> dict[str, Any]:
        total_spent = 0.0
        total_saved = 0.0
        per_model: dict[str, dict[str, Any]] = {}

        all_models = set(list(self._usage.keys()) + list(self._saved.keys()))
        for model in sorted(all_models):
            usage = self._usage.get(model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            saved = self._saved.get(model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            spent = self._cost(model, usage["prompt_tokens"], usage["completion_tokens"])
            avoided = self._cost(model, saved["prompt_tokens"], saved["completion_tokens"])
            total_spent += spent
            total_saved += avoided
            per_model[model] = {
                "spent_usd": round(spent, 6),
                "saved_usd": round(avoided, 6),
                "llm_calls": usage["calls"],
                "cache_hits": saved["calls"],
                "tokens_used": usage["prompt_tokens"] + usage["completion_tokens"],
                "tokens_saved": saved["prompt_tokens"] + saved["completion_tokens"],
            }

        total = total_spent + total_saved
        return {
            "total_spent_usd": round(total_spent, 4),
            "total_saved_usd": round(total_saved, 4),
            "savings_ratio": round(total_saved / total, 4) if total > 0 else 0.0,
            "per_model": per_model,
        }

    def reset(self) -> None:
        self._usage.clear()
        self._saved.clear()
