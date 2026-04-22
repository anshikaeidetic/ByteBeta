"""Registry state and target identity helpers for routed provider execution."""

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

from byte.utils.error import CacheError

_PROVIDER_MODULES = {
    "openai": "byte._backends.openai",
    "deepseek": "byte._backends.deepseek",
    "anthropic": "byte._backends.anthropic",
    "gemini": "byte._backends.gemini",
    "groq": "byte._backends.groq",
    "openrouter": "byte._backends.openrouter",
    "ollama": "byte._backends.ollama",
    "mistral": "byte._backends.mistral",
    "cohere": "byte._backends.cohere",
    "bedrock": "byte._backends.bedrock",
    "huggingface": "byte._backends.huggingface",
}

_SURFACE_CAPABILITIES = {
    "chat_completion": "chat_completion",
    "text_completion": "text_completion",
    "image": "image_generation",
    "audio_transcribe": "audio_transcription",
    "audio_translate": "audio_translation",
    "speech": "speech_generation",
    "moderation": "moderation",
}


@dataclass(frozen=True)
class RouteTarget:
    provider: str
    model: str
    source: str = "direct"
    alias: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def qualified_model(self) -> str:
        return f"{self.provider}/{self.model}" if self.provider else self.model


class _RouterRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._aliases: dict[str, list[str]] = {}
        self._round_robin: defaultdict[str, int] = defaultdict(int)
        self._target_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "successes": 0,
                "failures": 0,
                "latency_ms_total": 0.0,
                "latency_samples": 0,
                "cooldown_until": 0.0,
                "last_error": "",
            }
        )

    def register_alias(self, alias: str, targets: Sequence[str]) -> None:
        if not alias:
            raise CacheError("Alias name cannot be empty.")
        normalized_targets = [
            str(target).strip() for target in targets or [] if str(target).strip()
        ]
        if not normalized_targets:
            raise CacheError("Alias targets cannot be empty.")
        with self._lock:
            self._aliases[str(alias).strip()] = normalized_targets

    def clear_aliases(self) -> None:
        with self._lock:
            self._aliases.clear()
            self._round_robin.clear()

    def aliases(self) -> dict[str, list[str]]:
        with self._lock:
            return {key: list(value) for key, value in self._aliases.items()}

    def resolve_alias(self, alias: str) -> list[str]:
        with self._lock:
            return list(self._aliases.get(alias, []))

    def choose_round_robin(self, key: str, items: list[Any]) -> list[Any]:
        if len(items) <= 1:
            return items
        with self._lock:
            offset = self._round_robin[key] % len(items)
            self._round_robin[key] += 1
        return items[offset:] + items[:offset]

    def record_result(
        self,
        target: RouteTarget,
        *,
        success: bool,
        latency_ms: float,
        error: str = "",
        cooldown_seconds: float = 0.0,
    ) -> None:
        target_key = target.qualified_model
        with self._lock:
            bucket = self._target_stats[target_key]
            if success:
                bucket["successes"] += 1
                bucket["latency_ms_total"] += float(latency_ms or 0.0)
                bucket["latency_samples"] += 1
                bucket["cooldown_until"] = 0.0
                bucket["last_error"] = ""
            else:
                bucket["failures"] += 1
                bucket["last_error"] = str(error or "")
                if cooldown_seconds > 0:
                    bucket["cooldown_until"] = time.time() + float(cooldown_seconds)

    def stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            payload = {}
            for target_key, bucket in self._target_stats.items():
                success_rate = self._success_rate(bucket)
                payload[target_key] = {
                    "successes": int(bucket.get("successes", 0) or 0),
                    "failures": int(bucket.get("failures", 0) or 0),
                    "avg_latency_ms": round(self._avg_latency(bucket), 2),
                    "success_rate": round(success_rate, 4),
                    "health_score": round(self._health_score(bucket), 4),
                    "cooldown_until": float(bucket.get("cooldown_until", 0.0) or 0.0),
                    "last_error": str(bucket.get("last_error", "") or ""),
                }
        return payload

    def current_cooldown(self, target: RouteTarget) -> float:
        with self._lock:
            return float(
                self._target_stats[target.qualified_model].get("cooldown_until", 0.0) or 0.0
            )

    def avg_latency(self, target: RouteTarget) -> float:
        with self._lock:
            return self._avg_latency(self._target_stats[target.qualified_model])

    def health_score(self, target: RouteTarget) -> float:
        with self._lock:
            return self._health_score(self._target_stats[target.qualified_model])

    def clear_stats(self) -> None:
        with self._lock:
            self._target_stats.clear()

    @staticmethod
    def _avg_latency(bucket: dict[str, Any]) -> float:
        samples = int(bucket.get("latency_samples", 0) or 0)
        if samples <= 0:
            return 0.0
        return float(bucket.get("latency_ms_total", 0.0) or 0.0) / samples

    @staticmethod
    def _success_rate(bucket: dict[str, Any]) -> float:
        successes = int(bucket.get("successes", 0) or 0)
        failures = int(bucket.get("failures", 0) or 0)
        total = successes + failures
        if total <= 0:
            return 1.0
        return successes / total

    def _health_score(self, bucket: dict[str, Any]) -> float:
        success_rate = self._success_rate(bucket)
        avg_latency = self._avg_latency(bucket)
        cooldown_until = float(bucket.get("cooldown_until", 0.0) or 0.0)
        cooldown_penalty = 0.35 if cooldown_until > time.time() else 0.0
        latency_penalty = min(avg_latency / 5000.0, 1.0) * 0.25
        failure_penalty = (1.0 - success_rate) * 0.4
        health = 1.0 - cooldown_penalty - latency_penalty - failure_penalty
        return max(0.05, min(1.0, health))


_REGISTRY = _RouterRegistry()


def register_model_alias(alias: str, targets: Sequence[str]) -> None:
    _REGISTRY.register_alias(alias, targets)


def clear_model_aliases() -> None:
    _REGISTRY.clear_aliases()


def model_aliases() -> dict[str, list[str]]:
    return _REGISTRY.aliases()


def route_runtime_stats() -> dict[str, Any]:
    return {
        "aliases": model_aliases(),
        "targets": _REGISTRY.stats(),
    }


def clear_route_runtime_stats() -> None:
    _REGISTRY.clear_stats()
