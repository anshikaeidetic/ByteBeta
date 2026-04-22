from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_H2O_MODEL_FAMILIES = {
    "gemma",
    "gpt_neox",
    "llama",
    "mistral",
    "opt",
    "phi3",
    "qwen2",
}

_MODEL_FAMILY_ALIASES = {
    "gemma2": "gemma",
    "gpt-neox": "gpt_neox",
    "gptneox": "gpt_neox",
    "llama2": "llama",
    "llama3": "llama",
    "llama4": "llama",
    "phi-3": "phi3",
    "phi_3": "phi3",
    "qwen": "qwen2",
}


def _torch() -> Any:
    from byte.utils import import_torch

    import_torch()
    import torch

    return torch


@dataclass(frozen=True)
class H2OSettings:
    enabled: bool
    requested: bool
    applied: bool
    model_family: str
    prompt_tokens: int
    heavy_ratio: float
    recent_ratio: float
    heavy_budget: int
    recent_budget: int
    cache_budget: int
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "requested": self.requested,
            "applied": self.applied,
            "model_family": self.model_family,
            "prompt_tokens": self.prompt_tokens,
            "heavy_ratio": self.heavy_ratio,
            "recent_ratio": self.recent_ratio,
            "heavy_budget": self.heavy_budget,
            "recent_budget": self.recent_budget,
            "cache_budget": self.cache_budget,
            "fallback_reason": self.fallback_reason,
        }


def normalize_model_family(model_family: str) -> str:
    raw = str(model_family or "").strip().lower().replace("-", "_")
    if not raw:
        return ""
    return _MODEL_FAMILY_ALIASES.get(raw, raw)


def resolve_h2o_settings(
    *,
    enabled: bool,
    prompt_tokens: int,
    model_family: str,
    heavy_ratio: float,
    recent_ratio: float,
) -> H2OSettings:
    prompt_tokens = max(int(prompt_tokens or 0), 0)
    heavy_ratio = max(float(heavy_ratio or 0.0), 0.0)
    recent_ratio = max(float(recent_ratio or 0.0), 0.0)
    normalized_family = normalize_model_family(model_family)
    requested = bool(enabled)
    if not requested:
        return H2OSettings(
            enabled=False,
            requested=False,
            applied=False,
            model_family=normalized_family,
            prompt_tokens=prompt_tokens,
            heavy_ratio=heavy_ratio,
            recent_ratio=recent_ratio,
            heavy_budget=0,
            recent_budget=0,
            cache_budget=0,
            fallback_reason="disabled",
        )
    if normalized_family not in SUPPORTED_H2O_MODEL_FAMILIES:
        return H2OSettings(
            enabled=False,
            requested=True,
            applied=False,
            model_family=normalized_family,
            prompt_tokens=prompt_tokens,
            heavy_ratio=heavy_ratio,
            recent_ratio=recent_ratio,
            heavy_budget=0,
            recent_budget=0,
            cache_budget=0,
            fallback_reason="unsupported_model_family",
        )
    if prompt_tokens <= 0:
        return H2OSettings(
            enabled=False,
            requested=True,
            applied=False,
            model_family=normalized_family,
            prompt_tokens=prompt_tokens,
            heavy_ratio=heavy_ratio,
            recent_ratio=recent_ratio,
            heavy_budget=0,
            recent_budget=0,
            cache_budget=0,
            fallback_reason="empty_prompt",
        )

    heavy_budget = _resolve_budget(prompt_tokens, heavy_ratio)
    recent_budget = _resolve_budget(prompt_tokens, recent_ratio)
    if recent_budget > prompt_tokens:
        recent_budget = prompt_tokens
    if heavy_budget + recent_budget > prompt_tokens:
        heavy_budget = max(prompt_tokens - recent_budget, 0)
    cache_budget = heavy_budget + recent_budget
    if cache_budget <= 0:
        return H2OSettings(
            enabled=False,
            requested=True,
            applied=False,
            model_family=normalized_family,
            prompt_tokens=prompt_tokens,
            heavy_ratio=heavy_ratio,
            recent_ratio=recent_ratio,
            heavy_budget=0,
            recent_budget=0,
            cache_budget=0,
            fallback_reason="zero_cache_budget",
        )
    return H2OSettings(
        enabled=True,
        requested=True,
        applied=True,
        model_family=normalized_family,
        prompt_tokens=prompt_tokens,
        heavy_ratio=heavy_ratio,
        recent_ratio=recent_ratio,
        heavy_budget=heavy_budget,
        recent_budget=recent_budget,
        cache_budget=cache_budget,
        fallback_reason="",
    )


def _resolve_budget(prompt_tokens: int, ratio: float) -> int:
    if prompt_tokens <= 0 or ratio <= 0:
        return 0
    return min(prompt_tokens, max(1, int(prompt_tokens * ratio)))


class H2OLayerState:
    def __init__(self, heavy_budget: int, recent_budget: int) -> None:
        self.heavy_budget = max(int(heavy_budget or 0), 0)
        self.recent_budget = max(int(recent_budget or 0), 0)
        self.cache_budget = self.heavy_budget + self.recent_budget
        self.hh_score = None
        self.evictions = 0

    def reset(self) -> None:
        self.hh_score = None
        self.evictions = 0

    def update(self, attention: Any, kv_heads: int) -> None:
        if attention is None:
            return
        torch = _torch()
        scores = attention.detach().sum(dim=0).sum(dim=1)
        if scores.ndim != 2:
            return
        if kv_heads > 0 and scores.shape[0] != kv_heads:
            if scores.shape[0] % kv_heads == 0:
                scores = scores.view(kv_heads, scores.shape[0] // kv_heads, scores.shape[-1]).sum(
                    dim=1
                )
            else:
                scores = scores[:kv_heads]
        num_new_tokens = attention.shape[2]
        if self.hh_score is None:
            self.hh_score = scores
            return
        if scores.shape[-1] > num_new_tokens:
            scores[:, :-num_new_tokens] += self.hh_score[:, : scores.shape[-1] - num_new_tokens]
        self.hh_score = scores
        if self.hh_score.dtype != torch.float32:
            self.hh_score = self.hh_score.to(dtype=torch.float32)

    def evict(self, past_key_value: Any) -> tuple[Any, dict[str, Any]]:
        key_states, value_states = past_key_value[:2]
        seq_len = int(key_states.shape[-2])
        if self.hh_score is not None and self.hh_score.shape[-1] != seq_len:
            self.hh_score = self.hh_score[:, :seq_len]
        if self.cache_budget <= 0 or seq_len <= self.cache_budget or self.hh_score is None:
            return past_key_value, {
                "retained_tokens": seq_len,
                "original_tokens": seq_len,
                "evicted_tokens": 0,
            }

        torch = _torch()
        num_heads = int(key_states.shape[1])
        prefix_len = max(seq_len - self.recent_budget, 0)
        heavy_budget = min(self.heavy_budget, prefix_len)
        recent_budget = min(self.recent_budget, seq_len)
        if heavy_budget <= 0 and recent_budget >= seq_len:
            return past_key_value, {
                "retained_tokens": seq_len,
                "original_tokens": seq_len,
                "evicted_tokens": 0,
            }

        if heavy_budget > 0:
            global_scores = self.hh_score[:, :prefix_len].sum(dim=0)
            shared_topk = torch.topk(global_scores, k=heavy_budget, dim=-1).indices
            keep_topk = shared_topk.unsqueeze(0).expand(num_heads, heavy_budget)
        else:
            keep_topk = torch.empty((num_heads, 0), dtype=torch.long, device=key_states.device)
        if recent_budget > 0:
            keep_recent = (
                torch.arange(seq_len - recent_budget, seq_len, device=key_states.device)
                .unsqueeze(0)
                .expand(num_heads, recent_budget)
            )
        else:
            keep_recent = torch.empty((num_heads, 0), dtype=torch.long, device=key_states.device)

        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)
        keep_idx = torch.sort(keep_idx, dim=-1).values
        gather_idx = keep_idx.unsqueeze(0).unsqueeze(-1).expand(
            key_states.shape[0],
            key_states.shape[1],
            keep_idx.shape[-1],
            key_states.shape[-1],
        )
        kept_key_states = torch.gather(key_states, dim=2, index=gather_idx)
        kept_value_states = torch.gather(value_states, dim=2, index=gather_idx)
        self.hh_score = torch.gather(self.hh_score, dim=1, index=keep_idx)

        evicted_tokens = max(seq_len - keep_idx.shape[-1], 0)
        self.evictions += evicted_tokens
        updated_past = (kept_key_states, kept_value_states, *past_key_value[2:])
        return updated_past, {
            "retained_tokens": int(keep_idx.shape[-1]),
            "original_tokens": seq_len,
            "evicted_tokens": evicted_tokens,
        }


class H2OSequenceCache:
    def __init__(self, settings: H2OSettings) -> None:
        self.settings = settings
        self._layers: list[H2OLayerState] = []

    def reset(self) -> None:
        self._layers = []

    def apply(self, past_key_values: Any, attentions: Any) -> tuple[Any, dict[str, Any]]:
        if not self.settings.applied or not past_key_values:
            return past_key_values, {
                "retained_tokens": 0,
                "original_tokens": 0,
                "evicted_tokens": 0,
                "retained_fraction": 1.0,
            }

        self._ensure_layers(len(past_key_values))
        kept_layers = []
        retained_tokens = 0
        original_tokens = 0
        evicted_tokens = 0
        for index, layer_past in enumerate(past_key_values):
            key_states = layer_past[0]
            kv_heads = int(key_states.shape[1])
            attention = None
            if attentions and index < len(attentions):
                attention = attentions[index]
            self._layers[index].update(attention, kv_heads)
            updated_layer, stats = self._layers[index].evict(layer_past)
            kept_layers.append(updated_layer)
            retained_tokens += int(stats["retained_tokens"])
            original_tokens += int(stats["original_tokens"])
            evicted_tokens += int(stats["evicted_tokens"])
        retained_fraction = (
            float(retained_tokens) / float(original_tokens) if original_tokens > 0 else 1.0
        )
        return tuple(kept_layers), {
            "retained_tokens": retained_tokens,
            "original_tokens": original_tokens,
            "evicted_tokens": evicted_tokens,
            "retained_fraction": retained_fraction,
        }

    def _ensure_layers(self, layer_count: int) -> None:
        while len(self._layers) < layer_count:
            self._layers.append(
                H2OLayerState(
                    heavy_budget=self.settings.heavy_budget,
                    recent_budget=self.settings.recent_budget,
                )
            )
