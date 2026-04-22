
"""Token, device, and stop-sequence helpers for the H2O runtime."""

from __future__ import annotations

from typing import Any

from byte.h2o._runtime_common import _torch
from byte.h2o.policy import normalize_model_family
from byte.utils.error import CacheError


def _resolve_torch_dtype(torch_dtype: Any | None) -> Any | None:
    if torch_dtype in (None, "", "auto"):
        return None
    if not isinstance(torch_dtype, str):
        return torch_dtype
    torch = _torch()
    candidate = str(torch_dtype).replace("torch.", "").strip()
    if not hasattr(torch, candidate):
        raise CacheError(f"Unsupported torch dtype for Hugging Face runtime: {torch_dtype}")
    return getattr(torch, candidate)


def _resolve_model_device(model: Any, torch) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _infer_model_family(model: Any) -> str:
    config = getattr(model, "config", None)
    if config is None:
        return ""
    model_type = normalize_model_family(getattr(config, "model_type", ""))
    if model_type:
        return model_type
    architectures = getattr(config, "architectures", None) or []
    for architecture in architectures:
        normalized = normalize_model_family(str(architecture or ""))
        if normalized:
            return normalized
    return ""


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" or "text" in item:
                    parts.append(str(item.get("text", "") or ""))
            elif item not in (None, ""):
                parts.append(str(item))
        return "".join(parts)
    if content in (None, ""):
        return ""
    return str(content)


def _resolve_max_new_tokens(max_tokens: int | None, max_new_tokens: int | None) -> int:
    if max_new_tokens is not None and max_new_tokens != 0:
        return max(int(max_new_tokens), 1)
    if max_tokens is not None and max_tokens != 0:
        return max(int(max_tokens), 1)
    return 256


def _build_position_ids(seq_len: int, device: Any) -> Any:
    torch = _torch()
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)


def _select_next_token(
    logits: Any,
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool | None,
) -> Any:
    torch = _torch()
    should_sample = bool(do_sample) or (temperature is not None and temperature != 0 and float(temperature) > 0)
    if not should_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled_logits = logits / max(float(temperature or 1.0), 1e-5)
    if top_k is not None and top_k != 0:
        top_k = max(int(top_k), 1)
        values, _ = torch.topk(scaled_logits, k=min(top_k, scaled_logits.shape[-1]))
        threshold = values[:, -1].unsqueeze(-1)
        scaled_logits = torch.where(
            scaled_logits < threshold,
            torch.full_like(scaled_logits, torch.finfo(scaled_logits.dtype).min),
            scaled_logits,
        )
    if top_p is not None and top_p != 0 and float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > float(top_p)
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        filtered_logits = sorted_logits.masked_fill(
            sorted_mask, torch.finfo(sorted_logits.dtype).min
        )
        scaled_logits = torch.full_like(scaled_logits, torch.finfo(scaled_logits.dtype).min)
        scaled_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_logits)
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _is_terminal_token(token_ids: Any, tokenizer: Any) -> bool:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return False
    return bool((token_ids == eos_token_id).all())


def _stop_triggered(text: str, stop: list[str] | None) -> bool:
    if not stop:
        return False
    return any(str(token) and str(token) in text for token in stop)


def _trim_stop_sequences(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text
    earliest = None
    for token in stop:
        token = str(token or "")
        if not token:
            continue
        index = text.find(token)
        if index < 0:
            continue
        earliest = index if earliest is None else min(earliest, index)
    if earliest is None:
        return text
    return text[:earliest]

__all__ = [
    "_build_position_ids",
    "_infer_model_family",
    "_is_terminal_token",
    "_message_content_to_text",
    "_resolve_max_new_tokens",
    "_resolve_model_device",
    "_resolve_torch_dtype",
    "_select_next_token",
    "_stop_triggered",
    "_trim_stop_sequences",
]
