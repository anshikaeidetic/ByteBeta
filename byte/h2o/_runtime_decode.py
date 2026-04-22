
"""Decode loops for H2O runtime generation."""

from __future__ import annotations

from typing import Any

from byte.h2o._runtime_common import RuntimeDescriptor, _torch
from byte.h2o._runtime_kv import _KVCompressionController
from byte.h2o._runtime_tokens import (
    _build_position_ids,
    _is_terminal_token,
    _select_next_token,
    _stop_triggered,
    _trim_stop_sequences,
)
from byte.h2o.policy import H2OSequenceCache, H2OSettings


class H2ODecodeMixin:
    def _decode(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        stop: list[str] | None,
        max_new_tokens: int,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        do_sample: bool | None,
        descriptor: RuntimeDescriptor,
    ) -> tuple[str, dict[str, int], dict[str, Any]]:
        content_parts = []
        usage = None
        eviction_stats = None
        for piece, usage, eviction_stats in self._decode_stream(  # noqa: B007
            input_ids=input_ids,
            attention_mask=attention_mask,
            stop=stop,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            descriptor=descriptor,
        ):
            if piece:
                content_parts.append(piece)
        final_usage = usage or {"prompt_tokens": int(input_ids.shape[-1]), "completion_tokens": 0}
        final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"]
        generated_text = "".join(content_parts)
        if _stop_triggered(generated_text, stop):
            generated_text = _trim_stop_sequences(generated_text, stop)
        return generated_text, final_usage, eviction_stats or {}

    def _decode_stream(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        stop: list[str] | None,
        max_new_tokens: int,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        do_sample: bool | None,
        descriptor: RuntimeDescriptor,
    ) -> Any:
        torch = _torch()
        controller = _KVCompressionController(descriptor, device=self.device, torch_module=torch)
        cache = H2OSequenceCache(
            H2OSettings(
                enabled=descriptor.applied_h2o,
                requested=descriptor.requested_h2o,
                applied=descriptor.applied_h2o,
                model_family=descriptor.model_family,
                prompt_tokens=descriptor.prompt_tokens,
                heavy_ratio=descriptor.heavy_ratio,
                recent_ratio=descriptor.recent_ratio,
                heavy_budget=descriptor.heavy_budget,
                recent_budget=descriptor.recent_budget,
                cache_budget=descriptor.cache_budget,
                fallback_reason=descriptor.fallback_reason,
            )
        )
        generation_text = ""
        prompt_tokens = int(input_ids.shape[-1])
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=_build_position_ids(input_ids.shape[-1], self.device),
            use_cache=True,
            output_attentions=descriptor.applied_h2o,
            return_dict=True,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        eviction_stats = {
            "retained_tokens": 0,
            "original_tokens": 0,
            "evicted_tokens": 0,
            "retained_fraction": 1.0,
        }
        if descriptor.applied_h2o and past_key_values is not None:
            past_key_values, eviction_stats = cache.apply(
                past_key_values,
                getattr(outputs, "attentions", None),
            )
        controller.capture(past_key_values)
        generated_tokens = 0
        next_position = prompt_tokens
        next_token = _select_next_token(
            outputs.logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        )
        while generated_tokens < max_new_tokens:
            if _is_terminal_token(next_token, self.tokenizer):
                break
            token_text = self.tokenizer.decode(
                next_token[0].tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generation_text += token_text
            generated_tokens += 1
            yield token_text, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_tokens,
            }, eviction_stats
            if _stop_triggered(generation_text, stop):
                generation_text = _trim_stop_sequences(generation_text, stop)
                break
            materialized_past_key_values = controller.materialize()
            outputs = self.model(
                input_ids=next_token,
                attention_mask=torch.ones(
                    (input_ids.shape[0], controller.seq_len() + 1),
                    dtype=attention_mask.dtype,
                    device=self.device,
                ),
                position_ids=torch.full(
                    (input_ids.shape[0], 1),
                    fill_value=next_position,
                    dtype=torch.long,
                    device=self.device,
                ),
                past_key_values=materialized_past_key_values,
                use_cache=True,
                output_attentions=descriptor.applied_h2o,
                return_dict=True,
            )
            next_position += 1
            past_key_values = getattr(outputs, "past_key_values", None)
            if descriptor.applied_h2o and past_key_values is not None:
                past_key_values, eviction_stats = cache.apply(
                    past_key_values,
                    getattr(outputs, "attentions", None),
                )
            controller.capture(past_key_values)
            next_token = _select_next_token(
                outputs.logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
        if _stop_triggered(generation_text, stop):
            generation_text = _trim_stop_sequences(generation_text, stop)

__all__ = ["H2ODecodeMixin"]
