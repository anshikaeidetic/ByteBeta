
"""Generation loops for the Hugging Face H2O runtime."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from byte.h2o._runtime_common import (
    _RUNTIME_COUNTER,
    _STATS,
    RuntimeDescriptor,
    _build_response,
    _torch,
    describe_huggingface_runtime,
)
from byte.h2o._runtime_decode import H2ODecodeMixin
from byte.h2o._runtime_tokens import (
    _message_content_to_text,
    _resolve_max_new_tokens,
)


class H2OGenerationMixin(H2ODecodeMixin):
    def generate_chat(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        stop: list[str] | None = None,
        max_tokens: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        do_sample: bool | None = None,
        byte_h2o_enabled: bool = False,
        byte_h2o_heavy_ratio: float = 0.0,
        byte_h2o_recent_ratio: float = 0.0,
        byte_kv_codec: str = "disabled",
        byte_kv_bits: int = 8,
        byte_kv_hot_window_ratio: float = 0.25,
        byte_compression_mode: str = "shadow",
        compression_backend_policy: str = "auto",
        compression_verify_shadow_rate: float = 0.1,
        **kwargs,
    ) -> Any:
        prompt = self.render_messages(messages)
        return self.generate_text(
            model_name=model_name,
            prompt=prompt,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            response_mode="chat",
            byte_h2o_enabled=byte_h2o_enabled,
            byte_h2o_heavy_ratio=byte_h2o_heavy_ratio,
            byte_h2o_recent_ratio=byte_h2o_recent_ratio,
            byte_kv_codec=byte_kv_codec,
            byte_kv_bits=byte_kv_bits,
            byte_kv_hot_window_ratio=byte_kv_hot_window_ratio,
            byte_compression_mode=byte_compression_mode,
            compression_backend_policy=compression_backend_policy,
            compression_verify_shadow_rate=compression_verify_shadow_rate,
            **kwargs,
        )

    def generate_completion(
        self,
        *,
        model_name: str,
        prompt: str,
        stream: bool = False,
        stop: list[str] | None = None,
        max_tokens: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        do_sample: bool | None = None,
        byte_h2o_enabled: bool = False,
        byte_h2o_heavy_ratio: float = 0.0,
        byte_h2o_recent_ratio: float = 0.0,
        byte_kv_codec: str = "disabled",
        byte_kv_bits: int = 8,
        byte_kv_hot_window_ratio: float = 0.25,
        byte_compression_mode: str = "shadow",
        compression_backend_policy: str = "auto",
        compression_verify_shadow_rate: float = 0.1,
        **kwargs,
    ) -> Any:
        return self.generate_text(
            model_name=model_name,
            prompt=prompt,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            response_mode="completion",
            byte_h2o_enabled=byte_h2o_enabled,
            byte_h2o_heavy_ratio=byte_h2o_heavy_ratio,
            byte_h2o_recent_ratio=byte_h2o_recent_ratio,
            byte_kv_codec=byte_kv_codec,
            byte_kv_bits=byte_kv_bits,
            byte_kv_hot_window_ratio=byte_kv_hot_window_ratio,
            byte_compression_mode=byte_compression_mode,
            compression_backend_policy=compression_backend_policy,
            compression_verify_shadow_rate=compression_verify_shadow_rate,
            **kwargs,
        )

    def render_messages(self, messages: Iterable[dict[str, Any]]) -> str:
        normalized_messages = []
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            normalized_messages.append(
                {
                    "role": str(message.get("role", "user") or "user"),
                    "content": _message_content_to_text(message.get("content")),
                }
            )
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    normalized_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        rendered = []
        for message in normalized_messages:
            role = str(message["role"]).strip().lower() or "user"
            label = {
                "system": "System",
                "user": "User",
                "assistant": "Assistant",
                "tool": "Tool",
            }.get(role, role.title())
            rendered.append(f"{label}: {message['content']}")
        rendered.append("Assistant:")
        return "\n".join(rendered)

    def generate_text(
        self,
        *,
        model_name: str,
        prompt: str,
        stream: bool,
        stop: list[str] | None,
        max_tokens: int | None,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        do_sample: bool | None,
        response_mode: str,
        byte_h2o_enabled: bool,
        byte_h2o_heavy_ratio: float,
        byte_h2o_recent_ratio: float,
        byte_kv_codec: str,
        byte_kv_bits: int,
        byte_kv_hot_window_ratio: float,
        byte_compression_mode: str,
        compression_backend_policy: str,
        compression_verify_shadow_rate: float,
        **kwargs,
    ) -> Any:
        torch = _torch()
        with torch.inference_mode():
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            else:
                attention_mask = attention_mask.to(self.device)
            prompt_tokens = int(input_ids.shape[-1])
            descriptor = describe_huggingface_runtime(
                model_name=model_name,
                model_family=self.model_family,
                prompt_tokens=prompt_tokens,
                cache_hit=False,
                h2o_enabled=byte_h2o_enabled,
                h2o_heavy_ratio=byte_h2o_heavy_ratio,
                h2o_recent_ratio=byte_h2o_recent_ratio,
                kv_codec=byte_kv_codec,
                kv_bits=byte_kv_bits,
                kv_hot_window_ratio=byte_kv_hot_window_ratio,
                compression_mode=byte_compression_mode,
                compression_backend_policy=compression_backend_policy,
                compression_verify_shadow_rate=compression_verify_shadow_rate,
            )
            effective_new_tokens = _resolve_max_new_tokens(max_tokens, max_new_tokens)
            if stream:
                return self._stream_generation(
                    descriptor=descriptor,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    stop=stop,
                    max_new_tokens=effective_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    response_mode=response_mode,
                )
            return self._non_stream_generation(
                descriptor=descriptor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                stop=stop,
                max_new_tokens=effective_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                response_mode=response_mode,
            )

    def _non_stream_generation(
        self,
        *,
        descriptor: RuntimeDescriptor,
        input_ids: Any,
        attention_mask: Any,
        stop: list[str] | None,
        max_new_tokens: int,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        do_sample: bool | None,
        response_mode: str,
    ) -> dict[str, Any]:
        generated_text, usage, eviction_stats = self._decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stop=stop,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            descriptor=descriptor,
        )
        _STATS.record(descriptor, eviction_stats)
        return _build_response(
            response_mode=response_mode,
            model_name=descriptor.model_name,
            content=generated_text,
            usage=usage,
            descriptor=descriptor,
        )

    def _stream_generation(
        self,
        *,
        descriptor: RuntimeDescriptor,
        input_ids: Any,
        attention_mask: Any,
        stop: list[str] | None,
        max_new_tokens: int,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        do_sample: bool | None,
        response_mode: str,
    ) -> Any:
        stream_id = f"byte-hf-{next(_RUNTIME_COUNTER)}"
        created = int(time.time())
        if response_mode == "chat":
            yield {
                "id": stream_id,
                "created": created,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        content_parts = []
        usage, eviction_stats = None, None
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
            if not piece:
                continue
            content_parts.append(piece)
            if response_mode == "chat":
                yield {
                    "id": stream_id,
                    "created": created,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
            else:
                yield {
                    "id": stream_id,
                    "created": created,
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                }
        _STATS.record(descriptor, eviction_stats)
        final_usage = usage or {"prompt_tokens": int(input_ids.shape[-1]), "completion_tokens": 0}
        final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"]
        if response_mode == "chat":
            yield {
                "id": stream_id,
                "created": created,
                "object": "chat.completion.chunk",
                "byte_provider": "huggingface",
                "byte_runtime": descriptor.to_dict(),
                "usage": final_usage,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            return
        yield {
            "id": stream_id,
            "created": created,
            "object": "text_completion",
            "byte_provider": "huggingface",
            "byte_runtime": descriptor.to_dict(),
            "usage": final_usage,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }

__all__ = ["H2OGenerationMixin"]
