
"""Gemini request conversion helpers kept provider-import lazy."""

from __future__ import annotations

from typing import Any

from byte.utils.multimodal import extract_content_parts, extract_text_content

from .gemini_clients import _build_namespace, _get_genai_types


def _build_genai_part(*, text=None, data=None, mime_type=None, file_uri=None) -> Any:
    types = _get_genai_types()
    if types is not None:
        if text is not None:
            return types.Part.from_text(text=text)
        if data is not None:
            return types.Part.from_bytes(data=data, mime_type=mime_type or "application/octet-stream")
        if file_uri is not None:
            return types.Part.from_uri(file_uri=file_uri, mime_type=mime_type or None)
    if text is not None:
        return {"text": text}
    if data is not None:
        return {"inline_data": {"data": data, "mime_type": mime_type or "application/octet-stream"}}
    if file_uri is not None:
        return {"file_data": {"file_uri": file_uri, "mime_type": mime_type or None}}
    return {"text": ""}

def _build_content_config(system_instruction=None, **llm_kwargs) -> Any | None:
    config_kwargs = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if "max_tokens" in llm_kwargs:
        config_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")
    if "temperature" in llm_kwargs:
        config_kwargs["temperature"] = llm_kwargs.pop("temperature")
    if "top_p" in llm_kwargs:
        config_kwargs["top_p"] = llm_kwargs.pop("top_p")
    if "top_k" in llm_kwargs:
        config_kwargs["top_k"] = llm_kwargs.pop("top_k")
    if "n" in llm_kwargs:
        config_kwargs["candidate_count"] = llm_kwargs.pop("n")
    if "response_format" in llm_kwargs:
        config_kwargs["response_mime_type"] = llm_kwargs.pop("response_format")
    if "response_modalities" in llm_kwargs:
        config_kwargs["response_modalities"] = llm_kwargs.pop("response_modalities")
    if "speech_config" in llm_kwargs:
        config_kwargs["speech_config"] = llm_kwargs.pop("speech_config")
    if not config_kwargs:
        return None
    types = _get_genai_types()
    if types is not None:
        return types.GenerateContentConfig(**config_kwargs)
    return _build_namespace(**config_kwargs)


def _convert_messages(messages) -> tuple[Any, ...]:
    contents = []
    system_chunks = []

    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            system_text = extract_text_content(msg.get("content"))
            if system_text:
                system_chunks.append(system_text)
            continue

        parts = []
        for part in extract_content_parts(msg.get("content", "")):
            part_type = part.get("type")
            if part_type == "text":
                text = str(part.get("text") or "")
                if text:
                    parts.append(_build_genai_part(text=text))
                continue

            mime_type = str(part.get("mime_type") or "application/octet-stream")
            if part.get("bytes") is not None:
                parts.append(_build_genai_part(data=part["bytes"], mime_type=mime_type))
            elif part.get("uri"):
                parts.append(_build_genai_part(file_uri=str(part["uri"]), mime_type=mime_type))

        if not parts:
            parts.append(_build_genai_part(text=""))

        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            }
        )

    system_instruction = "\n\n".join(chunk for chunk in system_chunks if chunk).strip() or None
    return contents, system_instruction

__all__ = ["_build_content_config", "_build_genai_part", "_convert_messages"]
