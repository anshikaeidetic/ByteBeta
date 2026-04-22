
"""Audio and speech resources for the Gemini backend."""

from __future__ import annotations

import base64
import json
from typing import Any

from byte.adapter.adapter import adapt
from byte.manager.scalar_data.base import Answer, DataType
from byte.utils.multimodal import materialize_upload

from .gemini_clients import (
    _build_namespace,
    _get_genai_types,
    _resolve_backend_callable,
)
from .gemini_clients import (
    _get_client as _default_get_client,
)
from .gemini_messages import _build_content_config, _build_genai_part
from .gemini_responses import (
    _construct_audio_text_from_cache,
    _construct_speech_from_cache,
    _extract_gemini_audio_bytes,
)


class Audio:
    """Gemini audio understanding wrapper for transcription-style tasks."""

    llm = None

    @classmethod
    def _run_audio_prompt(cls, *, instruction: str, model: str, file: Any, **kwargs) -> Any:
        if cls.llm is not None:
            return cls.llm(model=model, file=file, instruction=instruction, **kwargs)

        api_key = kwargs.pop("api_key", None)
        get_client = _resolve_backend_callable("_get_client", _default_get_client)
        client = get_client(api_key=api_key)
        upload = materialize_upload(file, default_name="audio.bin")

        response = client.models.generate_content(
            model=kwargs.pop("model", model),
            contents=[
                _build_genai_part(text=instruction),
                _build_genai_part(
                    data=upload["bytes"], mime_type=upload.get("mime_type") or "audio/mpeg"
                ),
            ],
            config=_build_content_config(**kwargs),
        )
        return {
            "byte_provider": "gemini",
            "text": getattr(response, "text", "") or "",
        }

    @classmethod
    def transcribe(cls, model: str, file: Any, *args, **kwargs) -> Any:
        upload = materialize_upload(file, default_name="audio.bin")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            target_model = llm_kwargs.pop("model", model)
            file_payload = llm_kwargs.pop("file", upload)
            return cls._run_audio_prompt(
                instruction="Transcribe this audio verbatim. Return only the transcript.",
                model=target_model,
                file=file_payload,
                **llm_kwargs,
            )

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            update_cache_func(Answer(llm_data.get("text", ""), DataType.STR))
            return llm_data

        return adapt(
            llm_handler,
            _construct_audio_text_from_cache,
            update_cache_callback,
            *args,
            model=model,
            file=upload,
            **kwargs,
        )

    @classmethod
    def translate(cls, model: str, file: Any, *args, **kwargs) -> Any:
        upload = materialize_upload(file, default_name="audio.bin")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            target_model = llm_kwargs.pop("model", model)
            file_payload = llm_kwargs.pop("file", upload)
            return cls._run_audio_prompt(
                instruction="Translate this audio into English. Return only the translation.",
                model=target_model,
                file=file_payload,
                **llm_kwargs,
            )

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            update_cache_func(Answer(llm_data.get("text", ""), DataType.STR))
            return llm_data

        return adapt(
            llm_handler,
            _construct_audio_text_from_cache,
            update_cache_callback,
            *args,
            model=model,
            file=upload,
            **kwargs,
        )


class Speech:
    """Gemini text-to-speech wrapper with cacheable audio bytes."""

    llm = None

    @classmethod
    def create(cls, model: str, input: str, voice: str, *args, **kwargs) -> Any:
        response_format = kwargs.pop("response_format", "wav")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            target_model = llm_kwargs.pop("model", model)
            input_text = llm_kwargs.pop("input", input)
            voice_name = llm_kwargs.pop("voice", voice)
            fmt = llm_kwargs.pop("response_format", response_format)
            if cls.llm is not None:
                return cls.llm(
                    model=target_model,
                    input=input_text,
                    voice=voice_name,
                    response_format=fmt,
                    **llm_kwargs,
                )

            api_key = llm_kwargs.pop("api_key", None)
            get_client = _resolve_backend_callable("_get_client", _default_get_client)
            client = get_client(api_key=api_key)
            types = _get_genai_types()
            if types is not None:
                speech_config = types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                )
            else:
                speech_config = _build_namespace(
                    voice_config=_build_namespace(
                        prebuilt_voice_config=_build_namespace(voice_name=voice_name)
                    )
                )
            response = client.models.generate_content(
                model=target_model,
                contents=input_text,
                config=_build_content_config(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config,
                    **llm_kwargs,
                ),
            )
            audio_bytes, mime_type = _extract_gemini_audio_bytes(response)
            return {
                "audio": audio_bytes,
                "format": fmt,
                "mime_type": mime_type,
                "byte_provider": "gemini",
            }

        def cache_data_convert(cache_data) -> Any:
            return _construct_speech_from_cache(cache_data, response_format)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            update_cache_func(
                Answer(
                    json.dumps(
                        {
                            "audio": base64.b64encode(llm_data["audio"]).decode("ascii"),
                            "format": llm_data.get("format", response_format),
                            "mime_type": llm_data.get("mime_type"),
                        }
                    ),
                    DataType.STR,
                )
            )
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            *args,
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            **kwargs,
        )

__all__ = ["Audio", "Speech"]
