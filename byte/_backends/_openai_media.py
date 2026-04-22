"""Audio, image, speech, and moderation surfaces for the Byte OpenAI backend."""

from __future__ import annotations

import base64
import json
from typing import Any

from byte.adapter.adapter import adapt
from byte.adapter.base import BaseCacheLLM
from byte.manager.scalar_data.base import Answer, DataType
from byte.utils.error import wrap_error
from byte.utils.response import (
    get_audio_text_from_openai_answer,
)

from ._openai_response import (
    _construct_audio_text_from_cache,
    _construct_image_create_resp_from_cache,
    _openai_response_to_dict,
)
from ._openai_support import (
    _extract_audio_bytes,
    _extract_image_b64,
    _materialize_upload,
    _open_upload,
)


def _backend_module() -> Any:
    from byte._backends import openai as backend

    return backend


class Audio:
    """OpenAI Audio Wrapper (compatible with openai v1.x/v2.x)

    Example:
        .. code-block:: python

            from byte import cache
            from byte.processor.pre import get_file_bytes
            # init byte
            cache.init(pre_embedding_func=get_file_bytes)
            cache.set_openai_key()

            from byte.adapter import openai
            # run audio transcribe model with byte
            audio_file= open("/path/to/audio.mp3", "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

            # run audio translate model with byte
            audio_file= open("/path/to/audio.mp3", "rb")
            transcript = openai.Audio.translate("whisper-1", audio_file)
    """

    @classmethod
    def transcribe(cls, model: str, file: Any, *args, **kwargs) -> Any:
        import openai as _openai  # pylint: disable=C0415

        file_payload = _materialize_upload(file)

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            try:
                api_key = llm_kwargs.pop("api_key", None)
                api_base = llm_kwargs.pop("api_base", None)
                client = _backend_module()._get_client(api_key=api_key, api_base=api_base)
                m = llm_kwargs.pop("model", model)
                f = _open_upload(llm_kwargs.pop("file", file_payload))
                response = client.audio.transcriptions.create(model=m, file=f, **llm_kwargs)
                return _openai_response_to_dict(response)
            except _openai.OpenAIError as e:
                raise wrap_error(e) from e

        def cache_data_convert(cache_data) -> Any:
            return _construct_audio_text_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            update_cache_func(Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR))
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            model=model,
            *args,
            file=file_payload,
            **kwargs,
        )

    @classmethod
    def translate(cls, model: str, file: Any, *args, **kwargs) -> Any:
        import openai as _openai  # pylint: disable=C0415

        file_payload = _materialize_upload(file)

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            try:
                api_key = llm_kwargs.pop("api_key", None)
                api_base = llm_kwargs.pop("api_base", None)
                client = _backend_module()._get_client(api_key=api_key, api_base=api_base)
                m = llm_kwargs.pop("model", model)
                f = _open_upload(llm_kwargs.pop("file", file_payload))
                response = client.audio.translations.create(model=m, file=f, **llm_kwargs)
                return _openai_response_to_dict(response)
            except _openai.OpenAIError as e:
                raise wrap_error(e) from e

        def cache_data_convert(cache_data) -> Any:
            return _construct_audio_text_from_cache(cache_data)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            update_cache_func(Answer(get_audio_text_from_openai_answer(llm_data), DataType.STR))
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            model=model,
            *args,
            file=file_payload,
            **kwargs,
        )


class Speech:
    """OpenAI text-to-speech wrapper with cacheable audio bytes."""

    @classmethod
    def create(cls, model: str, input: str, voice: str, *args, **kwargs) -> Any:
        response_format = kwargs.pop("response_format", "mp3")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            import openai as _openai  # pylint: disable=C0415

            try:
                api_key = llm_kwargs.pop("api_key", None)
                api_base = llm_kwargs.pop("api_base", None)
                client = _backend_module()._get_client(api_key=api_key, api_base=api_base)
                m = llm_kwargs.pop("model", model)
                i = llm_kwargs.pop("input", input)
                v = llm_kwargs.pop("voice", voice)
                fmt = llm_kwargs.pop("response_format", response_format)
                response = client.audio.speech.create(
                    model=m,
                    input=i,
                    voice=v,
                    response_format=fmt,
                    **llm_kwargs,
                )
                return _extract_audio_bytes(response)
            except _openai.OpenAIError as e:
                raise wrap_error(e) from e

        def cache_data_convert(cache_data) -> dict[str, Any]:
            return {
                "audio": base64.b64decode(cache_data),
                "format": response_format,
                "byte": True,
            }

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> dict[str, Any]:  # pylint: disable=unused-argument
            update_cache_func(Answer(base64.b64encode(llm_data).decode("ascii"), DataType.STR))
            return {
                "audio": llm_data,
                "format": response_format,
            }

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            model=model,
            input=input,
            voice=voice,
            *args,
            response_format=response_format,
            **kwargs,
        )


class Image:
    """OpenAI Image Wrapper (compatible with openai v1.x/v2.x)

    Example:
        .. code-block:: python

            from byte import cache
            from byte.processor.pre import get_prompt
            # init byte
            cache.init(pre_embedding_func=get_prompt)
            cache.set_openai_key()

            from byte.adapter import openai
            # run image generation model with byte
            response = openai.Image.create(
              prompt="a white siamese cat",
              n=1,
              size="256x256"
            )
            response_url = response['data'][0]['url']
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        import openai as _openai  # pylint: disable=C0415

        try:
            api_key = llm_kwargs.pop("api_key", None)
            api_base = llm_kwargs.pop("api_base", None)
            llm_kwargs.pop("response_format", None)
            client = _backend_module()._get_client(api_key=api_key, api_base=api_base)
            response = client.images.generate(**llm_kwargs)
            return _openai_response_to_dict(response)
        except _openai.OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        response_format = kwargs.pop("response_format", "url")
        size = kwargs.pop("size", "256x256")

        def cache_data_convert(cache_data) -> Any:
            return _construct_image_create_resp_from_cache(
                image_data=cache_data, response_format=response_format, size=size
            )

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            img_b64 = _extract_image_b64(llm_data)
            update_cache_func(Answer(img_b64, DataType.IMAGE_BASE64))
            return llm_data

        return adapt(
            cls._llm_handler,
            cache_data_convert,
            update_cache_callback,
            *args,
            response_format=response_format,
            size=size,
            **kwargs,
        )


class Moderation(BaseCacheLLM):
    """OpenAI Moderation Wrapper (compatible with openai v1.x/v2.x)

    Example:
        .. code-block:: python

            from byte.adapter import openai
            from byte.adapter.api import init_similar_cache
            from byte.processor.pre import get_openai_moderation_input

            init_similar_cache(pre_func=get_openai_moderation_input)
            openai.Moderation.create(
                input="I want to kill them.",
            )
    """

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs) -> Any:
        import openai as _openai  # pylint: disable=C0415

        try:
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)

            api_key = llm_kwargs.pop("api_key", None)
            api_base = llm_kwargs.pop("api_base", None)
            client = _backend_module()._get_client(api_key=api_key, api_base=api_base)
            response = client.moderations.create(**llm_kwargs)
            return _openai_response_to_dict(response)
        except _openai.OpenAIError as e:
            raise wrap_error(e) from e

    @classmethod
    def _cache_data_convert(cls, cache_data) -> Any:
        payload = json.loads(cache_data)
        payload["byte"] = True
        return payload

    @classmethod
    def _update_cache_callback(cls, llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
        update_cache_func(Answer(json.dumps(llm_data, indent=4), DataType.STR))
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        kwargs = cls.fill_base_args(**kwargs)
        res = adapt(
            cls._llm_handler,
            cls._cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )

        input_request_param = kwargs.get("input")
        expect_res_len = 1
        if isinstance(input_request_param, list):
            expect_res_len = len(input_request_param)
        if len(res.get("results")) != expect_res_len:
            kwargs["cache_skip"] = True
            res = adapt(
                cls._llm_handler,
                cls._cache_data_convert,
                cls._update_cache_callback,
                *args,
                **kwargs,
            )
        return res
