from typing import Any

from byte.adapter.router_runtime import route_completion


class ChatCompletion:
    """Provider-agnostic ByteAI Cache chat surface inspired by unified LLM gateways."""

    @classmethod
    def create(cls, *args, model: str, **kwargs) -> Any:
        if args:
            raise TypeError(
                "ByteAI Cache unified ChatCompletion.create only supports keyword arguments."
            )
        return route_completion(
            surface="chat_completion",
            model=model,
            async_mode=False,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, model: str, **kwargs) -> Any:
        if args:
            raise TypeError(
                "ByteAI Cache unified ChatCompletion.acreate only supports keyword arguments."
            )
        return await route_completion(
            surface="chat_completion",
            model=model,
            async_mode=True,
            **kwargs,
        )


class Completion:
    @classmethod
    def create(cls, *args, model: str, **kwargs) -> Any:
        if args:
            raise TypeError(
                "ByteAI Cache Completion.create only supports keyword arguments."
            )
        return route_completion(
            surface="text_completion",
            model=model,
            async_mode=False,
            **kwargs,
        )

    @classmethod
    async def acreate(cls, *args, model: str, **kwargs) -> Any:
        if args:
            raise TypeError(
                "ByteAI Cache Completion.acreate only supports keyword arguments."
            )
        return await route_completion(
            surface="text_completion",
            model=model,
            async_mode=True,
            **kwargs,
        )


class Image:
    @classmethod
    def create(cls, *args, model: str, **kwargs) -> Any:
        if args:
            raise TypeError("ByteAI Cache unified Image.create only supports keyword arguments.")
        return route_completion(
            surface="image",
            model=model,
            async_mode=False,
            **kwargs,
        )


class Audio:
    @classmethod
    def transcribe(cls, *args, model: str, file, **kwargs) -> Any:
        if args:
            raise TypeError(
                "ByteAI Cache unified Audio.transcribe only supports keyword arguments."
            )
        return route_completion(
            surface="audio_transcribe",
            model=model,
            async_mode=False,
            file=file,
            **kwargs,
        )

    @classmethod
    def translate(cls, *args, model: str, file, **kwargs) -> Any:
        if args:
            raise TypeError("ByteAI Cache unified Audio.translate only supports keyword arguments.")
        return route_completion(
            surface="audio_translate",
            model=model,
            async_mode=False,
            file=file,
            **kwargs,
        )


class Speech:
    @classmethod
    def create(cls, *args, model: str, input: str, voice: str, **kwargs) -> Any:
        if args:
            raise TypeError("ByteAI Cache unified Speech.create only supports keyword arguments.")
        return route_completion(
            surface="speech",
            model=model,
            async_mode=False,
            input=input,
            voice=voice,
            **kwargs,
        )


class Moderation:
    @classmethod
    def create(cls, *args, model: str, **kwargs) -> Any:
        if args:
            raise TypeError(
                "ByteAI Cache unified Moderation.create only supports keyword arguments."
            )
        return route_completion(
            surface="moderation",
            model=model,
            async_mode=False,
            **kwargs,
        )
