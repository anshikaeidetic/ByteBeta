
"""Image generation resource for the Gemini backend."""

from __future__ import annotations

import base64
import json
from typing import Any

from byte.adapter.adapter import adapt
from byte.manager.scalar_data.base import Answer, DataType

from .gemini_clients import (
    _build_namespace,
    _get_genai_types,
    _resolve_backend_callable,
)
from .gemini_clients import (
    _get_client as _default_get_client,
)
from .gemini_responses import _construct_image_from_cache, _gemini_image_response_to_openai


class Image:
    """Gemini image-generation wrapper with cacheable image payloads."""

    llm = None

    @classmethod
    def create(cls, *args, **kwargs) -> Any:
        response_format = kwargs.pop("response_format", "b64_json")
        model = kwargs.get("model", "imagen-3.0-generate-002")
        prompt = kwargs.get("prompt", "")

        def llm_handler(*llm_args, **llm_kwargs) -> Any:
            if cls.llm is not None:
                return cls.llm(*llm_args, **llm_kwargs)

            api_key = llm_kwargs.pop("api_key", None)
            get_client = _resolve_backend_callable("_get_client", _default_get_client)
            client = get_client(api_key=api_key)
            prompt_text = llm_kwargs.pop("prompt", prompt)
            image_model = llm_kwargs.pop("model", model)
            n = llm_kwargs.pop("n", 1)
            size = llm_kwargs.pop("size", None)

            config_kwargs = {"number_of_images": n}
            if size:
                config_kwargs["image_size"] = size
            if "negative_prompt" in llm_kwargs:
                config_kwargs["negative_prompt"] = llm_kwargs.pop("negative_prompt")
            if "seed" in llm_kwargs:
                config_kwargs["seed"] = llm_kwargs.pop("seed")
            if "guidance_scale" in llm_kwargs:
                config_kwargs["guidance_scale"] = llm_kwargs.pop("guidance_scale")

            types = _get_genai_types()
            config = (
                types.GenerateImagesConfig(**config_kwargs)
                if types is not None
                else _build_namespace(**config_kwargs)
            )
            response = client.models.generate_images(
                model=image_model,
                prompt=prompt_text,
                config=config,
            )
            return _gemini_image_response_to_openai(response, response_format)

        def cache_data_convert(cache_data) -> Any:
            return _construct_image_from_cache(cache_data, response_format)

        def update_cache_callback(llm_data, update_cache_func, *args, **kwargs) -> Any:  # pylint: disable=unused-argument
            images = []
            for item in llm_data.get("data", []):
                b64_payload = item.get("b64_json")
                if not b64_payload and item.get("url"):
                    with open(item["url"], "rb") as file_obj:
                        b64_payload = base64.b64encode(file_obj.read()).decode("ascii")
                if b64_payload:
                    images.append({"b64": b64_payload, "mime_type": "image/png"})
            update_cache_func(Answer(json.dumps({"images": images}), DataType.STR))
            return llm_data

        return adapt(
            llm_handler,
            cache_data_convert,
            update_cache_callback,
            *args,
            **kwargs,
        )

__all__ = ["Image"]
