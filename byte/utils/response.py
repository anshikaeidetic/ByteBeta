import base64
from typing import Any

import requests


def get_message_from_openai_answer(openai_resp) -> Any:
    return openai_resp["choices"][0]["message"]["content"]


def get_stream_message_from_openai_answer(openai_data) -> Any:
    if not isinstance(openai_data, dict):
        return ""
    choices = openai_data.get("choices") or []
    if not choices:
        return ""
    delta = (choices[0] or {}).get("delta") or {}
    content = delta.get("content", "")
    return content if isinstance(content, str) else ""


def get_text_from_openai_answer(openai_resp) -> Any:
    return openai_resp["choices"][0]["text"]


def get_image_from_openai_b64(openai_resp) -> Any:
    return openai_resp["data"][0]["b64_json"]


def get_image_from_openai_url(openai_resp) -> Any:
    url = openai_resp["data"][0]["url"]
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img_content = response.content
    img_data = base64.b64encode(img_content)
    return img_data


def get_image_from_path(openai_resp) -> Any:
    img_path = openai_resp["data"][0]["url"]
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read())
    return img_data


def get_audio_text_from_openai_answer(openai_resp) -> Any:
    return openai_resp["text"]
