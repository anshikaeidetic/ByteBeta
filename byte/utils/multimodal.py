import base64
import hashlib
import mimetypes
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

_DATA_URL_PATTERN = re.compile(
    r"^data:(?P<mime>[-\w.+/]+);base64,(?P<data>[a-z0-9+/=\s]+)$",
    re.IGNORECASE,
)


class NamedBytesIO(BytesIO):
    """BytesIO wrapper that preserves the original upload name."""

    def __init__(self, payload: bytes, name: str) -> None:
        super().__init__(payload)
        self.name = name

    def peek(self, size: int = -1) -> bytes:
        position = self.tell()
        data = self.read() if size is None or size < 0 else self.read(size)
        self.seek(position)
        return data


def _coerce_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            return value.encode("utf-8")
    return bytes(value)


def guess_mime_type(
    *,
    name: str | None = None,
    mime_type: str | None = None,
    payload: bytes | None = None,
    fallback: str = "application/octet-stream",
) -> str:
    if mime_type:
        return str(mime_type)

    guessed = mimetypes.guess_type(str(name or ""))[0]
    if guessed:
        return guessed

    head = payload[:16] if payload else b""
    if head.startswith(b"%PDF"):
        return "application/pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head.startswith(b"RIFF") and payload and payload[8:12] == b"WAVE":
        return "audio/wav"
    if head.startswith(b"ID3"):
        return "audio/mpeg"
    if payload and payload.startswith(b"OggS"):
        return "audio/ogg"

    return fallback


def parse_data_url(value: str) -> dict[str, Any] | None:
    match = _DATA_URL_PATTERN.match(str(value or "").strip())
    if not match:
        return None
    return {
        "mime_type": match.group("mime"),
        "bytes": base64.b64decode(match.group("data")),
    }


def bytes_to_data_url(payload: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def materialize_upload(
    file_obj: Any,
    *,
    default_name: str = "upload.bin",
    mime_type: str | None = None,
) -> dict[str, Any]:
    if isinstance(file_obj, dict):
        if file_obj.get("path"):
            file_obj = file_obj["path"]
        elif (
            file_obj.get("bytes") is not None
            or file_obj.get("data") is not None
            or file_obj.get("file_data") is not None
        ):
            payload = file_obj.get("bytes")
            if payload is None:
                payload = file_obj.get("data")
            if payload is None:
                payload = file_obj.get("file_data")
            payload = _coerce_bytes(payload)
            name = str(file_obj.get("name") or file_obj.get("filename") or default_name)
            return {
                "name": name,
                "bytes": payload,
                "mime_type": guess_mime_type(
                    name=name,
                    mime_type=file_obj.get("mime_type") or mime_type,
                    payload=payload,
                ),
            }

    if isinstance(file_obj, (str, os.PathLike, Path)) and os.path.exists(os.fspath(file_obj)):
        path = Path(file_obj)
        payload = path.read_bytes()
        return {
            "name": path.name,
            "bytes": payload,
            "mime_type": guess_mime_type(name=path.name, mime_type=mime_type, payload=payload),
        }

    if isinstance(file_obj, (bytes, bytearray, memoryview)):
        payload = _coerce_bytes(file_obj)
        return {
            "name": default_name,
            "bytes": payload,
            "mime_type": guess_mime_type(name=default_name, mime_type=mime_type, payload=payload),
        }

    name = str(getattr(file_obj, "name", default_name) or default_name)
    if hasattr(file_obj, "read"):
        position = file_obj.tell() if hasattr(file_obj, "tell") else None
        payload = file_obj.read()
        if position is not None and hasattr(file_obj, "seek"):
            file_obj.seek(position)
        payload = _coerce_bytes(payload)
        return {
            "name": os.path.basename(name),
            "bytes": payload,
            "mime_type": guess_mime_type(name=name, mime_type=mime_type, payload=payload),
        }

    payload = _coerce_bytes(file_obj)
    return {
        "name": default_name,
        "bytes": payload,
        "mime_type": guess_mime_type(name=default_name, mime_type=mime_type, payload=payload),
    }


def open_upload(file_payload: Any) -> Any:
    if isinstance(file_payload, dict):
        return NamedBytesIO(
            _coerce_bytes(file_payload.get("bytes", b"")),
            str(file_payload.get("name") or "upload.bin"),
        )
    return file_payload


def coerce_media_source(
    value: Any,
    *,
    default_name: str,
    mime_type: str | None = None,
) -> dict[str, Any]:
    if isinstance(value, dict):
        source_url = value.get("url") or value.get("uri") or value.get("file_uri")
        if source_url:
            parsed = parse_data_url(str(source_url))
            if parsed:
                payload = parsed["bytes"]
                return {
                    "name": str(value.get("name") or value.get("filename") or default_name),
                    "bytes": payload,
                    "mime_type": guess_mime_type(
                        name=value.get("name") or value.get("filename") or default_name,
                        mime_type=value.get("mime_type") or parsed["mime_type"] or mime_type,
                        payload=payload,
                    ),
                }
            return {
                "name": str(value.get("name") or value.get("filename") or default_name),
                "uri": str(source_url),
                "mime_type": str(value.get("mime_type") or mime_type or ""),
            }

        if value.get("path"):
            return materialize_upload(
                value["path"],
                default_name=str(value.get("name") or value.get("filename") or default_name),
                mime_type=value.get("mime_type") or mime_type,
            )

    if isinstance(value, str):
        parsed = parse_data_url(value)
        if parsed:
            payload = parsed["bytes"]
            return {
                "name": default_name,
                "bytes": payload,
                "mime_type": guess_mime_type(
                    name=default_name,
                    mime_type=parsed["mime_type"] or mime_type,
                    payload=payload,
                ),
            }
        if re.match(r"^(?:https?|gs|file)://", value, re.IGNORECASE):
            return {"name": default_name, "uri": value, "mime_type": mime_type or ""}

    payload = materialize_upload(value, default_name=default_name, mime_type=mime_type)
    return payload


def _normalize_part_type(part_type: str) -> str:
    normalized = str(part_type or "").strip().lower()
    aliases = {
        "input_text": "text",
        "input_image": "image",
        "image_url": "image",
        "input_file": "file",
        "document": "file",
        "input_document": "file",
        "input_audio": "audio",
        "audio_url": "audio",
        "input_video": "video",
    }
    return aliases.get(normalized, normalized or "text")


def extract_content_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    parsed_parts: list[dict[str, Any]] = []
    for raw_part in content:
        if isinstance(raw_part, str):
            parsed_parts.append({"type": "text", "text": raw_part})
            continue
        if not isinstance(raw_part, dict):
            parsed_parts.append({"type": "text", "text": str(raw_part)})
            continue

        part_type = _normalize_part_type(raw_part.get("type"))
        if part_type == "text":
            text = raw_part.get("text")
            if text is None:
                text = raw_part.get("value")
            parsed_parts.append({"type": "text", "text": "" if text is None else str(text)})
            continue

        source = (
            raw_part.get(part_type)
            or raw_part.get(f"{part_type}_url")
            or raw_part.get(f"{part_type}_data")
            or raw_part.get("file")
            or raw_part.get("image_url")
            or raw_part.get("audio")
            or raw_part.get("video")
            or raw_part.get("value")
        )
        source_dict = coerce_media_source(
            source,
            default_name=str(
                raw_part.get("name") or raw_part.get("filename") or f"{part_type}.bin"
            ),
            mime_type=raw_part.get("mime_type") or raw_part.get("media_type"),
        )
        parsed_part = {"type": part_type, **source_dict}
        if raw_part.get("detail"):
            parsed_part["detail"] = raw_part.get("detail")
        parsed_parts.append(parsed_part)

    return parsed_parts


def extract_text_content(content: Any) -> str:
    text_parts = []
    for part in extract_content_parts(content):
        if part["type"] == "text":
            text_parts.append(str(part.get("text", "")))
    return "\n".join(part for part in text_parts if part).strip()


def content_signature(content: Any) -> str:
    parts = extract_content_parts(content)
    if not parts:
        return ""

    signature_parts = []
    for part in parts:
        part_type = part.get("type", "text")
        if part_type == "text":
            signature_parts.append(f"text::{part.get('text', '')}")
            continue

        if part.get("bytes") is not None:
            digest = hashlib.sha256(_coerce_bytes(part["bytes"])).hexdigest()[:16]
            location = f"sha256:{digest}"
        else:
            location = str(part.get("uri") or "")
        signature_parts.append(
            "::".join(
                [
                    part_type,
                    str(part.get("mime_type") or ""),
                    str(part.get("name") or part_type),
                    location,
                ]
            )
        )

    return "\n".join(signature_parts)
