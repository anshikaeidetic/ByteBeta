"""Request and response scope helpers for the Byte control plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _message_text(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "") or item.get("content", "") or ""))
            else:
                parts.append(str(item or ""))
        return " ".join(part for part in parts if part).strip()
    return str(value or "").strip()


def request_text(request_kwargs: dict[str, Any] | None) -> str:
    request_kwargs = request_kwargs or {}
    messages = request_kwargs.get("messages") or []
    if messages:
        return "\n".join(
            _message_text(message.get("content", ""))
            for message in messages
            if _message_text(message.get("content", ""))
        ).strip()
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "").strip()
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "").strip()
    return ""


def response_text(payload: Any) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices:
            choice = choices[0] or {}
            message = choice.get("message") or {}
            if message.get("content") not in (None, ""):
                return _message_text(message.get("content"))
            if choice.get("text") not in (None, ""):
                return str(choice.get("text") or "").strip()
        if payload.get("answer") not in (None, ""):
            return str(payload.get("answer") or "").strip()
    return str(payload or "").strip()

@dataclass(frozen=True)
class RequestScope:
    tenant: str
    session: str
    workflow: str
    source: str = "derived"

    @property
    def scope_key(self) -> str:
        return f"{self.tenant}:{self.session}:{self.workflow}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
