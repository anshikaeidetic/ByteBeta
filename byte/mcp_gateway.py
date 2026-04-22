import json
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from byte import Cache, cache
from byte.processor.optimization_memory import compact_text, summarize_artifact_payload

_READ_ONLY_POLICIES = {"read_only", "fresh_read"}


@dataclass(frozen=True)
class MCPToolDescriptor:
    server_name: str
    tool_name: str
    endpoint: str
    method: str = "POST"
    input_schema: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    cache_policy: str = "read_only"
    ttl: float | None = None
    timeout_s: float | None = None
    headers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def openai_tool_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": f"{self.server_name}__{self.tool_name}",
                "description": self.description
                or f"MCP tool {self.tool_name} from {self.server_name}",
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


class MCPGateway:
    """Bridge registered MCP-like tools into deterministic HTTP calls with Byte caching."""

    def __init__(self) -> None:
        self._tools: dict[str, MCPToolDescriptor] = {}

    def register_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        endpoint: str,
        method: str = "POST",
        input_schema: dict[str, Any] | None = None,
        description: str = "",
        cache_policy: str = "read_only",
        ttl: float | None = None,
        timeout_s: float | None = None,
        headers: dict[str, Any] | None = None,
    ) -> MCPToolDescriptor:
        descriptor = MCPToolDescriptor(
            server_name=str(server_name or "").strip(),
            tool_name=str(tool_name or "").strip(),
            endpoint=str(endpoint or "").strip(),
            method=str(method or "POST").upper(),
            input_schema=dict(input_schema or {}),
            description=str(description or "").strip(),
            cache_policy=str(cache_policy or "read_only").strip().lower(),
            ttl=ttl,
            timeout_s=timeout_s,
            headers=dict(headers or {}),
        )
        if not descriptor.server_name or not descriptor.tool_name or not descriptor.endpoint:
            raise ValueError("server_name, tool_name, and endpoint are required")
        self._tools[self._key(descriptor.server_name, descriptor.tool_name)] = descriptor
        return descriptor

    def list_tools(self) -> list[dict[str, Any]]:
        return [descriptor.to_dict() for _, descriptor in sorted(self._tools.items())]

    def openai_tool_specs(self) -> list[dict[str, Any]]:
        return [descriptor.openai_tool_spec() for _, descriptor in sorted(self._tools.items())]

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        cache_obj: Cache | None = None,
        scope: str = "",
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        descriptor = self._tools.get(self._key(server_name, tool_name))
        if descriptor is None:
            raise KeyError(f"Unknown MCP tool: {server_name}/{tool_name}")
        cache_obj = cache_obj or cache
        normalized_args = self._normalize_arguments(arguments)
        cache_scope = str(scope or descriptor.server_name)
        effective_timeout = self._resolve_timeout(
            descriptor,
            cache_obj=cache_obj,
            timeout_s=timeout_s,
        )
        cache_key = {
            "server": descriptor.server_name,
            "tool": descriptor.tool_name,
            "endpoint": descriptor.endpoint,
            "method": descriptor.method,
            "arguments": normalized_args,
        }
        if descriptor.cache_policy in _READ_ONLY_POLICIES:
            cached = cache_obj.recall_tool_result(
                f"mcp::{descriptor.server_name}::{descriptor.tool_name}",
                cache_key,
                scope=cache_scope,
                include_metadata=True,
            )
            if cached:
                return {
                    "server_name": descriptor.server_name,
                    "tool_name": descriptor.tool_name,
                    "cached": True,
                    "cache_policy": descriptor.cache_policy,
                    "result": cached.get("result"),
                    "summary": summarize_artifact_payload(
                        "tool_result_context", cached.get("result"), max_chars=220
                    ),
                }

        result = self._http_call(descriptor, normalized_args, timeout_s=effective_timeout)
        summary = compact_text(
            summarize_artifact_payload("tool_result_context", result, max_chars=320),
            max_chars=320,
        )
        if descriptor.cache_policy in _READ_ONLY_POLICIES:
            cache_obj.remember_tool_result(
                f"mcp::{descriptor.server_name}::{descriptor.tool_name}",
                cache_key,
                result,
                ttl=descriptor.ttl,
                scope=cache_scope,
            )
        return {
            "server_name": descriptor.server_name,
            "tool_name": descriptor.tool_name,
            "cached": False,
            "cache_policy": descriptor.cache_policy,
            "result": result,
            "summary": summary,
        }

    @staticmethod
    def _http_call(
        descriptor: MCPToolDescriptor, arguments: dict[str, Any], *, timeout_s: float
    ) -> Any:
        method = descriptor.method.upper()
        request_kwargs = {
            "headers": dict(descriptor.headers or {}),
            "timeout": timeout_s,
        }
        if method == "GET":
            request_kwargs["params"] = arguments
        else:
            request_kwargs["json"] = arguments
        response = requests.request(method, descriptor.endpoint, **request_kwargs)
        response.raise_for_status()
        content_type = str(response.headers.get("content-type", "") or "").lower()
        if "application/json" in content_type:
            return response.json()
        text = response.text
        try:
            return json.loads(text)
        except Exception:  # pylint: disable=W0703
            return {"text": text}

    @staticmethod
    def _key(server_name: str, tool_name: str) -> str:
        return f"{str(server_name or '').strip().lower()}::{str(tool_name or '').strip().lower()}"

    @staticmethod
    def _normalize_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            return {}
        return json.loads(json.dumps(arguments, sort_keys=True))

    @staticmethod
    def _resolve_timeout(
        descriptor: MCPToolDescriptor,
        *,
        cache_obj: Cache | None,
        timeout_s: float | None,
    ) -> float:
        candidates = [
            timeout_s,
            descriptor.timeout_s,
            getattr(getattr(cache_obj, "config", None), "mcp_timeout_s", None),
            30.0,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            resolved = float(candidate)
            if resolved <= 0:
                raise ValueError("timeout_s must be > 0")
            return resolved
        return 30.0
