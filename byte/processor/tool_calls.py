"""Pre-processing functions for tool/function call caching.

These functions extract cacheable keys from LLM requests that include
tool definitions or function calls, allowing the cache to match on
the *intent* of a tool call rather than the full verbose request.
"""

import hashlib
import json
from typing import Any


def last_content_with_tools(data: dict[str, Any], **params) -> Any:
    """Extract the last user message combined with a stable tool-call signature.

    When the request contains ``tools`` or ``functions``, the tool names
    and the last user message are combined into a composite cache key.
    This allows the cache to match on the *intent* — "call weather API
    for London" — regardless of whether the surrounding prompt template
    changes slightly.

    :param data: the user LLM request data
    :return: composite cache key string

    Example:
        .. code-block:: python

            from byte.processor.pre import last_content_with_tools

            key = last_content_with_tools({
                "messages": [{"role": "user", "content": "weather in London"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}]
            })
            # key = "get_weather|weather in London"
    """
    messages = data.get("messages", [])
    last_msg = messages[-1]["content"] if messages else ""

    # Extract a stable tool signature from the request
    tool_sig = _extract_tool_signature(data)

    if tool_sig:
        return f"{tool_sig}|{last_msg}"
    return last_msg


def request_tool_signature(data: dict[str, Any]) -> str:
    """Public helper that returns the stable tool signature for a request."""
    return _extract_tool_signature(data)


def _extract_tool_signature(data: dict[str, Any]) -> str:
    """Build a stable, sorted signature from tool/function definitions.

    Only the tool *names* are used (not the full JSON schemas) so that
    minor schema wording changes don't bust the cache.
    """
    tools = data.get("tools") or data.get("functions") or []
    names: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            # OpenAI v2 format: {"type": "function", "function": {"name": ...}}
            func = tool.get("function", tool)
            name = func.get("name", "")
            if name:
                names.append(name)
    names.sort()
    return ",".join(names)


def tool_call_response_key(response: Any) -> str | None:
    """Extract a deterministic cache key from a tool-call *response*.

    Given an LLM response that includes ``tool_calls``, this produces a
    stable hash of ``(function_name, arguments_json)`` so that identical
    tool invocations hit the cache.

    :param response: the LLM response dict or object
    :return: hash string or None if no tool calls present
    """
    tool_calls = _get_tool_calls(response)
    if not tool_calls:
        return None

    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        # Normalize arguments JSON by sorting keys
        try:
            args = json.loads(func.get("arguments", "{}"))
            args_normalized = json.dumps(args, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            args_normalized = func.get("arguments", "")
        parts.append(f"{name}:{args_normalized}")

    parts.sort()
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _get_tool_calls(response: Any) -> list[dict]:
    """Extract tool_calls from various response formats."""
    if isinstance(response, dict):
        # OpenAI dict format
        choices = response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("tool_calls", [])
    elif hasattr(response, "choices"):
        # OpenAI Pydantic object format
        if response.choices:
            msg = response.choices[0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
    return []
