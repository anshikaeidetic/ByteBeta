from unittest.mock import patch

from byte import Cache
from byte.mcp_gateway import MCPGateway


class _Response:
    def __init__(self, payload) -> None:
        self._payload = payload
        self.headers = {"content-type": "application/json"}
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_mcp_gateway_caches_read_only_tools() -> None:
    gateway = MCPGateway()
    gateway.register_tool(
        server_name="docs",
        tool_name="search",
        endpoint="https://example.com/search",
        cache_policy="read_only",
    )
    cache_obj = Cache()

    with patch(
        "byte.mcp_gateway.requests.request", return_value=_Response({"items": [1, 2, 3]})
    ) as mock_request:
        first = gateway.call_tool("docs", "search", {"q": "byte"}, cache_obj=cache_obj)
        second = gateway.call_tool("docs", "search", {"q": "byte"}, cache_obj=cache_obj)

    assert mock_request.call_count == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["result"] == {"items": [1, 2, 3]}


def test_mcp_gateway_does_not_cache_side_effecting_tools() -> None:
    gateway = MCPGateway()
    gateway.register_tool(
        server_name="deploy",
        tool_name="release",
        endpoint="https://example.com/release",
        cache_policy="write",
    )
    cache_obj = Cache()

    with patch(
        "byte.mcp_gateway.requests.request", return_value=_Response({"ok": True})
    ) as mock_request:
        gateway.call_tool("deploy", "release", {"version": "1.2.3"}, cache_obj=cache_obj)
        gateway.call_tool("deploy", "release", {"version": "1.2.3"}, cache_obj=cache_obj)

    assert mock_request.call_count == 2


def test_mcp_gateway_exposes_openai_tool_specs() -> None:
    gateway = MCPGateway()
    gateway.register_tool(
        server_name="docs",
        tool_name="search",
        endpoint="https://example.com/search",
        description="Search docs",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )

    specs = gateway.openai_tool_specs()

    assert specs[0]["type"] == "function"
    assert specs[0]["function"]["name"] == "docs__search"
