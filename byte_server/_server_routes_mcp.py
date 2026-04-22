"""MCP gateway routes for the Byte server."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from byte_server._server_security import _audit_event, _raise_route_error, _require_admin
from byte_server._server_state import ServerServices
from byte_server.models import MCPToolCall, MCPToolRegistration


def register_mcp_routes(app: FastAPI, services: ServerServices) -> None:
    @app.get(f"{services.mcp_root}/tools")
    async def list_mcp_tools(request: Request) -> Any:
        _require_admin(services, request, "mcp.list")
        payload = {"tools": services.runtime_state().mcp_gateway.list_tools()}
        _audit_event(services, request, "mcp.list", status="success", metadata={"count": len(payload["tools"])})
        return JSONResponse(content=payload)

    @app.get(f"{services.mcp_root}/tool-specs")
    async def list_mcp_tool_specs(request: Request) -> Any:
        _require_admin(services, request, "mcp.list")
        payload = {"tools": services.runtime_state().mcp_gateway.openai_tool_specs()}
        _audit_event(services, request, "mcp.list", status="success", metadata={"count": len(payload["tools"])})
        return JSONResponse(content=payload)

    @app.post(f"{services.mcp_root}/register")
    async def register_mcp_tool(request: Request, registration: MCPToolRegistration) -> Any:
        _require_admin(services, request, "mcp.register")
        descriptor = services.runtime_state().mcp_gateway.register_tool(**registration.model_dump())
        _audit_event(
            services,
            request,
            "mcp.register",
            status="success",
            metadata={"server_name": descriptor.server_name, "tool_name": descriptor.tool_name},
        )
        return JSONResponse(content={"tool": descriptor.to_dict()})

    @app.post(f"{services.mcp_root}/call")
    async def call_mcp_tool(request: Request, payload: MCPToolCall) -> Any:
        _require_admin(services, request, "mcp.call")
        try:
            result = await services.run_in_threadpool()(
                services.runtime_state().mcp_gateway.call_tool,
                payload.server_name,
                payload.tool_name,
                payload.arguments,
                cache_obj=services.active_cache(),
                scope=payload.scope,
                timeout_s=payload.timeout_s,
            )
            _audit_event(
                services,
                request,
                "mcp.call",
                status="success",
                metadata={
                    "server_name": payload.server_name,
                    "tool_name": payload.tool_name,
                    "cached": bool(result.get("cached", False)),
                },
            )
            return JSONResponse(content=result)
        except Exception as exc:  # pylint: disable=W0703
            _raise_route_error(
                services,
                request,
                "mcp.call",
                exc,
                public_detail="Byte MCP gateway request failed.",
            )


__all__ = ["register_mcp_routes"]
