# Byte MCP Gateway

Byte includes an MCP gateway on the public server for tool registration and tool execution.

## Routes

- `GET /byte/mcp/tools`
- `GET /byte/mcp/tool-specs`
- `POST /byte/mcp/register`
- `POST /byte/mcp/call`

## Boundary

The MCP surface is admin-only. Treat it as an operator integration surface, not a public client endpoint.

## Caching model

Registered tools can declare a cache policy. Read-only tool calls can be cached through the active Byte cache so repeated MCP calls do not always hit the upstream endpoint.

## Security notes

- keep MCP registration behind the admin token
- keep outbound targets inside the configured egress policy when provider host overrides are enabled
- audit tool registration and tool execution like the rest of the admin plane
