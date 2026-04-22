# Byte Route And Auth Matrix

| Surface | Routes | Auth model | Notes |
| --- | --- | --- | --- |
| Public probes | `/healthz`, `/readyz` | none | HTTPS policy can still block non-compliant ingress |
| Metrics | `/metrics` | public by default; admin when admin auth is enabled | keep behind the same ingress policy as the gateway |
| Public chat | `/v1/chat/completions`, `/byte/gateway/chat` | BYO bearer token, or server-managed creds behind admin auth | OpenAI-compatible public entrypoint is `/v1/chat/completions` |
| Public proxy | `/byte/gateway/images`, `/byte/gateway/moderations`, `/byte/gateway/audio/*` | same as public chat | upload limits apply to audio routes |
| Cache admin | `/put`, `/get`, `/flush`, `/invalidate`, `/clear`, `/stats`, `/warm`, `/feedback`, `/quality` | admin token | management surface only |
| Memory admin | `/memory`, `/memory/recent`, `/memory/import`, `/memory/export_artifact`, `/memory/import_artifact` | admin token | artifact routes also require export-root policy |
| MCP | `/byte/mcp/*` | admin token | tool registration and execution |
| Control plane | `/byte/control/*` | admin token | inspection, replay, recommendations, ROI, settings |
| Memory service internal | `byte_memory /internal/v1/*` | internal token when configured | `/healthz` stays open |
| Inference internal | `byte_inference /internal/v1/*` | internal token when configured | `/healthz` stays open |
