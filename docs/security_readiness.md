# Byte Security Readiness

Byte's security model is practical: the public gateway, admin surfaces, and private internal services each have distinct controls, and the repository only claims the boundaries it actually enforces.

## Boundary model

- Public surfaces: chat and proxy routes intended for callers and applications
- Admin surfaces: cache management, memory artifact management, MCP, and control-plane inspection/settings
- Internal surfaces: `byte_inference` and `byte_memory` `/internal/v1/*` endpoints

The public gateway is the only service that should be internet-exposed.

## Authentication model

### Public chat and proxy routes

Allowed patterns:

- caller supplies provider credentials with `Authorization: Bearer ...`
- server-managed credentials are used only when the gateway is configured for them

When the admin boundary is enabled, requests that depend on server-managed credentials require the admin token as well.

### Admin routes

Admin routes are protected by `BYTE_ADMIN_TOKEN` or the equivalent CLI/config setting. This applies to:

- cache management routes
- memory snapshot and artifact routes
- MCP registration and tool execution
- control-plane inspection, replay, recommendation, ROI, and settings routes

### Internal service routes

`byte_inference` and `byte_memory` accept `X-Byte-Internal-Token` on `/internal/v1/*` when `BYTE_INTERNAL_TOKEN` or `--internal-auth-token` is configured.

Unauthenticated health probe:

- `GET /healthz`

Authenticated internal surfaces:

- `GET /internal/v1/runtime`
- worker `POST /internal/v1/heartbeat`, `POST /internal/v1/prefill`, `POST /internal/v1/events/kv`, `POST /internal/v1/generate`
- memory `GET /internal/v1/intent-graph`, `POST /internal/v1/resolve`, `POST /internal/v1/remember`

## Credential handling

Byte supports two provider-credential modes:

1. BYO credentials supplied per request by the caller
2. Server-managed credentials stored in the runtime environment

Guidance:

- prefer BYO credentials when onboarding a new integration
- use server-managed credentials only after the admin boundary is enabled
- keep provider credentials separate from `BYTE_ADMIN_TOKEN` and `BYTE_INTERNAL_TOKEN`

## Transport and egress controls

Byte already supports:

- HTTPS enforcement for non-loopback traffic
- optional proxy-header trust
- outbound host override validation
- allowed egress host restrictions
- public and admin rate limiting
- public and admin concurrency limits

These guardrails are designed to fit the gateway runtime that already exists. They are not abstract policy scaffolding.

## Upload, archive, and artifact handling

The gateway enforces configured size limits for:

- JSON request bodies
- file uploads on audio routes
- cache archive download size and member count

Artifact import/export is constrained by the configured export root. In security mode, Byte will refuse artifact import/export when the export root is not configured.

## Auditability and redaction

Byte supports:

- audit logging for admin actions and gateway requests
- metadata-only request previews instead of full raw prompt bodies
- redacted public 5xx errors in security mode
- secure headers on responses in security mode

Audit logs should be shipped off-node or written to a managed volume. Treat them as security records, not temporary debug output.

## Operator guidance

On Kubernetes:

- keep `byte_server` public and keep `byte_memory` / `byte_inference` private
- propagate `BYTE_INTERNAL_TOKEN` through a secret-backed env reference
- keep admin and provider credentials in distinct secrets
- avoid literal secret values in manifests

See:

- [Route and auth matrix](route-auth-matrix.md)
- [Kubernetes operator](kubernetes-operator.md)
- [Operator runbook](operator-runbook.md)
