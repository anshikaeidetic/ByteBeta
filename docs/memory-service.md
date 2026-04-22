# Byte Memory Service

`byte_memory` is the private service that resolves reusable context before model execution and records reusable context after execution.

## Resolution order

The memory service resolves context in a fixed order:

1. tool-result deduplication
2. reasoning prefix or reasoning summary reuse
3. session summary
4. retrieval context

That ordering is important because the cheap deterministic hits should win before broader retrieval context is appended.

## Provider policy

- local provider mode: full reasoning-prefix reuse is allowed
- hosted provider mode: summary-and-trace reuse only

Hosted providers do not receive a hidden-prefix replay contract.

## Internal routes

- `GET /healthz`
- `GET /internal/v1/runtime`
- `GET /internal/v1/intent-graph`
- `POST /internal/v1/resolve`
- `POST /internal/v1/remember`

All `/internal/v1/*` routes require `X-Byte-Internal-Token` when internal auth is configured.
