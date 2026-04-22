# Byte Control Plane

`byte_server` is the public control plane.

## Responsibilities

- OpenAI-compatible public routing
- security boundary enforcement
- exact and semantic cache lookup
- scope extraction from headers and request metadata
- worker discovery and worker selection
- replay scheduling and recommendation state
- control-plane inspection APIs

## Lifecycle

At startup, the control plane:

1. initializes the cache and gateway runtime
2. opens the control-plane database
3. loads persisted worker and replay state
4. reads configured worker URLs and memory-service URL

At request time, the control plane:

1. authenticates and validates the request
2. resolves scope and route intent
3. checks cache policy
4. optionally resolves memory context
5. selects a worker when local runtime execution is needed
6. records cache inspection and replay metadata

## Internal service auth

When `BYTE_INTERNAL_TOKEN` is configured, the control plane sends `X-Byte-Internal-Token` on calls to `byte_memory` and `byte_inference`.

The token is not exposed in control-plane inspection output. The inspection surface only reports whether internal auth is configured.
