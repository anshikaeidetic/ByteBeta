# Byte Service Topology

This page explains how requests move through the Byte runtime.

## Topology

```
application or OpenAI SDK
  |
  v
byte_server
  |- auth and request limits
  |- exact / semantic cache
  |- routing and trust policy
  |- control-plane state
  |
  +--> byte_memory
  |      |- tool-result reuse
  |      |- reasoning summary or prefix reuse
  |      |- session summary
  |      `- intent graph
  |
  `--> byte_inference
         |- local model execution
         |- worker runtime stats
         `- worker-local cached state
```

## Public, admin, and internal boundaries

- public: caller-facing chat and proxy routes on `byte_server`
- admin: cache management, control-plane inspection, MCP, and memory artifact routes on `byte_server`
- internal: `/internal/v1/*` on `byte_memory` and `byte_inference`

Only `byte_server` should be internet-facing.

## Control-plane state

The control plane stores:

- worker heartbeats and health
- worker/session affinity data
- replay jobs and outcomes
- recommendation state
- cache inspection events

The repository currently uses SQLite as the initial SQL-backed implementation.
