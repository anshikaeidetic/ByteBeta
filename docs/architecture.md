# Byte Architecture Overview

Byte has one public identity and two runtime shapes.

## Product surfaces

- `byte`: the Python runtime, cache system, routing, trust, prompt distillation, telemetry, quantization, and CLI
- `byte_server`: the public FastAPI gateway and control plane
- `byte_inference`: the private inference worker service
- `byte_memory`: the private memory service

## Adoption modes

### Embedded or single-node

Use:

- `ByteClient`
- `byte start`
- `byte_server`

This shape is the fastest way to validate proxy compatibility or embed Byte inside an application process.

### Split-service

Use:

- `byte_server` as the public ingress and control plane
- `byte_memory` for private memory resolution and storage policy
- `byte_inference` for private worker execution and worker-local state

This shape is the production topology when Byte is responsible for worker selection, replay-backed optimization, and local-runtime behavior.

## Why the split exists

The public gateway and the private runtime do different jobs:

- the control plane owns auth, request policy, cache decisions, routing, replay, and inspection
- inference workers own execution and worker-local state
- the memory service owns tool reuse, reasoning summaries or prefixes, session summaries, and intent graph state

That separation keeps the public ingress small and lets local inference behavior evolve without turning `byte_server` back into a monolith.
