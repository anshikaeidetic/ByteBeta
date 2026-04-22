# Byte Inference Worker

`byte_inference` is the private worker service used by the control plane.

## Responsibilities

- local model execution
- worker runtime reporting
- worker inventory reporting
- worker-scoped generation metadata

## Internal routes

- `GET /healthz`
- `GET /internal/v1/runtime`
- `POST /internal/v1/heartbeat`
- `POST /internal/v1/prefill`
- `POST /internal/v1/events/kv`
- `POST /internal/v1/generate`

All `/internal/v1/*` routes require `X-Byte-Internal-Token` when internal auth is configured.

## Worker identity

Configure workers with:

- `BYTE_WORKER_ID`
- `BYTE_WORKER_MODELS`
- `BYTE_WORKER_CACHE_DIR`
- `BYTE_WORKER_FREE_VRAM_GB`

The control plane uses worker inventory and worker health to select a target. The worker returns `byte_worker` metadata on generation responses so the control plane can record affinity.
