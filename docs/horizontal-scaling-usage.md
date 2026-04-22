# Horizontal Scaling

Byte can be deployed across multiple nodes when the cache metadata and
vector/scalar backends are shared rather than kept purely in process.

## Single-Node Default

A single-node deployment is simplest when:

- request volume is modest
- local latency is the main goal
- in-process state is acceptable

## Multi-Node Direction

Horizontal scaling typically requires:

- shared scalar storage
- shared vector storage
- distributed eviction or coordination where needed
- external process supervision and load balancing

## Deployment Probes

`byte_server` now exposes:

- `GET /healthz` for liveness
- `GET /readyz` for readiness
- `GET /metrics` for Prometheus-compatible scraping
- OTLP trace and metric export when OpenTelemetry is enabled
- Kubernetes operator manifests under `examples/kubernetes`

These endpoints make container orchestration and autoscaler integration easier.
The repository now includes a Kubernetes operator, but it still does not ship a
custom autoscaling controller.

## Current Boundaries

Be explicit about the remaining gaps:

- no bundled dashboard or alerting plane
- vector-store pooling is still backend-specific
- no built-in autoscaling controller
