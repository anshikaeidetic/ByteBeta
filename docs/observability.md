# Byte Observability

Byte exposes three observability surfaces:

1. health and readiness endpoints
2. Prometheus-style metrics from `byte_server`
3. OTLP export for traces and metrics

## Health and readiness

Gateway probes:

- `GET /healthz`
- `GET /readyz`

Private service probe:

- `GET /healthz` on `byte_inference`
- `GET /healthz` on `byte_memory`

Use readiness to detect cache/runtime initialization issues. Use health to detect process availability.

## Prometheus metrics

`byte_server` exposes `GET /metrics`.

The metrics and service identifiers intentionally retain their compatibility names:

- metric namespace: `byteai_*`
- default Datadog service name examples: `byteai-cache`

That is an operational compatibility choice, not public product naming.

Control-plane signals already exported through the gateway metrics surface include:

- active worker count
- worker-routed request count
- recommendation count
- projected savings

## OpenTelemetry

Library mode can emit OpenTelemetry spans and metrics for cache stages such as preprocessing, embedding, search, evaluation, LLM calls, and save operations.

Typical gateway run:

```bash
byte_server \
  --host 0.0.0.0 \
  --port 8000 \
  --gateway True \
  --otel-enabled \
  --otel-endpoint http://otel-collector:4318 \
  --otel-environment production
```

Datadog-oriented run:

```bash
byte_server \
  --host 0.0.0.0 \
  --port 8000 \
  --gateway True \
  --datadog-enabled \
  --datadog-agent-host datadog-agent.monitoring.svc.cluster.local \
  --datadog-service byteai-cache \
  --datadog-env production \
  --datadog-version 1.0.5
```

## Operating guidance

- protect `/metrics` with the admin boundary when that boundary is enabled
- tag telemetry with deployment environment and service namespace
- scrape the public gateway; do not assume private services are exposed for scraping
- use the control-plane inspection routes when you need per-request cache reasoning, not just raw counters

## Related docs

- [Deployment guide](deployment_guide.md)
- [Kubernetes operator](kubernetes-operator.md)
- [Operator runbook](operator-runbook.md)
