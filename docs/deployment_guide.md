# Byte Deployment Guide

This guide covers the supported deployment shapes for Byte and the operational choices that matter before production rollout.

## Choose a topology

| Topology | Use when | Components |
| --- | --- | --- |
| Embedded SDK | local development, scripts, single-process services | application process + `ByteClient` |
| Single-node gateway | proxy-first adoption, drop-in OpenAI replacement | `byte_server` |
| Split-service | private local inference, memory isolation, replay-backed optimization | `byte_server` + `byte_memory` + `byte_inference` |

## Single-node gateway

Start the public gateway with exact-match caching only:

```bash
byte init
byte start
```

For local evaluation only, the repository also ships a `docker-compose.yml`
stack that mirrors the split-service shape. It installs from the live checkout
at container start, so it is not a production deployment recipe.

For a hardened public gateway, run `byte_server` directly:

```bash
byte_server \
  --gateway True \
  --host 0.0.0.0 \
  --port 8000 \
  --cache-dir /var/lib/byte/cache \
  --security-mode \
  --security-require-https \
  --security-rate-limit-public-per-minute 240 \
  --security-rate-limit-admin-per-minute 60 \
  --security-max-inflight-public 64 \
  --security-max-inflight-admin 16
```

Use this shape when:

- the primary goal is OpenAI-compatible proxy replacement
- hosted providers handle model execution
- local worker affinity and memory-service separation are not required yet

## Split-service production topology

Use the split-service topology when Byte owns routing, replay, private worker affinity, and memory resolution.

```
client
  |
  v
byte_server (public control plane)
  |- exact / semantic cache lookup
  |- request security and audit boundary
  |- worker registry + replay state (SQLite / SQL)
  |- memory resolve / remember calls
  |
  +--> byte_memory (private)
  |
  +--> byte_inference pool (private)
```

Recommended wiring:

- `byte_server` is the only public ingress
- `byte_memory` and `byte_inference` stay on a private network
- control-plane metadata stays on persistent storage
- cache storage stays on persistent storage when you need warm restarts
- internal service calls use `BYTE_INTERNAL_TOKEN`

## Secrets and credentials

Use three separate secret classes:

1. Admin boundary: `BYTE_ADMIN_TOKEN`
2. Provider credentials: `BYTE_BACKEND_API_KEY` or native provider keys such as `OPENAI_API_KEY`
3. Internal service authentication: `BYTE_INTERNAL_TOKEN`

Keep them separate. Rotating one should not force rotation of the others.

## State and storage

At minimum, plan these storage layers explicitly:

- control-plane DB: worker registry, replay jobs, recommendations, and cache inspection state
- cache storage: request/response cache data
- audit log path: write-once or managed log shipping path
- export root: controlled location for memory artifact import/export

`byte_memory` is CPU-oriented and stateless by default in this repository version. If you need persistence across restarts, persist the underlying stores or snapshot/export them operationally.

## Safe rollout checklist

- start in exact or normalized cache mode first
- enable admin auth before exposing operator or artifact surfaces
- keep `/metrics`, `/healthz`, and `/readyz` under the same ingress and TLS policy as the rest of the service
- verify `BYTE_INTERNAL_TOKEN` on all private services before enabling worker URLs or memory-service URLs
- use BYO provider credentials first when evaluating routing behavior
- only enable server-managed credentials where the admin boundary is already enforced

## Production smoke test

After deploy:

1. `GET /healthz`
2. `GET /readyz`
3. `GET /metrics`
4. `POST /v1/chat/completions` with a BYO bearer token
5. `POST /byte/control/cache/inspect` with the admin token
6. For split-service deploys, confirm the control plane can reach `/internal/v1/runtime` on workers and memory service

## Before merge

Run the repo-owned validation gates:

```bash
python scripts/run_tox.py -e hygiene,lint,typecheck,package,compile
python scripts/run_unit_tests.py
python scripts/run_coverage.py
python scripts/run_integration_smoke.py
python scripts/run_security_checks.py
```
