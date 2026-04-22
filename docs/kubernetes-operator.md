# Byte Kubernetes Operator

The bundled operator reconciles `ByteCache` custom resources into the Byte runtime topology.

## What the operator renders

Always:

- a public `Deployment` for `byte_server`
- a public `Service` for the gateway
- an optional cache `PersistentVolumeClaim`

When enabled:

- a private `Deployment` and `Service` for `byte_memory`
- a `StatefulSet` and headless `Service` per inference pool

## Secret and env wiring

The operator now supports three useful patterns:

1. top-level shared `env` / `envFrom`
2. gateway-only `gateway.env` / `gateway.envFrom`
3. a dedicated `internalAuthSecretRef` for `BYTE_INTERNAL_TOKEN`

Use gateway-only env for admin and provider credentials. Use `internalAuthSecretRef` for the shared internal token consumed by `byte_server`, `byte_memory`, and `byte_inference`.

## Minimal install

```bash
kubectl apply -f examples/kubernetes/bytecache-crd.yaml
kubectl apply -f examples/kubernetes/bytecache-rbac.yaml
kubectl apply -f examples/kubernetes/bytecache-operator.yaml
kubectl apply -f examples/kubernetes/bytecache-sample.yaml
```

## Operational model

The operator manages:

- container args for gateway, control-plane, and observability settings
- readiness and liveness probes
- persistent cache storage when requested
- service creation for the gateway, memory service, and inference pools

The platform still owns:

- ingress and TLS
- secret creation and rotation
- database and storage class policy
- autoscaling and node placement
- network policy

## Recommended secret layout

- `byte-gateway`: `admin-token`, `backend-api-key`
- `byte-internal`: `token`

Keep the gateway secrets distinct from the internal service token so rotation and access review stay clear.

## Related docs

- [Deployment guide](deployment_guide.md)
- [Environment reference](environment-reference.md)
- [Operator runbook](operator-runbook.md)
