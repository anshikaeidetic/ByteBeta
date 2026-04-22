# Byte Operator Runbook

Use this runbook when operating Byte on Kubernetes.

## Preflight

- confirm the CRD, RBAC, and operator deployment are applied
- create the gateway secret and internal token secret
- verify the runtime image tags and namespaces in the sample manifest
- confirm persistent storage policy for the cache directory

## First deploy smoke test

1. wait for the gateway deployment to become ready
2. if enabled, wait for the memory deployment to become ready
3. if enabled, wait for each inference StatefulSet pod to become ready
4. hit gateway `/healthz`
5. hit gateway `/readyz`
6. hit gateway `/metrics`
7. send a BYO-credential `POST /v1/chat/completions`
8. check `/byte/control/cache/inspect` with the admin token

## Common failure modes

### Gateway ready but worker routing never triggers

- confirm `controlPlane.workerUrls`
- confirm workers are reachable from the gateway pod
- confirm `BYTE_INTERNAL_TOKEN` matches on gateway and workers

### Memory context never appears

- confirm `controlPlane.memoryServiceUrl`
- confirm the memory service is healthy
- confirm `BYTE_INTERNAL_TOKEN` matches on gateway and memory service

### Artifact import or export fails

- confirm `security.exportRoot`
- confirm the target path stays inside the configured export root

### Metrics are inaccessible

- confirm whether admin auth is enabled for the gateway
- confirm the ingress policy and service exposure for `/metrics`

## Rotation guidance

- rotate `BYTE_INTERNAL_TOKEN` across gateway, memory, and workers together
- rotate `BYTE_ADMIN_TOKEN` independently of provider credentials
- rotate provider secrets independently of the admin boundary
