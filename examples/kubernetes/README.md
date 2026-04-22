# Byte on Kubernetes

This folder contains the operator path for running Byte as a public gateway with optional private memory and inference services.

## Files

- `bytecache-crd.yaml`: `ByteCache` custom resource definition
- `bytecache-rbac.yaml`: service account, cluster role, and binding for the operator
- `bytecache-operator.yaml`: operator deployment
- `bytecache-sample.yaml`: example `ByteCache` resource for the full topology
- `otel-collector-datadog.yaml`: example OpenTelemetry Collector resource that forwards to Datadog

## Typical flow

1. Build and push the runtime image from `byte_server/dockerfiles/Dockerfile`.
2. Build and push the operator image from `byte_server/dockerfiles/Dockerfile.operator`.
3. Create the gateway and internal secrets.
4. Apply the CRD and RBAC.
5. Deploy the operator.
6. Optionally install the OpenTelemetry Operator, then apply `otel-collector-datadog.yaml`.
7. Apply `bytecache-sample.yaml`, adjusting image names, namespaces, and secret names first.

## Secret layout used by the sample

- `byte-gateway`: `admin-token`, `backend-api-key`
- `byte-internal`: `token`
