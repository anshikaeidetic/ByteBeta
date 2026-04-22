"""Kubernetes manifest builders for the Byte operator."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

GROUP = "byte.ai"
VERSION = "v1alpha1"
PLURAL = "bytecaches"
KIND = "ByteCache"


def _labels(name: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "byteai-cache",
        "app.kubernetes.io/instance": name,
        "app.kubernetes.io/component": "cache-server",
        "app.kubernetes.io/managed-by": "byteai-operator",
    }


def _owner_reference(body: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = body.get("metadata", {}) or {}
    return [
        {
            "apiVersion": body.get("apiVersion", f"{GROUP}/{VERSION}"),
            "kind": body.get("kind", KIND),
            "name": metadata.get("name", ""),
            "uid": metadata.get("uid", ""),
            "controller": True,
            "blockOwnerDeletion": True,
        }
    ]


def _merge_labels(name: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    merged = _labels(name)
    for key, value in (extra or {}).items():
        if key:
            merged[str(key)] = str(value)
    return merged


def _internal_auth_env(spec: dict[str, Any]) -> list[dict[str, Any]]:
    secret_ref = spec.get("internalAuthSecretRef", {}) or {}
    name = str(secret_ref.get("name", "") or "").strip()
    key = str(secret_ref.get("key", "token") or "token").strip()
    if not name:
        return []
    return [
        {
            "name": "BYTE_INTERNAL_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": name, "key": key}},
        }
    ]


def _component_env(spec: dict[str, Any], component: str) -> list[dict[str, Any]]:
    component_spec = spec.get(component, {}) or {}
    if not isinstance(component_spec, dict):
        component_spec = {}
    env = deepcopy(spec.get("env", []) or [])
    env.extend(deepcopy(component_spec.get("env", []) or []))
    env.extend(_internal_auth_env(spec))
    return env


def _component_env_from(
    spec: dict[str, Any],
    component: str,
    *,
    include_gateway_only: bool = False,
) -> list[dict[str, Any]]:
    component_spec = spec.get(component, {}) or {}
    if not isinstance(component_spec, dict):
        component_spec = {}
    env_from = deepcopy(spec.get("envFrom", []) or [])
    env_from.extend(deepcopy(component_spec.get("envFrom", []) or []))
    if include_gateway_only:
        env_from.extend(deepcopy((spec.get("gateway", {}) or {}).get("envFrom", []) or []))
    return env_from


def _gateway_args(spec: dict[str, Any]) -> list[str]:
    gateway = spec.get("gateway", {}) or {}
    if not gateway.get("enabled"):
        return []

    args = ["--gateway", "True"]
    cache_mode = str(gateway.get("cacheMode", "normalized") or "normalized")
    args.extend(["--gateway-cache-mode", cache_mode])
    gateway_mode = str(gateway.get("mode", "backend") or "backend")
    args.extend(["--gateway-mode", gateway_mode])
    if gateway.get("enableRouting"):
        args.append("--gateway-enable-routing")
    for field_name, arg_name in (
        ("cheapRouteTarget", "--gateway-cheap-route-target"),
        ("expensiveRouteTarget", "--gateway-expensive-route-target"),
        ("toolRouteTarget", "--gateway-tool-route-target"),
        ("defaultRouteTarget", "--gateway-default-route-target"),
        ("coderRouteTarget", "--gateway-coder-route-target"),
        ("reasoningRouteTarget", "--gateway-reasoning-route-target"),
        ("verifierRouteTarget", "--gateway-verifier-route-target"),
    ):
        value = gateway.get(field_name)
        if value not in (None, ""):
            args.extend([arg_name, str(value)])
    return args


def _observability_args(spec: dict[str, Any]) -> list[str]:
    observability = spec.get("observability", {}) or {}
    args: list[str] = []
    if observability.get("otelEnabled"):
        args.append("--otel-enabled")
    if observability.get("endpoint"):
        args.extend(["--otel-endpoint", str(observability["endpoint"])])
    if observability.get("protocol"):
        args.extend(["--otel-protocol", str(observability["protocol"])])
    if observability.get("headers"):
        header_values = observability["headers"]
        if isinstance(header_values, dict):
            encoded_headers = ",".join(f"{key}={value}" for key, value in header_values.items())
        else:
            encoded_headers = str(header_values)
        if encoded_headers:
            args.extend(["--otel-headers", encoded_headers])
    if observability.get("insecure"):
        args.append("--otel-insecure")
    if observability.get("disableTraces"):
        args.append("--otel-disable-traces")
    if observability.get("disableMetrics"):
        args.append("--otel-disable-metrics")
    if observability.get("exportIntervalMs") not in (None, ""):
        args.extend(["--otel-export-interval-ms", str(observability["exportIntervalMs"])])
    if observability.get("serviceNamespace"):
        args.extend(["--otel-service-namespace", str(observability["serviceNamespace"])])
    if observability.get("environment"):
        args.extend(["--otel-environment", str(observability["environment"])])
    if observability.get("resourceAttributes"):
        resource_attributes = observability["resourceAttributes"]
        if isinstance(resource_attributes, dict):
            encoded_attributes = ",".join(
                f"{key}={value}" for key, value in resource_attributes.items()
            )
        else:
            encoded_attributes = str(resource_attributes)
        if encoded_attributes:
            args.extend(["--otel-resource-attributes", encoded_attributes])
    if observability.get("datadogEnabled"):
        args.append("--datadog-enabled")
        if observability.get("datadogAgentHost"):
            args.extend(["--datadog-agent-host", str(observability["datadogAgentHost"])])
        if observability.get("datadogService"):
            args.extend(["--datadog-service", str(observability["datadogService"])])
        if observability.get("datadogEnv"):
            args.extend(["--datadog-env", str(observability["datadogEnv"])])
        if observability.get("datadogVersion"):
            args.extend(["--datadog-version", str(observability["datadogVersion"])])
    return args


def _control_plane_args(spec: dict[str, Any]) -> list[str]:
    control_plane = spec.get("controlPlane", {}) or {}
    args: list[str] = []
    if control_plane.get("dbPath"):
        args.extend(["--control-plane-db", str(control_plane["dbPath"])])
    if control_plane.get("memoryServiceUrl"):
        args.extend(["--memory-service-url", str(control_plane["memoryServiceUrl"])])
    if control_plane.get("replayEnabled"):
        args.append("--replay-enabled")
    if control_plane.get("replaySampleRate") not in (None, ""):
        args.extend(["--replay-sample-rate", str(control_plane["replaySampleRate"])])
    for worker_url in control_plane.get("workerUrls", []) or []:
        if worker_url not in (None, ""):
            args.extend(["--control-plane-worker-url", str(worker_url)])
    return args


def _security_args(spec: dict[str, Any]) -> list[str]:
    security = spec.get("security", {}) or {}
    args: list[str] = []
    if security.get("mode"):
        args.append("--security-mode")
    for field_name, arg_name in (
        ("adminToken", "--security-admin-token"),
        ("auditLogPath", "--security-audit-log"),
        ("exportRoot", "--security-export-root"),
        ("allowedEgressHosts", "--security-allowed-egress-hosts"),
    ):
        value = security.get(field_name)
        if value not in (None, ""):
            args.extend([arg_name, str(value)])
    for field_name, arg_name in (
        ("requireHttps", "--security-require-https"),
        ("trustProxyHeaders", "--security-trust-proxy-headers"),
        ("allowProviderHostOverride", "--security-allow-provider-host-override"),
    ):
        if security.get(field_name):
            args.append(arg_name)
    for field_name, arg_name in (
        ("maxRequestBytes", "--security-max-request-bytes"),
        ("maxUploadBytes", "--security-max-upload-bytes"),
        ("maxArchiveBytes", "--security-max-archive-bytes"),
        ("maxArchiveMembers", "--security-max-archive-members"),
        ("rateLimitPublicPerMinute", "--security-rate-limit-public-per-minute"),
        ("rateLimitAdminPerMinute", "--security-rate-limit-admin-per-minute"),
        ("maxInflightPublic", "--security-max-inflight-public"),
        ("maxInflightAdmin", "--security-max-inflight-admin"),
    ):
        value = security.get(field_name)
        if value not in (None, ""):
            args.extend([arg_name, str(value)])
    return args


def build_server_args(spec: dict[str, Any]) -> list[str]:
    port = int(spec.get("port", 8000) or 8000)
    cache_dir = str(spec.get("cacheDir", "/var/lib/byteai/cache") or "/var/lib/byteai/cache")
    args = ["-s", "0.0.0.0", "-p", str(port), "-d", cache_dir]
    args.extend(_gateway_args(spec))
    args.extend(_observability_args(spec))
    args.extend(_control_plane_args(spec))
    args.extend(_security_args(spec))
    for flag in spec.get("extraArgs", []) or []:
        if flag not in (None, ""):
            args.append(str(flag))
    return args


def build_persistent_volume_claim(
    name: str,
    namespace: str,
    spec: dict[str, Any],
    body: dict[str, Any],
) -> dict[str, Any] | None:
    persistence = spec.get("persistence", {}) or {}
    if not persistence.get("enabled"):
        return None
    pvc_spec: dict[str, Any] = {
        "accessModes": list(persistence.get("accessModes", ["ReadWriteOnce"])),
        "resources": {"requests": {"storage": str(persistence.get("size", "10Gi"))}},
    }
    storage_class_name = persistence.get("storageClassName")
    if storage_class_name not in (None, ""):
        pvc_spec["storageClassName"] = str(storage_class_name)
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": f"{name}-cache",
            "namespace": namespace,
            "labels": _labels(name),
            "ownerReferences": _owner_reference(body),
        },
        "spec": pvc_spec,
    }


def build_deployment(
    name: str, namespace: str, spec: dict[str, Any], body: dict[str, Any]
) -> dict[str, Any]:
    labels = _merge_labels(name, spec.get("labels"))
    port = int(spec.get("port", 8000) or 8000)
    service_account_name = str(spec.get("serviceAccountName", "default") or "default")
    container = {
        "name": "byteai-cache",
        "image": str(spec.get("image", "byteai-cache:latest") or "byteai-cache:latest"),
        "imagePullPolicy": str(spec.get("imagePullPolicy", "IfNotPresent") or "IfNotPresent"),
        "ports": [{"containerPort": port, "name": "http"}],
        "args": build_server_args(spec),
        "env": _component_env(spec, "gateway"),
        "resources": deepcopy(spec.get("resources", {}) or {}),
        "readinessProbe": {
            "httpGet": {"path": "/readyz", "port": "http"},
            "initialDelaySeconds": 5,
            "periodSeconds": 10,
        },
        "livenessProbe": {
            "httpGet": {"path": "/healthz", "port": "http"},
            "initialDelaySeconds": 10,
            "periodSeconds": 15,
        },
        "volumeMounts": deepcopy(spec.get("volumeMounts", []) or []),
    }
    env_from = _component_env_from(spec, "gateway", include_gateway_only=True)
    if env_from:
        container["envFrom"] = env_from

    pod_spec: dict[str, Any] = {
        "serviceAccountName": service_account_name,
        "containers": [container],
        "volumes": deepcopy(spec.get("volumes", []) or []),
    }

    persistence = spec.get("persistence", {}) or {}
    if persistence.get("enabled"):
        mount_path = str(
            persistence.get("mountPath", spec.get("cacheDir", "/var/lib/byteai/cache"))
            or "/var/lib/byteai/cache"
        )
        pod_spec["volumes"].append(
            {
                "name": "byteai-cache-storage",
                "persistentVolumeClaim": {"claimName": f"{name}-cache"},
            }
        )
        container["volumeMounts"].append({"name": "byteai-cache-storage", "mountPath": mount_path})

    for field_name in (
        "nodeSelector",
        "affinity",
        "tolerations",
        "topologySpreadConstraints",
        "securityContext",
    ):
        value = spec.get(field_name)
        if value not in (None, {}, []):
            pod_spec[field_name] = deepcopy(value)

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": _owner_reference(body),
        },
        "spec": {
            "replicas": int(spec.get("replicas", 1) or 1),
            "selector": {"matchLabels": {"app.kubernetes.io/instance": name}},
            "template": {
                "metadata": {
                    "labels": labels,
                    "annotations": deepcopy(spec.get("podAnnotations", {}) or {}),
                },
                "spec": pod_spec,
            },
        },
    }
    strategy = spec.get("strategy")
    if strategy not in (None, {}):
        deployment["spec"]["strategy"] = deepcopy(strategy)
    return deployment


def build_service(
    name: str, namespace: str, spec: dict[str, Any], body: dict[str, Any]
) -> dict[str, Any]:
    port = int(spec.get("port", 8000) or 8000)
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": _merge_labels(name, spec.get("labels")),
            "ownerReferences": _owner_reference(body),
        },
        "spec": {
            "type": str(spec.get("serviceType", "ClusterIP") or "ClusterIP"),
            "selector": {"app.kubernetes.io/instance": name},
            "ports": [
                {
                    "name": "http",
                    "port": port,
                    "targetPort": "http",
                    "protocol": "TCP",
                }
            ],
        },
    }
    annotations = deepcopy(spec.get("serviceAnnotations", {}) or {})
    if annotations:
        service["metadata"]["annotations"] = annotations
    return service


def build_memory_resources(
    name: str, namespace: str, spec: dict[str, Any], body: dict[str, Any]
) -> list[dict[str, Any]]:
    memory = spec.get("memory", {}) or {}
    if not memory.get("enabled"):
        return []
    resource_name = f"{name}-memory"
    labels = _merge_labels(resource_name, spec.get("labels"))
    port = int(memory.get("port", 8091) or 8091)
    image = str(memory.get("image", spec.get("image", "byteai-cache:latest")) or "byteai-cache:latest")
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": resource_name,
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": _owner_reference(body),
        },
        "spec": {
            "replicas": int(memory.get("replicas", 1) or 1),
            "selector": {"matchLabels": {"app.kubernetes.io/instance": resource_name}},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "serviceAccountName": str(spec.get("serviceAccountName", "default") or "default"),
                    "containers": [
                        {
                            "name": "byte-memory",
                            "image": image,
                            "imagePullPolicy": str(spec.get("imagePullPolicy", "IfNotPresent") or "IfNotPresent"),
                            "ports": [{"containerPort": port, "name": "http"}],
                            "args": ["--host", "0.0.0.0", "--port", str(port)],
                            "env": _component_env(spec, "memory"),
                            "envFrom": _component_env_from(spec, "memory"),
                            "resources": deepcopy(memory.get("resources", {}) or {}),
                            "readinessProbe": {
                                "httpGet": {"path": "/healthz", "port": "http"},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/healthz", "port": "http"},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 15,
                            },
                        }
                    ],
                },
            },
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": resource_name,
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": _owner_reference(body),
        },
        "spec": {
            "type": "ClusterIP",
            "selector": {"app.kubernetes.io/instance": resource_name},
            "ports": [{"name": "http", "port": port, "targetPort": "http", "protocol": "TCP"}],
        },
    }
    return [deployment, service]


def build_inference_resources(
    name: str, namespace: str, spec: dict[str, Any], body: dict[str, Any]
) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for index, pool in enumerate(spec.get("inferencePools", []) or []):
        resource_name = str(pool.get("name", f"{name}-inference-{index}")).strip() or f"{name}-inference-{index}"
        labels = _merge_labels(resource_name, spec.get("labels"))
        port = int(pool.get("port", 8090) or 8090)
        image = str(pool.get("image", spec.get("image", "byteai-cache:latest")) or "byteai-cache:latest")
        statefulset = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": resource_name,
                "namespace": namespace,
                "labels": labels,
                "ownerReferences": _owner_reference(body),
            },
            "spec": {
                "serviceName": resource_name,
                "replicas": int(pool.get("replicas", 1) or 1),
                "selector": {"matchLabels": {"app.kubernetes.io/instance": resource_name}},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "serviceAccountName": str(spec.get("serviceAccountName", "default") or "default"),
                        "containers": [
                            {
                                "name": "byte-inference",
                                "image": image,
                                "imagePullPolicy": str(spec.get("imagePullPolicy", "IfNotPresent") or "IfNotPresent"),
                                "ports": [{"containerPort": port, "name": "http"}],
                                "args": [
                                    "--host",
                                    "0.0.0.0",
                                    "--port",
                                    str(port),
                                    "--worker-id",
                                    resource_name,
                                    "--models",
                                    ",".join(str(item) for item in (pool.get("models", []) or [])),
                                ],
                                "env": _component_env(spec, "inferencePools")
                                + deepcopy(pool.get("env", []) or []),
                                "envFrom": _component_env_from(spec, "inferencePools"),
                                "resources": deepcopy(pool.get("resources", {}) or {}),
                                "readinessProbe": {
                                    "httpGet": {"path": "/healthz", "port": "http"},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 10,
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/healthz", "port": "http"},
                                    "initialDelaySeconds": 10,
                                    "periodSeconds": 15,
                                },
                            }
                        ],
                    },
                },
            },
        }
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": resource_name,
                "namespace": namespace,
                "labels": labels,
                "ownerReferences": _owner_reference(body),
            },
            "spec": {
                "clusterIP": "None",
                "selector": {"app.kubernetes.io/instance": resource_name},
                "ports": [{"name": "http", "port": port, "targetPort": "http", "protocol": "TCP"}],
            },
        }
        resources.extend([statefulset, service])
    return resources
