"""Manifest apply and reconciliation helpers for the Byte operator."""

from __future__ import annotations

from typing import Any

from byte_server._operator_manifest import (
    build_deployment,
    build_inference_resources,
    build_memory_resources,
    build_persistent_volume_claim,
    build_service,
)


def _load_cluster_config(config_module: Any) -> None:
    """Load in-cluster config first and fall back to the local kube config."""
    config_error = config_module.config_exception.ConfigException
    try:  # pragma: no cover
        config_module.load_incluster_config()
    except config_error:  # pragma: no cover
        config_module.load_kube_config()


def _apply_namespaced_object(
    api: Any,
    *,
    read_method: str,
    create_method: str,
    patch_method: str,
    name: str,
    namespace: str,
    body: dict[str, Any],
    api_exception_type: type[Exception],
) -> str:
    """Create or patch a namespaced resource, creating only on a typed 404 read miss."""
    try:
        getattr(api, read_method)(name=name, namespace=namespace)
    except api_exception_type as exc:  # pragma: no cover
        if getattr(exc, "status", None) != 404:
            raise RuntimeError(
                f"Failed to read Kubernetes resource {name!r} via {read_method}."
            ) from exc
        try:
            getattr(api, create_method)(namespace=namespace, body=body)
        except api_exception_type as create_exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to create Kubernetes resource {name!r} via {create_method}."
            ) from create_exc
        return "created"

    try:
        getattr(api, patch_method)(name=name, namespace=namespace, body=body)
    except api_exception_type as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to patch Kubernetes resource {name!r} via {patch_method}."
        ) from exc
    return "patched"


def _manifest_binding(client_module: Any, kind: str) -> tuple[Any, str, str, str]:
    if kind == "Deployment":
        return (
            client_module.AppsV1Api(),
            "read_namespaced_deployment",
            "create_namespaced_deployment",
            "patch_namespaced_deployment",
        )
    if kind == "StatefulSet":
        return (
            client_module.AppsV1Api(),
            "read_namespaced_stateful_set",
            "create_namespaced_stateful_set",
            "patch_namespaced_stateful_set",
        )
    if kind == "Service":
        return (
            client_module.CoreV1Api(),
            "read_namespaced_service",
            "create_namespaced_service",
            "patch_namespaced_service",
        )
    if kind == "PersistentVolumeClaim":
        return (
            client_module.CoreV1Api(),
            "read_namespaced_persistent_volume_claim",
            "create_namespaced_persistent_volume_claim",
            "patch_namespaced_persistent_volume_claim",
        )
    raise RuntimeError(f"Unsupported manifest kind: {kind}")


def _apply_manifest(
    namespace: str,
    manifest: dict[str, Any],
    *,
    client_module: Any,
    api_exception_type: type[Exception],
) -> tuple[str, str]:
    kind = str(manifest.get("kind", "") or "")
    name = str(((manifest.get("metadata") or {}).get("name", "")) or "")
    api, read_method, create_method, patch_method = _manifest_binding(client_module, kind)
    result = _apply_namespaced_object(
        api,
        read_method=read_method,
        create_method=create_method,
        patch_method=patch_method,
        name=name,
        namespace=namespace,
        body=manifest,
        api_exception_type=api_exception_type,
    )
    return (name, result)


def reconcile_resource(
    name: str, namespace: str, spec: dict[str, Any], body: dict[str, Any]
) -> dict[str, str]:
    """Apply all manifests needed for a ByteCache resource and return per-object actions."""
    try:
        from kubernetes import client, config
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Kubernetes operator support requires the optional operator dependencies. "
            "Install the package with byte[operator] before running byte_operator."
        ) from exc

    _load_cluster_config(config)

    api_exception_type = client.exceptions.ApiException
    results: dict[str, str] = {}
    base_manifests = {
        "deployment": build_deployment(name, namespace, spec, body),
        "service": build_service(name, namespace, spec, body),
    }
    pvc = build_persistent_volume_claim(name, namespace, spec, body)
    if pvc is not None:
        base_manifests["persistentVolumeClaim"] = pvc

    for result_key, manifest in base_manifests.items():
        _, action = _apply_manifest(
            namespace,
            manifest,
            client_module=client,
            api_exception_type=api_exception_type,
        )
        results[result_key] = action
    for manifest in build_memory_resources(name, namespace, spec, body):
        resource_name, action = _apply_manifest(
            namespace,
            manifest,
            client_module=client,
            api_exception_type=api_exception_type,
        )
        results[resource_name] = action
    for manifest in build_inference_resources(name, namespace, spec, body):
        resource_name, action = _apply_manifest(
            namespace,
            manifest,
            client_module=client,
            api_exception_type=api_exception_type,
        )
        results[resource_name] = action
    return results
