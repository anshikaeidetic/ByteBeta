from types import SimpleNamespace

import pytest

from byte import __version__
from byte_server._operator_reconcile import _apply_namespaced_object, _load_cluster_config
from byte_server.operator import (
    build_deployment,
    build_inference_resources,
    build_memory_resources,
    build_persistent_volume_claim,
    build_server_args,
    build_service,
)


def _body(name="byteai-cache") -> object:
    return {
        "apiVersion": "byte.ai/v1alpha1",
        "kind": "ByteCache",
        "metadata": {
            "name": name,
            "namespace": "default",
            "uid": "abc123",
        },
    }


def test_build_server_args_include_gateway_and_observability_flags() -> None:
    args = build_server_args(
        {
            "port": 8000,
            "cacheDir": "/cache",
            "gateway": {
                "enabled": True,
                "cacheMode": "normalized",
                "mode": "adaptive",
                "enableRouting": True,
                "cheapRouteTarget": "openai/gpt-4o-mini",
            },
            "observability": {
                "otelEnabled": True,
                "endpoint": "http://otel-collector:4318",
                "protocol": "http/protobuf",
                "environment": "prod",
                "resourceAttributes": {"team": "platform"},
                "datadogEnabled": True,
                "datadogAgentHost": "dd-agent",
                "datadogService": "byteai-cache",
                "datadogEnv": "prod",
                "datadogVersion": __version__,
            },
            "security": {
                "mode": True,
                "requireHttps": True,
                "rateLimitPublicPerMinute": 180,
                "maxInflightPublic": 32,
            },
            "extraArgs": ["--security-mode"],
        }
    )

    assert "--gateway" in args
    assert "--gateway-enable-routing" in args
    assert "--gateway-mode" in args
    assert "--otel-enabled" in args
    assert "--otel-endpoint" in args
    assert "--datadog-enabled" in args
    assert "--security-mode" in args
    assert "--security-require-https" in args
    assert "--security-rate-limit-public-per-minute" in args
    assert "--security-max-inflight-public" in args


def test_build_deployment_mounts_persistent_cache_volume() -> None:
    spec = {
        "image": "byteai-cache:latest",
        "cacheDir": "/var/lib/byteai/cache",
        "persistence": {"enabled": True, "size": "20Gi"},
    }

    deployment = build_deployment("byteai-cache", "default", spec, _body())
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert any(mount["mountPath"] == "/var/lib/byteai/cache" for mount in container["volumeMounts"])
    assert any(
        volume["name"] == "byteai-cache-storage"
        for volume in deployment["spec"]["template"]["spec"]["volumes"]
    )


def test_build_service_and_pvc_follow_resource_name() -> None:
    spec = {
        "port": 9000,
        "serviceType": "LoadBalancer",
        "persistence": {"enabled": True, "size": "5Gi"},
    }

    service = build_service("byteai-cache", "default", spec, _body())
    pvc = build_persistent_volume_claim("byteai-cache", "default", spec, _body())

    assert service["metadata"]["name"] == "byteai-cache"
    assert service["spec"]["type"] == "LoadBalancer"
    assert service["spec"]["ports"][0]["port"] == 9000
    assert pvc["metadata"]["name"] == "byteai-cache-cache"
    assert pvc["spec"]["resources"]["requests"]["storage"] == "5Gi"


def test_operator_propagates_internal_auth_and_gateway_only_env() -> None:
    spec = {
        "envFrom": [{"secretRef": {"name": "shared-config"}}],
        "internalAuthSecretRef": {"name": "byte-internal", "key": "token"},
        "gateway": {
            "enabled": True,
            "env": [
                {
                    "name": "BYTE_ADMIN_TOKEN",
                    "valueFrom": {"secretKeyRef": {"name": "byte-gateway", "key": "admin-token"}},
                }
            ],
            "envFrom": [{"secretRef": {"name": "gateway-only"}}],
        },
        "memory": {"enabled": True, "env": [{"name": "BYTE_MEMORY_FLAG", "value": "1"}]},
        "inferencePools": [
            {
                "name": "byteai-cache-inference",
                "models": ["huggingface/*"],
                "env": [{"name": "BYTE_INFERENCE_FLAG", "value": "1"}],
            }
        ],
    }

    deployment = build_deployment("byteai-cache", "default", spec, _body())
    memory_resources = build_memory_resources("byteai-cache", "default", spec, _body())
    inference_resources = build_inference_resources("byteai-cache", "default", spec, _body())

    gateway_container = deployment["spec"]["template"]["spec"]["containers"][0]
    memory_container = next(item for item in memory_resources if item["kind"] == "Deployment")[
        "spec"
    ]["template"]["spec"]["containers"][0]
    inference_container = next(
        item for item in inference_resources if item["kind"] == "StatefulSet"
    )["spec"]["template"]["spec"]["containers"][0]

    gateway_env_names = {item["name"] for item in gateway_container["env"]}
    memory_env_names = {item["name"] for item in memory_container["env"]}
    inference_env_names = {item["name"] for item in inference_container["env"]}

    assert "BYTE_ADMIN_TOKEN" in gateway_env_names
    assert "BYTE_INTERNAL_TOKEN" in gateway_env_names
    assert "BYTE_INTERNAL_TOKEN" in memory_env_names
    assert "BYTE_INTERNAL_TOKEN" in inference_env_names
    assert "BYTE_ADMIN_TOKEN" not in memory_env_names
    assert "BYTE_ADMIN_TOKEN" not in inference_env_names
    assert any(item["secretRef"]["name"] == "gateway-only" for item in gateway_container["envFrom"])
    assert any(item["secretRef"]["name"] == "shared-config" for item in memory_container["envFrom"])
    assert memory_container["readinessProbe"]["httpGet"]["path"] == "/healthz"
    assert inference_container["livenessProbe"]["httpGet"]["path"] == "/healthz"


class _FakeApiException(Exception):
    def __init__(self, status) -> None:
        super().__init__(f"status={status}")
        self.status = status


class _FakeApi:
    def __init__(self, read_error=None) -> None:
        self.read_error = read_error
        self.calls = []

    def read_namespaced_service(self, *, name, namespace) -> None:
        self.calls.append(("read", name, namespace))
        if self.read_error is not None:
            raise self.read_error

    def create_namespaced_service(self, *, namespace, body) -> None:
        self.calls.append(("create", namespace, body["metadata"]["name"]))

    def patch_namespaced_service(self, *, name, namespace, body) -> None:
        self.calls.append(("patch", name, namespace, body["metadata"]["name"]))


def test_apply_namespaced_object_creates_on_not_found() -> None:
    api = _FakeApi(read_error=_FakeApiException(status=404))
    action = _apply_namespaced_object(
        api,
        read_method="read_namespaced_service",
        create_method="create_namespaced_service",
        patch_method="patch_namespaced_service",
        name="byteai-cache",
        namespace="default",
        body={"metadata": {"name": "byteai-cache"}},
        api_exception_type=_FakeApiException,
    )

    assert action == "created"
    assert ("create", "default", "byteai-cache") in api.calls
    assert not any(call[0] == "patch" for call in api.calls)


def test_apply_namespaced_object_rejects_non_not_found_read_errors() -> None:
    api = _FakeApi(read_error=_FakeApiException(status=500))

    with pytest.raises(RuntimeError, match="Failed to read Kubernetes resource"):
        _apply_namespaced_object(
            api,
            read_method="read_namespaced_service",
            create_method="create_namespaced_service",
            patch_method="patch_namespaced_service",
            name="byteai-cache",
            namespace="default",
            body={"metadata": {"name": "byteai-cache"}},
            api_exception_type=_FakeApiException,
        )

    assert not any(call[0] in {"create", "patch"} for call in api.calls)


def test_load_cluster_config_falls_back_to_local_kube_config() -> None:
    state = {"incluster": 0, "kube": 0}

    class _FakeConfigException(Exception):
        pass

    config_module = SimpleNamespace(
        config_exception=SimpleNamespace(ConfigException=_FakeConfigException),
        load_incluster_config=lambda: (_ for _ in ()).throw(_FakeConfigException("missing")),
        load_kube_config=lambda: state.__setitem__("kube", state["kube"] + 1),
    )

    _load_cluster_config(config_module)

    assert state["kube"] == 1
