from unittest.mock import patch

from fastapi.testclient import TestClient

from byte.cli import main as byte_main
from byte_inference import server as inference_server
from byte_memory import server as memory_server
from byte_server import operator


def test_memory_service_remember_then_resolve() -> None:
    memory_server.runtime = memory_server.MemoryServiceRuntime(max_entries=32)
    client = TestClient(memory_server.app)
    remember = client.post(
        "/internal/v1/remember",
        json={
            "scope": {"tenant": "acme", "session": "s1", "workflow": "chat"},
            "request": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Summarize the deploy plan"}],
            },
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "Deploy via blue green."}}]
            },
            "provider_mode": "hosted",
        },
    )
    assert remember.status_code == 200

    resolved = client.post(
        "/internal/v1/resolve",
        json={
            "scope": {"tenant": "acme", "session": "s1", "workflow": "chat"},
            "request": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Summarize the deploy plan"}],
            },
            "provider_mode": "hosted",
        },
    )
    payload = resolved.json()
    assert resolved.status_code == 200
    assert payload["context"]["byte_repo_summary"]
    assert payload["context"]["byte_retrieval_context"]


def test_memory_service_internal_auth_rejects_unsigned_requests() -> None:
    original_token = memory_server._internal_auth_token
    try:
        memory_server._internal_auth_token = "secret-token"
        memory_server.runtime = memory_server.MemoryServiceRuntime(max_entries=32)
        client = TestClient(memory_server.app)

        denied = client.post(
            "/internal/v1/resolve",
            json={
                "scope": {"tenant": "acme", "session": "s1", "workflow": "chat"},
                "request": {"messages": [{"role": "user", "content": "hello"}]},
                "provider_mode": "hosted",
            },
        )
        allowed = client.post(
            "/internal/v1/resolve",
            headers={"X-Byte-Internal-Token": "secret-token"},
            json={
                "scope": {"tenant": "acme", "session": "s1", "workflow": "chat"},
                "request": {"messages": [{"role": "user", "content": "hello"}]},
                "provider_mode": "hosted",
            },
        )
    finally:
        memory_server._internal_auth_token = original_token

    assert denied.status_code == 403
    assert allowed.status_code == 200


def test_inference_worker_generate_adds_worker_metadata(monkeypatch) -> None:
    inference_server.runtime = inference_server.InferenceWorkerRuntime(
        worker_id="worker-test",
        cache_dir="byte_worker_test_cache",
        model_inventory=["gpt-4o-mini"],
        free_vram_gb=24.0,
    )
    client = TestClient(inference_server.app)
    with patch.object(
        inference_server.byte_adapter.ChatCompletion,
        "create",
        return_value={
            "choices": [{"message": {"role": "assistant", "content": "hello from worker"}}]
        },
    ):
        response = client.post(
            "/internal/v1/generate",
            json={
                "scope": {"tenant": "acme", "session": "s1", "workflow": "chat"},
                "request": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "selection_source": "health_weighted",
            },
        )
    assert response.status_code == 200
    assert response.json()["byte_worker"]["worker_id"] == "worker-test"


def test_inference_worker_internal_auth_rejects_unsigned_requests() -> None:
    original_token = inference_server._internal_auth_token
    try:
        inference_server._internal_auth_token = "secret-token"
        inference_server.runtime = inference_server.InferenceWorkerRuntime(
            worker_id="worker-auth",
            cache_dir="byte_worker_test_cache",
        )
        client = TestClient(inference_server.app)
        denied = client.get("/internal/v1/runtime")
        allowed = client.get(
            "/internal/v1/runtime",
            headers={"X-Byte-Internal-Token": "secret-token"},
        )
    finally:
        inference_server._internal_auth_token = original_token

    assert denied.status_code == 403
    assert allowed.status_code == 200


def test_byte_cli_init_and_start(tmp_path, monkeypatch) -> object:
    config_path = tmp_path / "byteai.toml"
    monkeypatch.setattr("sys.argv", ["byte", "init", "--path", str(config_path)])
    byte_main()
    assert config_path.exists()

    captured = {}

    def _fake_call(command) -> object:
        captured["command"] = command
        return 0

    monkeypatch.setattr("subprocess.call", _fake_call)
    monkeypatch.setattr("sys.argv", ["byte", "start", "--config", str(config_path)])
    try:
        byte_main()
    except SystemExit as exc:
        assert exc.code == 0
    assert "--control-plane-db" in captured["command"]
    assert "byte_server.server" in captured["command"]


def test_operator_builds_memory_and_inference_resources() -> None:
    spec = {
        "image": "byteai-cache:latest",
        "labels": {"team": "platform"},
        "memory": {"enabled": True, "port": 8091},
        "inferencePools": [
            {"name": "byteai-infer", "replicas": 1, "port": 8090, "models": ["huggingface/*"]}
        ],
    }
    body = {"metadata": {"name": "byteai-cache", "uid": "abc"}, "kind": "ByteCache"}
    memory_resources = operator.build_memory_resources("byteai-cache", "default", spec, body)
    inference_resources = operator.build_inference_resources("byteai-cache", "default", spec, body)
    assert any(item["kind"] == "Deployment" for item in memory_resources)
    assert any(item["kind"] == "StatefulSet" for item in inference_resources)
