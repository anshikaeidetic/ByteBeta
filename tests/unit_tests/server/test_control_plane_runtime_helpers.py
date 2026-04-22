from __future__ import annotations

from unittest.mock import patch

from byte_server._control_plane import (
    ControlPlaneRuntime,
    RequestScope,
    apply_memory_resolution,
    provider_mode_for_request,
)
from byte_server._control_plane_routing import WorkerSelection
from byte_server._control_plane_scope import request_text, response_text


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.content = b"{}"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return dict(self._payload)


def test_control_plane_runtime_rehydrates_existing_settings_and_normalizes_updates(tmp_path) -> None:
    db_path = tmp_path / "control-plane.db"
    runtime = ControlPlaneRuntime(
        db_path=str(db_path),
        worker_urls=["http://worker.one:8090/"],
        memory_service_url="http://memory.one:8091/",
        internal_auth_token=" token-one ",
        replay_enabled=True,
        replay_sample_rate=1.5,
    )
    rehydrated = ControlPlaneRuntime(db_path=str(db_path))

    assert rehydrated.feature_flags["worker_urls"] == ["http://worker.one:8090"]
    assert rehydrated.feature_flags["memory_service_url"] == "http://memory.one:8091"
    assert rehydrated.inspect()["internal_auth_configured"] is True

    updated = rehydrated.update_feature_flags(
        {
            "worker_urls": "invalid",
            "memory_service_url": " http://memory.two:8091/ ",
            "internal_auth_token": " token-two ",
            "replay_enabled": 1,
            "replay_sample_rate": "0.25",
        }
    )

    assert updated["worker_urls"] == []
    assert updated["memory_service_url"] == "http://memory.two:8091"
    assert updated["internal_auth_token"] == "token-two"
    assert updated["replay_enabled"] is True
    assert rehydrated._feature_float("replay_sample_rate") == 0.25
    assert rehydrated._internal_headers() == {"X-Byte-Internal-Token": "token-two"}


def test_control_plane_scope_and_text_helpers_cover_header_and_derived_paths(tmp_path) -> None:
    runtime = ControlPlaneRuntime(db_path=str(tmp_path / "control-plane.db"))

    derived_scope = runtime.extract_scope(
        {},
        {
            "user": "acme",
            "messages": [{"role": "user", "content": [{"text": "deploy"}, {"content": "plan"}]}],
        },
    )
    header_scope = runtime.extract_scope(
        {
            "x-byte-tenant": "acme",
            "x-byte-session": "session-1",
            "x-byte-workflow": "ops",
        },
        {"messages": [{"role": "user", "content": "hello"}]},
    )

    assert derived_scope.tenant == "acme"
    assert derived_scope.source == "derived"
    assert derived_scope.scope_key.startswith("acme:")
    assert header_scope.session == "session-1"
    assert header_scope.workflow == "ops"
    assert header_scope.source == "headers"

    assert request_text({"prompt": " hello "}) == "hello"
    assert request_text({"input": " plan "}) == "plan"
    assert response_text({"choices": [{"text": "done"}]}) == "done"
    assert response_text({"answer": "stored"}) == "stored"


def test_control_plane_runtime_records_intent_edges_and_handles_missing_selection_inputs(tmp_path) -> None:
    runtime = ControlPlaneRuntime(db_path=str(tmp_path / "control-plane.db"))
    scope = RequestScope(tenant="acme", session="s1", workflow="chat")

    runtime.record_intent(scope=scope, route_key="")
    runtime.record_intent(scope=scope, route_key="deploy")
    runtime.record_intent(scope=scope, route_key="review")

    snapshot = runtime.store.intent_graph_snapshot(limit=10)
    assert snapshot["edges"][0]["source_intent"] == "deploy"
    assert snapshot["edges"][0]["target_intent"] == "review"
    assert runtime.maybe_select_worker(scope=scope, request_payload={"messages": []}) is None


def test_control_plane_runtime_replay_flow_updates_recommendations(tmp_path) -> None:
    runtime = ControlPlaneRuntime(
        db_path=str(tmp_path / "control-plane.db"),
        replay_enabled=True,
        replay_sample_rate=1.0,
    )
    scope = RequestScope(tenant="acme", session="s1", workflow="chat")
    request_payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Summarize the deployment plan."}],
    }
    response_payload = {
        "choices": [{"message": {"role": "assistant", "content": "Deploy with blue-green."}}]
    }

    with patch("byte_server._control_plane_runtime.random.random", return_value=0.0):
        result = runtime.maybe_schedule_replay(
            scope=scope,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    assert result is not None
    assert result["job"]["route_key"]
    assert result["outcome"]["projected_savings"] > 0
    recommendations = runtime.inspect()["recommendations"]
    assert recommendations
    assert recommendations[0]["sample_count"] >= 1


def test_control_plane_runtime_dispatch_and_memory_posts_include_internal_headers(tmp_path) -> object:
    runtime = ControlPlaneRuntime(
        db_path=str(tmp_path / "control-plane.db"),
        memory_service_url="http://memory.example:8091",
        internal_auth_token="secret-token",
    )
    scope = RequestScope(tenant="acme", session="s1", workflow="chat")
    worker = WorkerSelection(
        worker_id="worker-1",
        url="http://worker.example:8090",
        source="health",
        model_name="gpt-4o-mini",
        score=1.0,
    )
    seen_headers: list[dict[str, str]] = []

    def _fake_post(url, headers=None, json=None, timeout=None) -> object:
        seen_headers.append(dict(headers or {}))
        if url.endswith("/generate"):
            return _Response({"choices": [{"message": {"content": "worker result"}}]})
        return _Response({"status": "ok"})

    with patch("byte_server._control_plane_runtime.requests.post", side_effect=_fake_post):
        runtime.remember_memory(
            scope=scope,
            request_payload={"model": "gpt-4o-mini"},
            response_payload={"choices": [{"message": {"content": "stored"}}]},
            provider_mode="hosted",
            worker_id="worker-1",
        )
        payload = runtime.dispatch_to_worker(
            worker=worker,
            scope=scope,
            request_payload={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert payload["choices"][0]["message"]["content"] == "worker result"
    assert seen_headers
    assert all(item["X-Byte-Internal-Token"] == "secret-token" for item in seen_headers)


def test_control_plane_resolution_helpers_merge_context_and_choose_provider_mode() -> None:
    request_payload = {"messages": [{"role": "user", "content": "hello"}]}
    resolution = {
        "context": {
            "byte_tool_result_context": ["tool-a"],
            "byte_repo_summary": "repo-summary",
            "byte_retrieval_context": "retrieval-a",
        }
    }
    merged = apply_memory_resolution(
        {
            **request_payload,
            "byte_tool_result_context": ["tool-b"],
            "byte_retrieval_context": "retrieval-b",
        },
        resolution,
    )

    assert merged["byte_tool_result_context"] == ["tool-b", "tool-a"]
    assert merged["byte_repo_summary"] == "repo-summary"
    assert merged["byte_retrieval_context"] == "retrieval-b\nretrieval-a"
    assert provider_mode_for_request({"model": "huggingface/llama"}, worker_selected=False) == "local"
    assert provider_mode_for_request({"model": "gpt-4o-mini"}, worker_selected=True) == "local"
    assert provider_mode_for_request({"model": "gpt-4o-mini"}, worker_selected=False) == "hosted"
