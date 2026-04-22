from byte import Cache, Config
from byte.adapter.api import (
    note_session_delta,
    recall_artifact,
    recall_retrieval_result,
    remember_artifact,
    remember_retrieval_result,
    remember_workflow_plan,
    run_retrieval,
    workflow_plan_hint,
)
from byte.manager.factory import get_data_manager
from byte.processor.batching import batch_embed
from byte.processor.optimization_memory import (
    ArtifactMemoryStore,
    PromptPieceStore,
    SessionDeltaStore,
    WorkflowPlanStore,
    distill_artifact_for_focus,
    extract_prompt_pieces,
)
from byte.processor.pre import compile_request_context


def _make_cache(tmp_path, config=None) -> object:
    cache_obj = Cache()
    cache_obj.init(
        data_manager=get_data_manager(data_path=str(tmp_path / "memory.txt")),
        config=config or Config(enable_token_counter=False),
    )
    return cache_obj


def test_prompt_piece_store_tracks_reuse() -> None:
    store = PromptPieceStore(max_entries=8, codec_name="qjl", bits=8)
    pieces = extract_prompt_pieces(
        {
            "messages": [
                {"role": "system", "content": "You are a support classifier."},
                {"role": "user", "content": "Classify this ticket."},
            ],
            "byte_prompt_pieces": ["labels: BILLING, TECHNICAL, SHIPPING"],
        }
    )
    first = store.remember_many(pieces)
    second = store.remember_many(pieces)

    assert len(first) == len(second) >= 2
    assert store.stats()["hits"] >= 1
    assert store.stats()["compressed_entries"] >= 1
    assert first[0]["compression"]["codec"] == "qjl"


def test_artifact_memory_store_can_recall_precomputed_summary() -> None:
    store = ArtifactMemoryStore(max_entries=4, codec_name="turboquant", bits=8)
    entry = store.remember(
        "retrieval_context",
        [{"title": "Billing FAQ", "snippet": "Refund duplicate charges within 5 business days."}],
    )
    recalled = store.get("retrieval_context", fingerprint=entry["fingerprint"])

    assert recalled is not None
    assert "Billing FAQ" in recalled["summary"]
    assert recalled["compression"]["codec"] == "turboquant"


def test_artifact_memory_store_can_find_related_summary_for_unique_prompt() -> None:
    store = ArtifactMemoryStore(max_entries=8, codec_name="turboquant", bits=8)
    store.remember(
        "retrieval_context",
        [{"title": "Billing FAQ", "snippet": "Refund duplicate charges within 5 business days."}],
        scope="support",
    )
    store.remember(
        "retrieval_context",
        [{"title": "Password Reset", "snippet": "Use the forgot password link to reset access."}],
        scope="support",
    )

    related = store.find_related(
        "retrieval_context",
        query_text="How long do refunds for duplicate subscription charges take?",
        scope="support",
        top_k=1,
    )

    assert len(related) == 1
    assert "Billing FAQ" in related[0]["summary"]
    assert related[0]["related_score"] > 0.18
    assert store.stats()["compressed_entries"] >= 1


def test_workflow_plan_store_prefers_successful_plan() -> None:
    store = WorkflowPlanStore(max_entries=4)
    request = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Fix the bug.\n```python\nprint('x')\n```"}],
    }
    store.remember(request, action="tool_first", success=False)
    store.remember(request, action="direct_expensive", success=True)

    hint = store.hint(request)

    assert hint["workflow_available"] is True
    assert hint["preferred_action"] == "direct_expensive"
    assert hint["prefer_expensive"] is True


def test_workflow_plan_store_returns_counterfactual_hint_for_failed_action() -> None:
    store = WorkflowPlanStore(max_entries=4)
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Extract the fields: name, city. Return JSON only."}
        ],
    }
    store.remember(
        request,
        action="cheap_then_verify",
        counterfactual_action="direct_expensive",
        counterfactual_reason="cheap_consensus_disagreement",
        success=False,
    )

    hint = store.hint(request)

    assert hint["workflow_available"] is True
    assert hint["avoid_action"] == "cheap_then_verify"
    assert hint["counterfactual_action"] == "direct_expensive"


def test_session_delta_store_reports_unchanged_context() -> None:
    store = SessionDeltaStore(max_entries=4)
    first = store.note("session-a", "repo_snapshot", {"files": ["a.py", "b.py"]})
    second = store.note("session-a", "repo_snapshot", {"files": ["a.py", "b.py"]})

    assert first["changed"] is True
    assert second["changed"] is False
    assert store.stats()["hits"] == 1


def test_compile_request_context_compacts_artifacts_and_uses_session_delta() -> None:
    artifact_store = ArtifactMemoryStore()
    session_store = SessionDeltaStore()
    request = {
        "messages": [
            {"role": "user", "content": "Answer the support request using the available context."}
        ],
        "byte_retrieval_context": [
            {
                "title": "Billing FAQ",
                "snippet": "Refund duplicate subscription charges in 5 business days.",
            },
            {
                "title": "Billing FAQ",
                "snippet": "Refund duplicate subscription charges in 5 business days.",
            },
        ],
        "byte_repo_snapshot": {"repo": "byte", "branch": "main", "files": ["a.py", "b.py", "c.py"]},
    }

    compiled_first, stats_first = compile_request_context(
        request,
        max_chars=1200,
        artifact_memory_store=artifact_store,
        session_delta_store=session_store,
        session_key="session-x",
        memory_scope="test",
    )
    compiled_second, stats_second = compile_request_context(
        request,
        max_chars=1200,
        artifact_memory_store=artifact_store,
        session_delta_store=session_store,
        session_key="session-x",
        memory_scope="test",
    )

    last_message = compiled_first["messages"][-1]["content"]
    assert "retrieval context" in last_message.lower()
    assert stats_first["compiled_aux_contexts"] >= 1
    assert stats_first["artifact_entries"] >= 1
    assert "unchanged" in compiled_second["messages"][-1]["content"].lower()
    assert stats_second["session_delta_hits"] >= 1


def test_compile_request_context_uses_related_artifact_memory_for_new_unique_prompt() -> None:
    artifact_store = ArtifactMemoryStore()
    artifact_store.remember(
        "retrieval_context",
        [
            {
                "title": "Refund Policy",
                "snippet": "Refund duplicate subscription charges in 5 business days.",
            }
        ],
        scope="support",
    )

    compiled, stats = compile_request_context(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Answer the refund question using the available context.",
                }
            ],
            "byte_retrieval_context": [
                {
                    "title": "Refund Questions",
                    "snippet": "Duplicate subscription charges are refunded in 5 business days.",
                },
                {"title": "Password Reset", "snippet": "Reset passwords from the settings page."},
            ],
        },
        max_chars=1200,
        artifact_memory_store=artifact_store,
        memory_scope="support",
        relevance_top_k=1,
        related_memory=True,
        related_min_score=0.18,
    )

    assert stats["related_artifact_hits"] >= 1
    assert "refund" in compiled["messages"][-1]["content"].lower()


def test_distill_artifact_for_focus_prefers_query_matched_facts() -> None:
    summary = distill_artifact_for_focus(
        "retrieval_context",
        [
            {
                "title": "Refund SLA",
                "snippet": "Duplicate subscription charges are refunded in 5 business days. Reference ID REF-22.",
            },
            {
                "title": "Password Reset",
                "snippet": "Use the forgot password link in settings to reset access.",
            },
        ],
        query_text="How long do duplicate subscription refunds take?",
        max_chars=220,
    )

    lowered = summary.lower()
    assert "refund sla" in lowered
    assert "5 business days" in lowered
    assert "password reset" not in lowered


def test_api_retrieval_and_artifact_helpers_use_shared_memory(tmp_path) -> object:
    cache_obj = _make_cache(tmp_path)
    remember_retrieval_result("refund policy", [{"title": "FAQ"}], cache_obj=cache_obj)
    assert recall_retrieval_result("refund policy", cache_obj=cache_obj) == [{"title": "FAQ"}]

    call_count = [0]

    def fake_retrieval(query) -> object:
        call_count[0] += 1
        return [{"title": query, "snippet": "cached"}]

    first, first_hit = run_retrieval("billing", fake_retrieval, cache_obj=cache_obj)
    second, second_hit = run_retrieval("billing", fake_retrieval, cache_obj=cache_obj)

    assert first == second
    assert first_hit is False
    assert second_hit is True
    assert call_count[0] == 1

    artifact = remember_artifact(
        "repo_summary", {"repo": "byte", "files": ["a.py"]}, cache_obj=cache_obj
    )
    recalled = recall_artifact(
        "repo_summary", fingerprint=artifact["fingerprint"], cache_obj=cache_obj
    )
    assert recalled is not None
    assert recalled["artifact_type"] == "repo_summary"


def test_api_workflow_and_session_helpers_use_cache_runtime(tmp_path) -> None:
    cache_obj = _make_cache(tmp_path)
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Fix the same bug again.\n```python\nprint('x')\n```"}
        ],
    }

    remember_workflow_plan(
        request,
        action="direct_expensive",
        repo_fingerprint="repo-a",
        success=True,
        cache_obj=cache_obj,
    )
    hint = workflow_plan_hint(request, repo_fingerprint="repo-a", cache_obj=cache_obj)
    delta_first = note_session_delta(
        "session-z", "repo_snapshot", {"files": ["a.py"]}, cache_obj=cache_obj
    )
    delta_second = note_session_delta(
        "session-z", "repo_snapshot", {"files": ["a.py"]}, cache_obj=cache_obj
    )

    assert hint["workflow_available"] is True
    assert hint["prefer_expensive"] is True
    assert delta_first["changed"] is True
    assert delta_second["changed"] is False


def test_memory_snapshot_includes_new_optimization_sections(tmp_path) -> None:
    cache_obj = _make_cache(tmp_path)
    cache_obj.remember_prompt_pieces([{"type": "system", "content": "You are Byte."}])
    cache_obj.remember_artifact("repo_summary", {"repo": "byte", "files": ["a.py"]})
    cache_obj.remember_workflow_plan(
        {"messages": [{"role": "user", "content": "Explain this."}]},
        action="cheap_then_verify",
        success=True,
    )
    cache_obj.note_session_delta("session-1", "repo_snapshot", {"files": ["a.py"]})

    snapshot = cache_obj.export_memory_snapshot()

    assert "prompt_pieces" in snapshot
    assert "artifact_memory" in snapshot
    assert "workflow_plans" in snapshot
    assert "session_deltas" in snapshot


def test_batch_embed_dedupes_repeated_inputs() -> object:
    calls = []

    def fake_embedding(values, extra_param=None) -> object:
        calls.append(list(values))
        return [[len(item), index] for index, item in enumerate(values)]

    embeddings = batch_embed(fake_embedding, ["alpha", "alpha", "beta"])

    assert calls == [["alpha", "beta"]]
    assert embeddings[0] == embeddings[1]
    assert embeddings[2] != embeddings[0]
