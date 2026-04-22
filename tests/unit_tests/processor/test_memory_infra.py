import time
import uuid

from byte import Cache, Config
from byte.adapter.api import (
    export_memory_snapshot,
    import_memory_snapshot,
    recall_tool_result,
    recent_interactions,
    remember_interaction,
    remember_tool_result,
    run_tool,
)
from byte.manager.factory import get_data_manager
from byte.processor.shared_memory import clear_shared_memory


def _make_cache(config=None) -> object:
    cache_obj = Cache()
    cache_obj.init(
        data_manager=get_data_manager(data_path=f"data_map_{uuid.uuid4().hex}.txt"),
        config=config or Config(enable_token_counter=False),
    )
    return cache_obj


def test_tool_result_store_respects_ttl() -> None:
    cache_obj = _make_cache(Config(enable_token_counter=False, tool_result_ttl=0.01))
    cache_obj.remember_tool_result("weather.lookup", {"city": "Paris"}, {"temp_c": 21})

    assert cache_obj.recall_tool_result("weather.lookup", {"city": "Paris"}) == {"temp_c": 21}
    time.sleep(0.02)
    assert cache_obj.recall_tool_result("weather.lookup", {"city": "Paris"}) is None
    assert cache_obj.tool_memory_stats()["expired"] == 1


def test_cache_run_tool_returns_cache_hit_on_second_call() -> object:
    cache_obj = _make_cache()
    call_count = [0]

    def fake_tool(city) -> object:
        call_count[0] += 1
        return {"city": city, "temp_c": 21}

    first, first_hit = cache_obj.run_tool("weather.lookup", {"city": "Paris"}, fake_tool)
    second, second_hit = cache_obj.run_tool("weather.lookup", {"city": "Paris"}, fake_tool)

    assert first == second == {"city": "Paris", "temp_c": 21}
    assert first_hit is False
    assert second_hit is True
    assert call_count[0] == 1


def test_api_tool_wrappers_use_shared_cache_tool_memory() -> object:
    cache_obj = _make_cache()
    remember_tool_result(
        "crm.lookup",
        {"email": "alice@example.com"},
        {"id": 7},
        cache_obj=cache_obj,
    )
    assert recall_tool_result(
        "crm.lookup",
        {"email": "alice@example.com"},
        cache_obj=cache_obj,
    ) == {"id": 7}

    call_count = [0]

    def fake_tool(email) -> object:
        call_count[0] += 1
        return {"email": email, "segment": "enterprise"}

    first, first_hit = run_tool(
        "segment.lookup",
        {"email": "alice@example.com"},
        fake_tool,
        cache_obj=cache_obj,
    )
    second, second_hit = run_tool(
        "segment.lookup",
        {"email": "alice@example.com"},
        fake_tool,
        cache_obj=cache_obj,
    )

    assert first == second == {"email": "alice@example.com", "segment": "enterprise"}
    assert first_hit is False
    assert second_hit is True
    assert call_count[0] == 1


def test_shared_memory_scope_shares_tool_results_between_caches() -> None:
    clear_shared_memory("team-alpha")
    cache_a = _make_cache(Config(enable_token_counter=False, memory_scope="team-alpha"))
    cache_b = _make_cache(Config(enable_token_counter=False, memory_scope="team-alpha"))

    cache_a.remember_tool_result(
        "inventory.lookup",
        {"sku": "SKU-42"},
        {"stock": 9},
    )

    assert cache_b.recall_tool_result("inventory.lookup", {"sku": "SKU-42"}) == {"stock": 9}
    assert cache_b.tool_memory_stats()["hits"] == 1


def test_intent_graph_tracks_multi_step_session_flow() -> None:
    cache_obj = _make_cache(Config(enable_token_counter=False, intent_memory=True))

    cache_obj.record_intent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": 'Classify the sentiment. Labels: POSITIVE, NEGATIVE, NEUTRAL Review: "I loved it."',
                }
            ]
        },
        session_id="session-1",
    )
    cache_obj.record_intent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": 'Summarize the following article in one sentence. Article: "Byte Cache reduces repeated LLM calls."',
                }
            ]
        },
        session_id="session-1",
    )

    stats = cache_obj.intent_stats()
    assert stats["total_records"] == 2
    assert stats["transition_count"] == 1
    assert stats["top_transitions"][0]["from"] == "classification"
    assert stats["top_transitions"][0]["to"] == "summarization::one_sentence"


def test_memory_snapshot_can_share_intents_and_tool_results_between_caches() -> None:
    cache_a = _make_cache(Config(enable_token_counter=False))
    cache_b = _make_cache(Config(enable_token_counter=False))

    cache_a.record_intent(
        {
            "messages": [
                {
                    "role": "user",
                    "content": 'Classify the sentiment. Labels: POSITIVE, NEGATIVE, NEUTRAL Review: "I loved it."',
                }
            ]
        },
        session_id="session-share",
    )
    cache_a.remember_tool_result(
        "weather.lookup",
        {"city": "Paris"},
        {"temp_c": 21},
    )

    snapshot = export_memory_snapshot(cache_obj=cache_a)
    imported = import_memory_snapshot(snapshot, cache_obj=cache_b)

    assert imported["tool_results"]["imported"] == 1
    assert cache_b.recall_tool_result("weather.lookup", {"city": "Paris"}) == {"temp_c": 21}
    assert cache_b.intent_stats()["total_records"] == 1
    assert cache_b.intent_stats()["unique_intents"] == 1


def test_ai_memory_snapshot_shares_answers_reasoning_and_embedding_metadata_between_caches() -> None:
    cache_a = _make_cache(Config(enable_token_counter=False))
    cache_b = _make_cache(Config(enable_token_counter=False))

    remember_interaction(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Translate to French: Hello world"}],
        },
        answer="Bonjour le monde",
        reasoning="direct translation",
        tool_outputs={"dictionary": "hello -> bonjour"},
        embedding_data=[0.1, 0.2, 0.3],
        metadata={"source": "unit-test"},
        cache_obj=cache_a,
    )

    snapshot = export_memory_snapshot(cache_obj=cache_a)
    imported = import_memory_snapshot(snapshot, cache_obj=cache_b)
    recent = recent_interactions(cache_obj=cache_b)

    assert imported["ai_memory"]["imported"] == 1
    assert cache_b.ai_memory_stats()["total_entries"] == 1
    assert recent[0]["answer"] == "Bonjour le monde"
    assert recent[0]["reasoning"] == "direct translation"
    assert recent[0]["tool_outputs"] == {"dictionary": "hello -> bonjour"}
    assert recent[0]["embedding"]["dimensions"] == 3


def test_reasoning_memory_snapshot_round_trips_between_caches() -> None:
    cache_a = _make_cache(Config(enable_token_counter=False))
    cache_b = _make_cache(Config(enable_token_counter=False))

    cache_a.remember_reasoning_result(
        kind="capital_city",
        key="capital-france",
        answer="Paris",
        verified=True,
        metadata={"reason": "seed"},
    )

    snapshot = export_memory_snapshot(cache_obj=cache_a)
    imported = import_memory_snapshot(snapshot, cache_obj=cache_b)
    reused = cache_b.lookup_reasoning_result(key="capital-france", kind="capital_city")

    assert imported["reasoning_memory"]["imported"] == 1
    assert reused is not None
    assert reused["answer"] == "Paris"
