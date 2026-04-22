import uuid

from byte import Cache, Config
from byte.adapter.api import export_memory_snapshot, recent_interactions, remember_interaction
from byte.manager.factory import get_data_manager
from byte.manager.scalar_data.base import Answer, DataType, Question
from byte.security import SecureDataManager


def _make_cache(tmp_path, config=None) -> object:
    cache_obj = Cache()
    cache_obj.init(
        data_manager=get_data_manager(data_path=str(tmp_path / f"data_map_{uuid.uuid4().hex}.txt")),
        config=config or Config(enable_token_counter=False),
    )
    return cache_obj


def test_secure_data_manager_encrypts_cache_values_at_rest(tmp_path) -> None:
    cache_obj = _make_cache(
        tmp_path,
        Config(
            enable_token_counter=False,
            security_mode=True,
            security_encryption_key="unit-test-secret",
        ),
    )

    assert isinstance(cache_obj.data_manager, SecureDataManager)

    cache_obj.data_manager.save(
        Question(content="patient diagnosis"),
        [Answer("private-answer", DataType.STR)],
        embedding_data="fingerprint-1",
    )

    stored_entry = next(iter(cache_obj.data_manager.delegate.data.values()))
    stored_question = stored_entry[0]
    stored_answers = stored_entry[1]
    stored_question_text = (
        stored_question.content if hasattr(stored_question, "content") else stored_question
    )
    assert stored_question_text != "patient diagnosis"
    assert stored_answers[0].answer != "private-answer"
    assert (
        cache_obj.data_manager.protector.decrypt_text(stored_question_text) == "patient diagnosis"
    )
    assert (
        cache_obj.data_manager.protector.decrypt_text(stored_answers[0].answer) == "private-answer"
    )


def test_security_redacts_memory_views(tmp_path) -> None:
    cache_obj = _make_cache(
        tmp_path,
        Config(
            enable_token_counter=False,
            security_mode=True,
            security_redact_memory=True,
        ),
    )

    remember_interaction(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Patient Alice needs a follow-up note"}],
        },
        answer="Schedule a cardiology follow-up for Alice.",
        reasoning="patient-specific recommendation",
        metadata={"mrn": "12345"},
        cache_obj=cache_obj,
    )

    recent = recent_interactions(cache_obj=cache_obj)
    snapshot = export_memory_snapshot(cache_obj=cache_obj)

    assert str(recent[0]["answer"]).startswith("[redacted len=")
    assert str(recent[0]["reasoning"]).startswith("[redacted len=")
    assert str(snapshot["ai_memory"]["entries"][0]["metadata"]["mrn"]).startswith("[redacted len=")
