from byte import Cache, Config
from byte.adapter.pipeline.consensus import _run_cheap_consensus_sync
from byte.processor.model_router import ModelRouteDecision
from byte.processor.quality import ResponseAssessment


def _route_decision() -> object:
    return ModelRouteDecision(
        original_model="cheap-a",
        selected_model="cheap-a",
        tier="cheap",
        reason="classification_request",
        category="classification",
        route_key="classification",
        prompt_chars=80,
        message_count=1,
        has_tools=False,
        applied=False,
        signals={},
    )


def test_cheap_consensus_accepts_agreement() -> object:
    cache_obj = Cache()
    cache_obj.config = Config(
        cheap_consensus_enabled=True,
        cheap_consensus_models=["cheap-b"],
        cheap_consensus_min_score=0.5,
        routing_verify_min_score=0.9,
    )
    context = {
        "_byte_model_route": _route_decision(),
        "_byte_uncertainty": {"requires_consensus": True},
    }
    llm_data = {"choices": [{"message": {"content": "TECHNICAL"}}]}
    assessment = ResponseAssessment(
        score=0.7,
        accepted=True,
        repaired_answer="TECHNICAL",
        reason="label_matched",
        constraint="label_set",
    )

    def fake_llm(**kwargs) -> object:
        assert kwargs["model"] == "cheap-b"
        return {"choices": [{"message": {"content": "TECHNICAL"}}]}

    updated_llm_data, updated_assessment = _run_cheap_consensus_sync(
        cache_obj,
        fake_llm,
        (),
        {
            "model": "cheap-a",
            "messages": [
                {
                    "role": "user",
                    "content": "Classify and return one label from {BILLING, TECHNICAL}.",
                }
            ],
        },
        context,
        llm_data,
        assessment,
        task_policy={"verify_min_score": 0.9},
    )

    assert updated_llm_data["choices"][0]["message"]["content"] == "TECHNICAL"
    assert updated_assessment.accepted is True
    assert context["_byte_consensus"]["agreed"] is True


def test_cheap_consensus_marks_disagreement_for_escalation() -> object:
    cache_obj = Cache()
    cache_obj.config = Config(
        cheap_consensus_enabled=True,
        cheap_consensus_models=["cheap-b"],
        cheap_consensus_min_score=0.5,
        routing_verify_min_score=0.9,
    )
    context = {
        "_byte_model_route": _route_decision(),
        "_byte_uncertainty": {"requires_consensus": True},
    }

    def fake_llm(**kwargs) -> object:
        assert kwargs["model"] == "cheap-b"
        return {"choices": [{"message": {"content": "BILLING"}}]}

    _, updated_assessment = _run_cheap_consensus_sync(
        cache_obj,
        fake_llm,
        (),
        {
            "model": "cheap-a",
            "messages": [
                {
                    "role": "user",
                    "content": "Classify and return one label from {BILLING, TECHNICAL}.",
                }
            ],
        },
        context,
        {"choices": [{"message": {"content": "TECHNICAL"}}]},
        ResponseAssessment(
            score=0.7,
            accepted=True,
            repaired_answer="TECHNICAL",
            reason="label_matched",
            constraint="label_set",
        ),
        task_policy={"verify_min_score": 0.9},
    )

    assert updated_assessment.accepted is False
    assert updated_assessment.reason == "cheap_consensus_disagreement"
    assert context["_byte_counterfactual"]["action"] == "direct_expensive"
