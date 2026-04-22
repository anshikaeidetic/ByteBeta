from byte import Config
from byte.processor.uncertainty import estimate_request_uncertainty


def test_uncertainty_scales_high_for_wide_extraction_schema() -> None:
    config = Config(
        routing_long_prompt_chars=1200,
        context_budget_low_risk_chars=2000,
        context_budget_medium_risk_chars=4200,
        context_budget_high_risk_chars=7600,
    )
    assessment = estimate_request_uncertainty(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Extract the fields.\n"
                        "Fields: ticket_id, customer_name, issue_type, refund_amount, due_date, owner\n"
                        "Ticket: customer was charged twice and needs help.\n"
                        "Return JSON only."
                    ),
                }
            ]
        },
        config,
        failure_hint={"prefer_expensive": True},
    )

    assert assessment.band in {"medium", "high"}
    assert assessment.recommended_context_chars >= 4200
    assert assessment.structured is True


def test_uncertainty_keeps_simple_exact_answer_low_risk() -> None:
    config = Config(
        context_budget_low_risk_chars=1800,
        context_budget_medium_risk_chars=4200,
        context_budget_high_risk_chars=7600,
    )
    assessment = estimate_request_uncertainty(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly BYTE_OK and nothing else.",
                }
            ]
        },
        config,
    )

    assert assessment.band == "low"
    assert assessment.recommended_context_chars == 1800
