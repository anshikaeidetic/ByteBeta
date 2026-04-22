from typing import Any

from byte.processor.intent import extract_request_intent
from byte.processor.quality import ResponseAssessment
from byte.utils.time import time_cal

from .context import (
    _build_output_contract_instruction,
    _output_contract_payload,
    _provider_safe_kwargs,
)
from .utils import (
    _clamp01,
    _extract_llm_answer,
    _set_llm_answer,
    _try_record_budget,
    get_quality_scorer,
)


def _assess_and_repair_response(
    chat_cache, request_kwargs, llm_data, *, context=None, task_policy=None
) -> tuple[Any, ...]:
    answer_text = _extract_llm_answer(llm_data)
    if answer_text in (None, ""):
        return llm_data, None
    scorer = get_quality_scorer(chat_cache)
    assessment = scorer.assess_request_answer(
        request_kwargs,
        answer_text,
        context_hints=context,
        config=chat_cache.config,
        task_policy=task_policy,
    )
    repaired_answer = assessment.repaired_answer
    applied = False
    if (
        getattr(chat_cache.config, "response_repair", True)
        and repaired_answer not in (None, "")
        and repaired_answer != answer_text
    ):
        llm_data = _set_llm_answer(llm_data, repaired_answer)
        applied = True
    if getattr(chat_cache.config, "response_repair", True):
        scorer.record_repair(applied=applied)
    return llm_data, assessment


def _repair_cached_answer(
    chat_cache, request_kwargs, return_message, *, context=None, task_policy=None
) -> tuple[Any, ...]:
    if not isinstance(return_message, str):
        return return_message, None
    assessment = get_quality_scorer(chat_cache).assess_request_answer(
        request_kwargs,
        return_message,
        context_hints=context,
        config=chat_cache.config,
        task_policy=task_policy,
    )
    if getattr(chat_cache.config, "response_repair", True) and assessment.repaired_answer not in (
        None,
        "",
    ):
        return assessment.repaired_answer, assessment
    return return_message, assessment


def _should_run_verifier_model(chat_cache, route_decision, assessment, context) -> bool:
    verifier_model = str(getattr(chat_cache.config, "routing_verifier_model", "") or "")
    if not getattr(chat_cache.config, "routing_verifier_enabled", False):
        return False
    if (
        not verifier_model
        or assessment is None
        or not assessment.accepted
        or route_decision is None
    ):
        return False
    if str(getattr(route_decision, "selected_model", "") or "") == verifier_model:
        return False
    signals = dict(getattr(route_decision, "signals", {}) or {})
    category = str(getattr(route_decision, "category", "") or "")
    if category in {"code_fix", "code_refactor", "test_generation", "documentation"}:
        return True
    if assessment.constraint != "freeform":
        return True
    if route_decision.tier in {"coder", "reasoning"}:
        return True
    return bool(
        signals.get("needs_reasoning")
        or signals.get("factual_risk")
        or signals.get("has_multimodal_input")
    )


def _extract_request_text_for_verifier(request_kwargs, *, max_chars: int = 6000) -> str:
    text = ""
    messages = request_kwargs.get("messages") or []
    if messages:
        parts = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "") or item.get("content", "") or ""))
                    else:
                        parts.append(str(item or ""))
            else:
                parts.append(str(content or ""))
        text = "\n".join(part for part in parts if part)
    elif request_kwargs.get("prompt") is not None:
        text = str(request_kwargs.get("prompt") or "")
    elif request_kwargs.get("input") is not None:
        text = str(request_kwargs.get("input") or "")
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _build_verifier_messages(request_kwargs, route_decision, candidate_answer, context) -> list[Any]:
    category = str(getattr(route_decision, "category", "") or "unknown")
    task_text = _extract_request_text_for_verifier(request_kwargs)
    contract_instruction = _build_output_contract_instruction(_output_contract_payload(context))
    user_lines = [
        f"Task category: {category}",
        "Original request:",
        task_text or "(empty request)",
        "",
        "Candidate answer:",
        str(candidate_answer or ""),
    ]
    if contract_instruction:
        user_lines.extend(["", "Output contract:", contract_instruction])
    user_lines.extend(
        [
            "",
            "Reply with exactly ACCEPT if the candidate answer fully satisfies the request.",
            "Reply with exactly REJECT if any requirement is missing, wrong, unsafe, or unverifiable.",
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are Byte's strict answer verifier. "
                "Return exactly ACCEPT or REJECT. If unsure, return REJECT."
            ),
        },
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _parse_verifier_verdict(llm_data) -> str:
    answer = str(_extract_llm_answer(llm_data) or "").strip().lower()
    if not answer:
        return ""
    normalized = []
    for char in answer:
        normalized.append(char if char.isalnum() or char.isspace() else " ")
    tokens = [token for token in "".join(normalized).split() if token]
    if not tokens:
        return ""
    first = tokens[0]
    if first in {"accept", "accepted", "approve", "approved", "yes", "pass", "valid", "correct"}:
        return "accept"
    if first in {"reject", "rejected", "deny", "denied", "no", "fail", "invalid", "incorrect"}:
        return "reject"
    return ""


def _record_verifier_result(context, *, verifier_model, verdict, raw_answer, route_decision) -> None:
    context["_byte_verifier"] = {
        "model": verifier_model,
        "verdict": verdict,
        "raw_answer": raw_answer,
        "route_tier": str(getattr(route_decision, "tier", "") or ""),
        "category": str(getattr(route_decision, "category", "") or ""),
    }


def _verifier_rejection_assessment(assessment) -> Any:
    if assessment is None:
        return ResponseAssessment(
            score=0.0,
            accepted=False,
            repaired_answer=None,
            reason="verifier_model_rejected",
            constraint="freeform",
        )
    return ResponseAssessment(
        score=0.0,
        accepted=False,
        repaired_answer=None,
        reason="verifier_model_rejected",
        constraint=assessment.constraint,
    )


def _run_verifier_model_sync(
    chat_cache,
    llm_handler,
    args,
    request_kwargs,
    context,
    llm_data,
    assessment,
) -> tuple[Any, ...]:
    route_decision = context.get("_byte_model_route")
    if not _should_run_verifier_model(chat_cache, route_decision, assessment, context):
        return llm_data, assessment
    from .consensus import _consensus_value  # local import to avoid pipeline import cycles

    verifier_model = str(getattr(chat_cache.config, "routing_verifier_model", "") or "")
    candidate_answer = _consensus_value(assessment, llm_data)
    if not candidate_answer:
        return llm_data, assessment
    verifier_kwargs = {
        "model": verifier_model,
        "messages": _build_verifier_messages(
            request_kwargs, route_decision, candidate_answer, context
        ),
        "temperature": 0,
        "max_tokens": 4,
    }
    try:
        verifier_llm_data = time_cal(
            llm_handler,
            func_name="llm_verifier",
            report_func=chat_cache.report.llm,
        )(*args, **_provider_safe_kwargs(verifier_kwargs))
        _try_record_budget(verifier_llm_data, verifier_model, chat_cache=chat_cache)
    except Exception:  # pylint: disable=W0703
        return llm_data, assessment
    verdict = _parse_verifier_verdict(verifier_llm_data)
    _record_verifier_result(
        context,
        verifier_model=verifier_model,
        verdict=verdict or "unknown",
        raw_answer=_extract_llm_answer(verifier_llm_data),
        route_decision=route_decision,
    )
    if verdict == "reject":
        return llm_data, _verifier_rejection_assessment(assessment)
    return llm_data, assessment


async def _run_verifier_model_async(
    chat_cache,
    llm_handler,
    args,
    request_kwargs,
    context,
    llm_data,
    assessment,
) -> tuple[Any, ...]:
    route_decision = context.get("_byte_model_route")
    if not _should_run_verifier_model(chat_cache, route_decision, assessment, context):
        return llm_data, assessment
    from .consensus import _consensus_value  # local import to avoid pipeline import cycles

    verifier_model = str(getattr(chat_cache.config, "routing_verifier_model", "") or "")
    candidate_answer = _consensus_value(assessment, llm_data)
    if not candidate_answer:
        return llm_data, assessment
    verifier_kwargs = {
        "model": verifier_model,
        "messages": _build_verifier_messages(
            request_kwargs, route_decision, candidate_answer, context
        ),
        "temperature": 0,
        "max_tokens": 4,
    }
    try:
        verifier_llm_data = await llm_handler(*args, **_provider_safe_kwargs(verifier_kwargs))
        _try_record_budget(verifier_llm_data, verifier_model, chat_cache=chat_cache)
    except Exception:  # pylint: disable=W0703
        return llm_data, assessment
    verdict = _parse_verifier_verdict(verifier_llm_data)
    _record_verifier_result(
        context,
        verifier_model=verifier_model,
        verdict=verdict or "unknown",
        raw_answer=_extract_llm_answer(verifier_llm_data),
        route_decision=route_decision,
    )
    if verdict == "reject":
        return llm_data, _verifier_rejection_assessment(assessment)
    return llm_data, assessment


def _verification_bands(chat_cache, route_decision, assessment, context, *, task_policy=None) -> tuple[Any, ...]:
    base_verify = float(
        (task_policy or {}).get("verify_min_score")
        or getattr(chat_cache.config, "routing_verify_min_score", 0.75)
        or 0.75
    )
    base_grey = float(
        (task_policy or {}).get("grey_zone_min_score")
        or getattr(chat_cache.config, "routing_grey_zone_min_score", 0.55)
        or 0.55
    )
    category = str(getattr(route_decision, "category", "") or "")
    if not category:
        ambiguity = context.get("_byte_ambiguity")
        category = str(getattr(ambiguity, "category", "") or "")
    if not category:
        category = str(
            extract_request_intent(context.get("_byte_request_kwargs", {}) or {}).category or ""
        )

    if category in {"classification", "exact_answer", "translation"}:
        base_verify -= 0.07
        base_grey -= 0.05
    elif category in {
        "extraction",
        "code_fix",
        "code_refactor",
        "test_generation",
        "documentation",
    }:
        base_verify += 0.04
        base_grey += 0.03

    if str(getattr(assessment, "constraint", "") or "") == "exact_token":
        base_verify -= 0.05
        base_grey -= 0.04
    elif str(getattr(assessment, "constraint", "") or "") in {"json", "yaml", "csv"}:
        base_verify += 0.03

    uncertainty = dict(context.get("_byte_uncertainty", {}) or {})
    band = str(uncertainty.get("band", "") or "").lower()
    if band == "high":
        base_verify += 0.03
        base_grey += 0.02
    elif band == "low":
        base_verify -= 0.03
        base_grey -= 0.02

    if bool((context.get("_byte_failure_hint", {}) or {}).get("avoid_cache_reuse")):
        base_verify += 0.02
        base_grey += 0.02

    verify_threshold = _clamp01(base_verify)
    grey_threshold = _clamp01(min(verify_threshold, base_grey))
    if grey_threshold > verify_threshold - 0.02:
        grey_threshold = max(0.0, verify_threshold - 0.02)
    return grey_threshold, verify_threshold


__all__ = [name for name in globals() if not name.startswith("__")]
