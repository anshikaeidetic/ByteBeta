from typing import Any

from byte.processor.quality import ResponseAssessment
from byte.utils.time import time_cal

from .context import _provider_request_kwargs
from .utils import _extract_llm_answer, _set_llm_answer, _try_record_budget
from .verifier import _assess_and_repair_response, _verification_bands


def _should_attempt_cheap_consensus(
    chat_cache, route_decision, assessment, context, *, task_policy=None
) -> bool:
    if not getattr(chat_cache.config, "cheap_consensus_enabled", False):
        return False
    if route_decision is None or route_decision.tier != "cheap":
        return False
    if assessment is None or assessment.constraint == "freeform":
        return False
    uncertainty = dict(context.get("_byte_uncertainty", {}) or {})
    if not uncertainty.get("requires_consensus", False):
        return False
    grey_threshold, verify_threshold = _verification_bands(
        chat_cache,
        route_decision,
        assessment,
        context,
        task_policy=task_policy,
    )
    min_score = float(getattr(chat_cache.config, "cheap_consensus_min_score", 0.5) or 0.5)
    lower_bound = max(min_score, grey_threshold)
    score = float(assessment.score or 0.0)
    if not assessment.accepted:
        return score >= lower_bound
    return lower_bound <= score < verify_threshold


def _cheap_consensus_candidates(chat_cache, request_kwargs) -> Any:
    current_model = str(request_kwargs.get("model", "") or "")
    configured = [
        str(item)
        for item in (getattr(chat_cache.config, "cheap_consensus_models", []) or [])
        if str(item).strip()
    ]
    if current_model and current_model not in configured:
        configured.insert(0, current_model)
    if len(configured) <= 1:
        return []
    return [model for model in configured if model != current_model]


def _consensus_value(assessment, llm_data) -> Any:
    if assessment is not None and assessment.repaired_answer not in (None, ""):
        return str(assessment.repaired_answer).strip()
    answer = _extract_llm_answer(llm_data)
    if answer in (None, ""):
        return ""
    return str(answer).strip()


def _run_cheap_consensus_sync(
    chat_cache,
    llm_handler,
    args,
    request_kwargs,
    context,
    llm_data,
    assessment,
    *,
    task_policy=None,
) -> tuple[Any, ...]:
    if not _should_attempt_cheap_consensus(
        chat_cache, context.get("_byte_model_route"), assessment, context, task_policy=task_policy
    ):
        return llm_data, assessment
    candidates = _cheap_consensus_candidates(chat_cache, request_kwargs)
    if not candidates:
        return llm_data, assessment

    base_value = _consensus_value(assessment, llm_data)
    for candidate_model in candidates[:1]:
        try:
            candidate_kwargs = dict(request_kwargs)
            candidate_kwargs["model"] = candidate_model
            provider_candidate_kwargs = _provider_request_kwargs(
                chat_cache, candidate_kwargs, context
            )
            candidate_llm_data = time_cal(
                llm_handler,
                func_name="llm_consensus",
                report_func=chat_cache.report.llm,
            )(*args, **provider_candidate_kwargs)
            _try_record_budget(candidate_llm_data, candidate_model, chat_cache=chat_cache)
            candidate_llm_data, candidate_assessment = _assess_and_repair_response(
                chat_cache,
                candidate_kwargs,
                candidate_llm_data,
            )
            candidate_value = _consensus_value(candidate_assessment, candidate_llm_data)
            if base_value and candidate_value and base_value == candidate_value:
                context["_byte_consensus"] = {"agreed": True, "model": candidate_model}
                return _set_llm_answer(llm_data, base_value), assessment
            context["_byte_counterfactual"] = {
                "action": "direct_expensive",
                "reason": "cheap_consensus_disagreement",
            }
            return llm_data, ResponseAssessment(
                score=0.0,
                accepted=False,
                repaired_answer=None,
                reason="cheap_consensus_disagreement",
                constraint=assessment.constraint,
            )
        except Exception:  # pylint: disable=W0703
            return llm_data, assessment
    return llm_data, assessment


async def _run_cheap_consensus_async(
    chat_cache,
    llm_handler,
    args,
    request_kwargs,
    context,
    llm_data,
    assessment,
    *,
    task_policy=None,
) -> tuple[Any, ...]:
    if not _should_attempt_cheap_consensus(
        chat_cache, context.get("_byte_model_route"), assessment, context, task_policy=task_policy
    ):
        return llm_data, assessment
    candidates = _cheap_consensus_candidates(chat_cache, request_kwargs)
    if not candidates:
        return llm_data, assessment

    base_value = _consensus_value(assessment, llm_data)
    for candidate_model in candidates[:1]:
        try:
            candidate_kwargs = dict(request_kwargs)
            candidate_kwargs["model"] = candidate_model
            provider_candidate_kwargs = _provider_request_kwargs(
                chat_cache, candidate_kwargs, context
            )
            candidate_llm_data = await llm_handler(*args, **provider_candidate_kwargs)
            _try_record_budget(candidate_llm_data, candidate_model, chat_cache=chat_cache)
            candidate_llm_data, candidate_assessment = _assess_and_repair_response(
                chat_cache,
                candidate_kwargs,
                candidate_llm_data,
            )
            candidate_value = _consensus_value(candidate_assessment, candidate_llm_data)
            if base_value and candidate_value and base_value == candidate_value:
                context["_byte_consensus"] = {"agreed": True, "model": candidate_model}
                return _set_llm_answer(llm_data, base_value), assessment
            context["_byte_counterfactual"] = {
                "action": "direct_expensive",
                "reason": "cheap_consensus_disagreement",
            }
            return llm_data, ResponseAssessment(
                score=0.0,
                accepted=False,
                repaired_answer=None,
                reason="cheap_consensus_disagreement",
                constraint=assessment.constraint,
            )
        except Exception:  # pylint: disable=W0703
            return llm_data, assessment
    return llm_data, assessment


__all__ = [name for name in globals() if not name.startswith("__")]
