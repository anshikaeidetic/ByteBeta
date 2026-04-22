"""Scoring and metadata assembly for Byte trust evaluation."""

from __future__ import annotations

from typing import Any

from ._calibration import DEFAULT_CALIBRATION_VERSION, TrustCalibration, load_trust_calibration
from ._risk import QueryRiskAssessment, evaluate_query_risk


def build_trust_metadata(
    request_kwargs: dict[str, Any] | None,
    *,
    config: Any,
    context: dict[str, Any] | None = None,
    served_via: str = "",
    accepted: bool = True,
    assessment_score: float = 0.0,
    byte_reason: str = "",
    reuse_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    calibration_version = str(
        getattr(config, "calibration_artifact_version", DEFAULT_CALIBRATION_VERSION)
        or DEFAULT_CALIBRATION_VERSION
    )
    calibration = load_trust_calibration(calibration_version)
    risk = evaluate_query_risk(request_kwargs, config, context=context)
    verifier = dict(context.get("_byte_verifier", {}) or {})
    assessment = dict(context.get("_byte_assessment", {}) or {})
    reuse_evidence = dict(reuse_evidence or {})
    raw_confidence = _raw_confidence_score(
        risk=risk,
        served_via=served_via,
        accepted=accepted,
        assessment_score=assessment_score,
        byte_reason=byte_reason,
        verifier=verifier,
        reuse_evidence=reuse_evidence,
        calibration=calibration,
    )
    confidence = _apply_calibration(raw_confidence, calibration=calibration)
    if risk.direct_only and served_via == "reuse":
        confidence = min(confidence, calibration.confidence_threshold("direct_only_cap"))
    if not accepted:
        confidence = min(confidence, calibration.confidence_threshold("not_accepted_cap"))
    if (
        str(getattr(config, "conformal_mode", "guarded") or "guarded").strip().lower()
        in {"guarded", "enforced"}
        and risk.direct_only
    ):
        confidence = min(confidence, calibration.confidence_threshold("direct_only_cap"))
    promotion_state = str(
        reuse_evidence.get("promotion_state", "")
        or reuse_evidence.get("tier", "")
        or "verified"
    )
    if served_via == "reuse" and not reuse_evidence.get("promotion_state") and not reuse_evidence.get("tier"):
        promotion_state = "static_curated" if byte_reason == "reasoning_memory_reuse" else "verified"
    elif risk.direct_only and promotion_state == "verified":
        promotion_state = "shadow"
    verifier_scores = {
        "assessment": round(max(0.0, min(1.0, float(assessment_score or 0.0))), 4),
        "provider_accept": 1.0 if verifier.get("verdict") == "accept" else 0.0,
        "provider_reject": 1.0 if verifier.get("verdict") == "reject" else 0.0,
    }
    novelty_reason = _primary_novelty_reason(risk)
    contract_constraint = str(assessment.get("constraint", "") or "")
    contract_validated = bool(
        accepted
        and contract_constraint
        and contract_constraint != "freeform"
    )
    repair_applied = bool(assessment.get("repair_applied", False))
    reuse_tier = _reuse_tier(served_via, byte_reason, promotion_state)
    return {
        "query_risk": risk.to_dict(),
        "novelty_score": risk.novelty_score,
        "novelty_reason": novelty_reason,
        "support_margin": round(risk.support_score - risk.novelty_score, 4),
        "reuse_evidence": reuse_evidence,
        "reuse_tier": reuse_tier,
        "verifier_scores": verifier_scores,
        "calibrated_confidence": round(confidence, 4),
        "abstained": bool(risk.direct_only or not accepted),
        "deterministic_path": bool(risk.deterministic_path),
        "contract_validated": contract_validated,
        "repair_applied": repair_applied,
        "promotion_state": promotion_state,
        "fallback_reason": risk.fallback_reason
        or ("verification_failed" if not accepted else ""),
        "artifact_version": str(
            getattr(config, "calibration_artifact_version", DEFAULT_CALIBRATION_VERSION)
            or DEFAULT_CALIBRATION_VERSION
        ),
        "calibration_version": calibration.version,
        "calibration_status": calibration.status,
        "calibration_source_status": calibration.source_status,
        "calibration_public_proof_status": calibration.public_proof_status,
        "calibration_public_proof_manifest": calibration.public_proof_manifest,
        "calibration_checksum": calibration.checksum,
        "calibration_method": calibration.method,
        "benchmark_contract_version": str(
            getattr(config, "benchmark_contract_version", "byte-benchmark-v2")
            or "byte-benchmark-v2"
        ),
        "debug": {
            "reasons": dict(risk.reasons),
            "raw_confidence": round(raw_confidence, 4),
        }
        if bool((request_kwargs or {}).get("byte_confidence_debug"))
        else {},
    }

def _raw_confidence_score(
    *,
    risk: QueryRiskAssessment,
    served_via: str,
    accepted: bool,
    assessment_score: float,
    byte_reason: str,
    verifier: dict[str, Any],
    reuse_evidence: dict[str, Any],
    calibration: TrustCalibration,
) -> float:
    kind = str(reuse_evidence.get("kind", "") or "")
    promotion_state = str(
        reuse_evidence.get("promotion_state", "") or reuse_evidence.get("tier", "") or ""
    )
    if not accepted:
        return calibration.confidence_score("not_accepted")
    if risk.direct_only:
        if (
            served_via == "upstream"
            and risk.deterministic_path
            and assessment_score >= calibration.confidence_threshold("direct_high_assessment")
        ):
            return calibration.confidence_score("direct_upstream_verified")
        return calibration.confidence_score("direct_default")
    if served_via == "reuse":
        score = calibration.confidence_score("reuse_verified")
        if byte_reason == "reasoning_memory_reuse":
            score = (
                calibration.confidence_score("reasoning_dynamic_verified")
                if promotion_state == "dynamic_verified"
                else calibration.confidence_score("reasoning_static")
            )
    elif served_via == "local_compute":
        if kind in {"profit_margin", "policy_label"} and promotion_state != "dynamic_verified":
            score = calibration.confidence_score("guarded_local_compute")
        else:
            score = (
                calibration.confidence_score("local_deterministic")
                if risk.deterministic_path
                else calibration.confidence_score("local_default")
            )
    else:
        if risk.context_only and assessment_score >= calibration.confidence_threshold("context_high_assessment"):
            score = calibration.confidence_score("context_high_assessment")
        elif (
            risk.context_only
            and risk.deterministic_path
            and assessment_score >= calibration.confidence_threshold("context_min_assessment")
        ):
            score = calibration.confidence_score("context_deterministic")
        else:
            score = (
                calibration.confidence_score("upstream_deterministic_high_assessment")
                if risk.deterministic_path
                and assessment_score >= calibration.confidence_threshold("upstream_high_assessment")
                else calibration.confidence_score("upstream_deterministic")
                if risk.deterministic_path
                else calibration.confidence_score("upstream_default")
            )
    score += min(
        calibration.confidence_adjustment("assessment_boost_cap"),
        max(0.0, float(assessment_score or 0.0) - calibration.confidence_threshold("assessment_boost_floor"))
        * calibration.confidence_adjustment("assessment_boost_scale"),
    )
    score -= risk.novelty_score * calibration.confidence_adjustment("novelty_penalty")
    score += (
        risk.support_score - calibration.confidence_threshold("support_baseline")
    ) * calibration.confidence_adjustment("support_margin_scale")
    verdict = str(verifier.get("verdict", "") or "").lower()
    if verdict == "accept":
        score += calibration.confidence_adjustment("verifier_accept_boost")
    elif verdict == "reject":
        score -= calibration.confidence_adjustment("verifier_reject_penalty")
    return max(0.0, min(calibration.confidence_threshold("raw_cap"), score))


def _apply_calibration(raw_score: float, *, calibration: TrustCalibration) -> float:
    if not calibration.version:
        return round(raw_score, 4)
    for bucket in calibration.buckets:
        if raw_score >= bucket.threshold:
            return round(bucket.mapped, 4)
    return round(raw_score, 4)


def _primary_novelty_reason(risk: QueryRiskAssessment) -> str:
    if risk.fallback_reason:
        return str(risk.fallback_reason)
    weighted = sorted(
        ((str(key), float(value or 0.0)) for key, value in (risk.reasons or {}).items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return weighted[0][0] if weighted else ""


def _reuse_tier(served_via: str, byte_reason: str, promotion_state: str) -> str:
    if served_via != "reuse":
        return "none"
    if byte_reason == "reasoning_memory_reuse":
        return promotion_state or "dynamic_verified"
    return "static_curated"
