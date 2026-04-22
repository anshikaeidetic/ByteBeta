"""Risk assessment helpers for Byte trust decisions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from byte.processor.intent import extract_request_intent
from byte.processor.pre import normalize_text
from byte.processor.route_signals import extract_route_signals

from ._calibration import DEFAULT_CALIBRATION_VERSION, load_trust_calibration
from ._contracts import extract_contract, request_text
from ._references import _auxiliary_context_values, _curated_label_reference

_LABEL_PATTERNS = (
    re.compile(r"(?is)\blabels?\s*:\s*(?P<labels>[^\n]+)"),
    re.compile(
        r"(?is)\b(?:one|single)\s+(?:label|action)\s+(?:from|out\s+of)\s*\{(?P<labels>[^}]+)\}"
    ),
)
_JSON_KEYS_PATTERN = re.compile(
    r"(?is)\bvalid\s+json\s+only\s+with\s+keys?\s+(?P<keys>[^.\n]+)"
)
_SET_VALUE_PATTERN = re.compile(
    r"(?is)(?:\bset\b|\band\b)\s+(?P<key>[A-Za-z0-9_ -]+?)\s+to\s+(?P<value>[A-Za-z0-9_.:-]+)"
)
_RULE_PATTERN = re.compile(
    r"(?is)\bif\s+(?P<condition>.+?)\s+return\s+(?P<label>[A-Z][A-Z0-9_:-]{1,})"
)
_ELSE_IF_PATTERN = re.compile(
    r"(?is)\belse\s+if\s+(?P<condition>.+?)\s+return\s+(?P<label>[A-Z][A-Z0-9_:-]{1,})"
)
_ELSE_PATTERN = re.compile(
    r"(?is)\b(?:otherwise|else)\b(?:[^A-Z\n]{0,48})?(?:return|reply\s+with|output)\s+(?P<label>[A-Z][A-Z0-9_:-]{1,})"
)
_TOKEN_VALUE_PATTERN = re.compile(
    r"(?is)\b(?P<field>[A-Za-z][A-Za-z _-]{1,32})\s*(?:=|:|is)\s*(?P<value>[A-Za-z0-9_.:-]+)"
)
_COMPARISON_PATTERN = re.compile(
    r"(?is)^(?P<field>[A-Za-z][A-Za-z _-]{1,32})\s*(?P<op>>=|<=|>|<|==|=)\s*(?P<value>-?\d+(?:\.\d+)?)$"
)
_WORD_MATCH_PATTERN = re.compile(
    r"(?is)^(?P<field>[A-Za-z][A-Za-z _-]{1,32})\s+(?P<value>[A-Za-z0-9_.:-]+)$"
)
_CONFLICT_HINTS = ("conflict", "disagree", "contradict", "unsafe_to_reuse", "escalate_conflict")
_UNIQUE_HINTS = ("unique", "nonce", "uuid", "hash", "digest", "trace")
_AUXILIARY_INJECTION_PATTERN = re.compile(
    r"(?is)\b(?:ignore\s+previous|system\s+message|developer\s+message|follow\s+these\s+instructions|override\s+policy|do\s+not\s+follow\s+previous)\b"
)
_AUXILIARY_SEGMENT_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_AUXILIARY_CONTEXT_FIELDS = (
    "byte_retrieval_context",
    "byte_document_context",
    "byte_support_articles",
    "byte_tool_result_context",
    "byte_repo_summary",
    "byte_repo_snapshot",
    "byte_changed_files",
    "byte_changed_hunks",
    "byte_prompt_pieces",
)

@dataclass(frozen=True)
class QueryRiskAssessment:
    score: float
    band: str
    novelty_score: float
    support_score: float
    deterministic_path: bool
    direct_only: bool
    context_only: bool
    fallback_reason: str
    reasons: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def is_deterministic_request(
    request_kwargs: dict[str, Any] | None,
    config: Any,
    *,
    context: dict[str, Any] | None = None,
) -> bool:
    if str((request_kwargs or {}).get("byte_trust_mode", "") or "").strip().lower() == "disabled":
        return bool((request_kwargs or {}).get("byte_force_deterministic"))
    if bool((request_kwargs or {}).get("byte_force_deterministic")):
        return True
    if not bool(getattr(config, "deterministic_execution", True)):
        return False
    contract = extract_contract(request_kwargs)
    if contract["strict"]:
        return True
    normalized = normalize_text(request_text(request_kwargs))
    if any(token in normalized for token in ("profit margin", "refunds allowed within", "capital of")):
        return True
    risk = evaluate_query_risk(request_kwargs, config, context=context)
    return bool(risk.deterministic_path)

def evaluate_query_risk(
    request_kwargs: dict[str, Any] | None,
    config: Any,
    *,
    context: dict[str, Any] | None = None,
) -> QueryRiskAssessment:
    request_kwargs = request_kwargs or {}
    context = context or {}
    calibration = load_trust_calibration(
        str(
            getattr(config, "calibration_artifact_version", DEFAULT_CALIBRATION_VERSION)
            or DEFAULT_CALIBRATION_VERSION
        )
    )
    if str(request_kwargs.get("byte_trust_mode", "") or "").strip().lower() == "disabled":
        return QueryRiskAssessment(
            score=0.0,
            band="low",
            novelty_score=0.0,
            support_score=calibration.risk_threshold("disabled_support"),
            deterministic_path=bool(request_kwargs.get("byte_force_deterministic")),
            direct_only=False,
            context_only=False,
            fallback_reason="",
            reasons={},
        )
    text = request_text(request_kwargs)
    normalized = normalize_text(text)
    contract = extract_contract(request_kwargs)
    intent = extract_request_intent(request_kwargs)
    signals = extract_route_signals(
        request_kwargs,
        long_prompt_chars=max(1, int(getattr(config, "routing_long_prompt_chars", 1200) or 1200)),
        multi_turn_threshold=max(
            1, int(getattr(config, "routing_multi_turn_threshold", 6) or 6)
        ),
    )
    reasons: dict[str, float] = {}

    novelty_score = 0.0
    support_score = (
        calibration.risk_score("strict_support")
        if contract["strict"]
        else calibration.risk_score("default_support")
    )
    direct_only = False
    context_only = False
    fallback_reason = ""
    deterministic_path = bool(contract["strict"])

    if signals.factual_risk:
        reasons["factual_risk"] = calibration.risk_score("factual_risk")
    if signals.jailbreak_risk or signals.pii_risk:
        reasons["policy_sensitivity"] = calibration.risk_score("policy_sensitivity")
    if signals.needs_reasoning:
        reasons["reasoning"] = calibration.risk_score("reasoning")

    if any(token in normalized for token in _CONFLICT_HINTS):
        novelty_score = max(novelty_score, calibration.risk_threshold("conflicting_context_novelty"))
        reasons["conflicting_context"] = calibration.risk_score("conflicting_context")
        direct_only = True
        fallback_reason = "conflicting_context"

    if _has_auxiliary_instruction_injection(request_kwargs, context):
        novelty_score = max(novelty_score, calibration.risk_threshold("retrieval_injection_novelty"))
        reasons["retrieval_injection"] = calibration.risk_score("retrieval_injection")
        direct_only = True
        context_only = False
        fallback_reason = fallback_reason or "retrieval_injection"

    if _has_grounded_context(request_kwargs, context):
        support_score = max(support_score, calibration.risk_threshold("grounded_context_support"))
        context_only = True
        reasons["grounded_context"] = calibration.risk_score("grounded_context")
        if _looks_like_grounded_exact_lookup(normalized):
            deterministic_path = True
            support_score = max(
                support_score,
                calibration.risk_threshold("grounded_exact_lookup_support"),
            )
            reasons["grounded_exact_lookup"] = max(
                float(reasons.get("grounded_exact_lookup", 0.0) or 0.0),
                calibration.risk_score("grounded_exact_lookup"),
            )

    if _looks_like_unique_request(normalized, contract["exact_token"]):
        direct_only = True
        novelty_score = max(novelty_score, calibration.risk_threshold("unique_output_novelty"))
        fallback_reason = fallback_reason or "unique_output_contract"
        reasons["unique_output"] = calibration.risk_score("unique_output")

    if _looks_like_novel_rule_prompt(text, contract):
        novelty_score = max(novelty_score, calibration.risk_threshold("novel_rule_novelty"))
        direct_only = True
        deterministic_path = True
        fallback_reason = fallback_reason or "novel_rule_prompt"
        reasons["novel_rule"] = calibration.risk_score("novel_rule")

    if _looks_like_novel_json_prompt(text, contract):
        novelty_score = max(novelty_score, calibration.risk_threshold("novel_json_novelty"))
        direct_only = True
        deterministic_path = True
        fallback_reason = fallback_reason or "novel_structured_contract"
        reasons["novel_json_contract"] = calibration.risk_score("novel_json_contract")

    if _looks_like_curated_policy_prompt(text, contract):
        novelty_score = max(novelty_score, calibration.risk_threshold("curated_policy_novelty"))
        direct_only = True
        deterministic_path = True
        fallback_reason = fallback_reason or "curated_policy_family"
        reasons["curated_policy_family"] = calibration.risk_score("curated_policy_family")

    if contract["strict"] and not direct_only:
        reasons["strict_contract"] = max(
            reasons.get("strict_contract", 0.0),
            calibration.risk_score("strict_contract"),
        )
        support_score = max(support_score, calibration.risk_threshold("strict_contract_support"))

    if intent.category in {"classification", "extraction", "exact_answer", "translation"}:
        deterministic_path = True
        support_score = max(support_score, calibration.risk_threshold("deterministic_intent_support"))

    if context_only and direct_only:
        context_only = False

    score = max(
        0.0,
        min(
            1.0,
            round(
                sum(reasons.values())
                + novelty_score * calibration.risk_threshold("novelty_score_scale"),
                4,
            ),
        ),
    )
    band = (
        "high"
        if score >= calibration.risk_threshold("high_band")
        else "medium"
        if score >= calibration.risk_threshold("medium_band")
        else "low"
    )

    return QueryRiskAssessment(
        score=score,
        band=band,
        novelty_score=round(max(0.0, min(1.0, novelty_score)), 4),
        support_score=round(max(0.0, min(1.0, support_score)), 4),
        deterministic_path=deterministic_path,
        direct_only=direct_only,
        context_only=context_only,
        fallback_reason=fallback_reason,
        reasons=reasons,
    )

def _looks_like_novel_rule_prompt(text: str, contract: dict[str, Any]) -> bool:
    normalized = normalize_text(text)
    return bool(
        contract["labels"]
        and " if " in f" {normalized} "
        and (" otherwise " in f" {normalized} " or " else " in f" {normalized} ")
        and "refunds allowed within" not in normalized
    )


def _looks_like_novel_json_prompt(text: str, contract: dict[str, Any]) -> bool:
    normalized = normalize_text(text)
    return bool(
        contract["structured_format"] == "json"
        and "valid json only" in normalized
        and "set" in normalized
    )


def _looks_like_curated_policy_prompt(text: str, contract: dict[str, Any]) -> bool:
    labels = {
        str(label or "").strip().strip(".").upper()
        for label in contract.get("labels", [])
        if str(label or "").strip().strip(".")
    }
    if not {"ALLOW", "REVIEW", "BLOCK"}.issubset(labels):
        return False
    return bool(_curated_label_reference(text))


def _looks_like_grounded_exact_lookup(normalized: str) -> bool:
    if not any(
        phrase in normalized
        for phrase in ("return exactly the", "return only the", "nothing else")
    ):
        return False
    return any(
        token in normalized
        for token in (
            "invoice identifier",
            "function name",
            "city name",
            "policy label",
            "action label",
            "owner label",
            "queue identifier",
            "capital of",
            "root cause label",
        )
    )


def _has_grounded_context(
    request_kwargs: dict[str, Any] | None,
    context: dict[str, Any],
) -> bool:
    request_kwargs = request_kwargs or {}
    raw_aux = context.get("_byte_raw_aux_context", {}) or {}
    if not isinstance(raw_aux, dict):
        raw_aux = {}
    for field_name in _AUXILIARY_CONTEXT_FIELDS:
        if raw_aux.get(field_name) not in (None, "", [], {}):
            return True
        if request_kwargs.get(field_name) not in (None, "", [], {}):
            return True
    return False


def _has_auxiliary_instruction_injection(
    request_kwargs: dict[str, Any] | None,
    context: dict[str, Any],
) -> bool:
    for value in _auxiliary_context_values(request_kwargs, context):
        if _AUXILIARY_INJECTION_PATTERN.search(value):
            return True
    return False

def _looks_like_unique_request(normalized: str, exact_token: str) -> bool:
    if exact_token and any(token in normalize_text(exact_token) for token in _UNIQUE_HINTS):
        return True
    return any(token in normalized for token in _UNIQUE_HINTS)
