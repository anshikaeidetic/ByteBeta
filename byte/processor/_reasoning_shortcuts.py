"""Public reasoning shortcut resolution and assessment helpers."""

# ruff: noqa: F401
import re
from typing import Any, Optional

from byte.processor._reasoning_patterns import (
    _answer_matches_shortcut,
    _canonical_knowledge_answer,
    _capital_prefers_brief_answer,
    _cleanup_entity,
    _extract_exact_token,
    _extract_labels,
    _extract_request_text,
    _grounded_policy_text,
    _has_aux_context,
    _number,
    _prefers_brief_answer,
    _raw_aux_context,
    _requested_grounded_target,
    _resolve_grounded_target_value,
    _solve_profit_margin,
    _solve_refund_policy_label,
)
from byte.processor._reasoning_store import (
    ReasoningMemoryStore,
    ReasoningShortcut,
    _candidate_promoted,
    _prompt_module_signatures,
    _prompt_signature,
    _related_lookup_min_score,
)
from byte.processor.coding_analysis import (
    extract_label_candidates,
    infer_coding_label_from_request,
    supports_coding_exact_contract,
)
from byte.processor.intent import extract_request_intent
from byte.processor.optimization_memory import stable_digest
from byte.processor.pre import normalize_text

_PRICE_PATTERNS = (
    re.compile(
        r"(?is)\b(?:selling|sale|sell|product)\s+price\s*(?:=|:)\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    ),
    re.compile(
        r"(?is)\b(?:selling|sale)\s+price(?:\s+is)?\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    ),
    re.compile(r"(?is)\bprice\s*(?:=|:)\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?is)\bprice(?:\s+is)?\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?is)\bsold\b.{0,24}?\bat\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?is)\bsells?\b.{0,48}?\bfor\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?is)\brevenue\s*(?:=|:|is)?\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"),
)
_COST_PATTERNS = (
    re.compile(
        r"(?is)\b(?:production|marketing|shipping|operating|labor|labour|materials?|support|service|overhead)\s+cost\s*(?:=|:)\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    ),
    re.compile(
        r"(?is)\b(?:production|marketing|shipping|operating|labor|labour|materials?|support|service|overhead)\s*=\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
    ),
)
_POLICY_LABEL_PATTERN = re.compile(r"(?is)\blabels?\s*:\s*(?P<labels>[^\n]+)")
_POLICY_LABEL_SET_PATTERN = re.compile(
    r"(?is)\b(?:one|single)\s+(?:label|action)\s+(?:from|out\s+of)\s*\{(?P<labels>[^}]+)\}"
)
_POLICY_WINDOW_PATTERN = re.compile(
    r"(?is)\brefunds?\s+(?:are\s+)?allowed\s+within\s+(?P<days>\d+)\s+days?"
)
_POLICY_DAY_PATTERN = re.compile(r"(?is)\bday\s*(?P<day>\d+)\b")
_CAPITAL_QUERY_PATTERNS = (
    re.compile(
        r"(?is)\bwhat\s+is\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"
    ),
    re.compile(
        r"(?is)\bwhich\s+city\s+is\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"
    ),
    re.compile(r"(?is)\bname\s+the\s+capital(?:\s+city)?\s+of\s+(?P<entity>[a-z][a-z .'-]{1,60})"),
)
_POLICY_APPROVE_HINTS = ("approve", "approved", "accept", "allow", "yes", "refund_approve")
_POLICY_DENY_HINTS = ("deny", "denied", "reject", "decline", "no", "refund_deny")
_POLICY_ESCALATE_HINTS = ("escalate", "review", "manual")
_EXACT_TOKEN_PATTERNS = (
    re.compile(
        r"(?is)(?:return|reply|respond|answer)\s+with\s+exactly\s+(?P<token>[A-Za-z0-9_.:-]+)"
    ),
    re.compile(r"(?is)exactly\s+(?P<token>[A-Za-z0-9_.:-]+)\s+and\s+nothing\s+else"),
)
_GENERIC_EXACT_TOKENS = {"one", "label", "labels", "token", "word", "json", "yaml", "csv"}
_AUX_CONTEXT_FIELDS = (
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
_GROUNDED_TARGET_RULES = (
    (
        "primary_service",
        ("validated primary service identifier", "primary service identifier", "primary service"),
    ),
    ("fallback_service", ("fallback service identifier", "fallback service")),
    ("queue_name", ("queue identifier", "queue name", "queue")),
    ("policy_label", ("architecture policy label", "policy label")),
    ("invoice_id", ("invoice identifier", "invoice id")),
    ("amount", ("total amount due", "amount due", "open amount", "total amount")),
    ("owner", ("owner label", "owner")),
    (
        "cause",
        (
            "incident root-cause label",
            "incident root cause label",
            "root-cause label",
            "root cause label",
            "cause label",
        ),
    ),
    ("due_date", ("follow-up due date", "due date")),
    (
        "action_label",
        ("prescribed action label", "final action label", "action label", "next action label"),
    ),
)
_INVOICE_PATTERN = re.compile(r"\bINV-\d+\b")
_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_AMOUNT_PATTERN = re.compile(r"\$\d[\d,]*(?:\.\d+)?")
_OWNER_PATTERNS = (
    re.compile(r"(?is)\bbelongs\s+to\s+owner\s+(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"),
    re.compile(r"(?is)\bowner(?:\s+label)?\s*(?:is|:)?\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"),
)
_CAUSE_PATTERNS = (
    re.compile(
        r"(?is)\broot[\s-]+cause\s+label(?:\s+for\s+case\s+\d+)?\s*(?:is|:)?\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"
    ),
    re.compile(
        r"(?is)\bcause\s+label(?:\s+for\s+case\s+\d+)?\s*(?:is|:)\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"
    ),
)
_ACTION_PATTERNS = (
    re.compile(r"(?is)\baction\s+label\s+should\s+be\s+(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"),
    re.compile(
        r"(?is)\bprescribed\s+action\s*(?:label\s*)?(?:is|:)?\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"
    ),
    re.compile(
        r"(?is)\bfinal\s+action\s+label(?:\s+should\s+be|\s+is|:)\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"
    ),
)
_PRIMARY_SERVICE_PATTERNS = (
    re.compile(r"(?is)api\s+gateway\s*->\s*(?P<value>[A-Za-z0-9_-]+)\s*->"),
    re.compile(r"(?is)\bfor\s+(?P<value>svc-\d+)\b"),
)
_FALLBACK_SERVICE_PATTERNS = (
    re.compile(r"(?is)\bfallback\s+traffic\s+routes\s+to\s+(?P<value>[A-Za-z0-9_-]+)\b"),
    re.compile(r"(?is)\bdiverts\s+to\s+(?P<value>[A-Za-z0-9_-]+)\b"),
)
_QUEUE_PATTERNS = (
    re.compile(r"(?is)->\s*(?P<value>queue-[A-Za-z0-9_-]+)\b"),
    re.compile(
        r"(?is)\bqueue(?:\s+identifier|\s+name)?\s*(?:is|:)?\s*(?P<value>queue-[A-Za-z0-9_-]+)\b"
    ),
)
_POLICY_LABEL_PATTERNS = (
    re.compile(r"(?is)\bpolicy\s+label\s*(?:is|:)?\s*(?P<value>[A-Z][A-Z0-9_:-]{2,})\b"),
)
_CURATED_CAPITAL_CITIES = {
    "france": "Paris",
    "italy": "Rome",
    "japan": "Tokyo",
    "canada": "Ottawa",
    "australia": "Canberra",
    "spain": "Madrid",
    "germany": "Berlin",
    "portugal": "Lisbon",
    "austria": "Vienna",
    "ireland": "Dublin",
    "norway": "Oslo",
    "sweden": "Stockholm",
    "finland": "Helsinki",
    "denmark": "Copenhagen",
    "belgium": "Brussels",
    "switzerland": "Bern",
    "poland": "Warsaw",
    "greece": "Athens",
    "czech republic": "Prague",
    "hungary": "Budapest",
}


def resolve_reasoning_shortcut(
    request_kwargs: dict[str, Any] | None,
    *,
    store: ReasoningMemoryStore | None = None,
    config: Any | None = None,
    context_hints: dict[str, Any] | None = None,
) -> ReasoningShortcut | None:
    if not getattr(config, "reasoning_reuse", True):
        return None
    contract = _exact_contract_shortcut(request_kwargs, context_hints=context_hints)
    if contract is not None:
        return contract
    curated = _curated_knowledge_shortcut(request_kwargs)
    if curated is not None:
        return curated
    memory_candidate = _reasoning_memory_candidate(request_kwargs, context_hints=context_hints)
    if getattr(config, "reasoning_memory", True) and store is not None and memory_candidate is not None:
        entry = store.lookup(
            key=memory_candidate["key"],
            kind=memory_candidate["kind"],
            verified_only=True,
        )
        if entry is None and memory_candidate.get("allow_related"):
            min_related_score = _related_lookup_min_score(memory_candidate, config=config)
            related = store.lookup_related(
                query_text=str(memory_candidate.get("query_text", "") or ""),
                kind=str(memory_candidate.get("kind", "") or ""),
                verified_only=True,
                min_score=min_related_score,
                top_k=1,
            )
            entry = related[0] if related else None
        if entry is not None and _candidate_promoted(memory_candidate, entry):
            related_score = float(entry.get("related_score", 1.0) or 1.0)
            confidence = max(0.9, min(0.99, 0.88 + related_score * 0.1))
            return ReasoningShortcut(
                kind=str(memory_candidate.get("kind", "") or ""),
                answer=str(entry.get("answer", "") or ""),
                confidence=round(confidence, 4),
                reason="knowledge_memory_hit",
                byte_reason="reasoning_memory_reuse",
                key=str(memory_candidate.get("key", "") or ""),
                source="memory",
                constraint=str(
                    memory_candidate.get("constraint", "knowledge_reuse") or "knowledge_reuse"
                ),
                promotion_state=str(
                    (entry.get("metadata", {}) or {}).get("promotion_state", "") or "dynamic_verified"
                ),
            )
    grounded = _grounded_context_shortcut(request_kwargs, context_hints=context_hints)
    if grounded is not None:
        return grounded
    if getattr(config, "coding_reasoning_shortcut", False):
        coding = _coding_shortcut(request_kwargs)
        if coding is not None:
            return coding
    deterministic = _deterministic_shortcut(request_kwargs)
    if deterministic is not None:
        return deterministic
    return None


def assess_reasoning_answer(
    request_kwargs: dict[str, Any] | None,
    answer_text: str,
    *,
    config: Any | None = None,
) -> dict[str, Any] | None:
    if not getattr(config, "reasoning_repair", True):
        return None
    shortcut = _deterministic_shortcut(request_kwargs)
    if shortcut is None:
        return None
    matched = _answer_matches_shortcut(shortcut, answer_text)
    return {
        "accepted": True,
        "repaired_answer": shortcut.answer,
        "reason": "deterministic_reasoning_verified"
        if matched
        else "deterministic_reasoning_repaired",
        "constraint": shortcut.constraint,
        "score": 0.995 if matched else 0.975,
        "shortcut": shortcut.to_dict(),
    }


def derive_reasoning_memory_record(
    request_kwargs: dict[str, Any] | None,
    answer_text: Any,
    *,
    context_hints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    candidate = _reasoning_memory_candidate(request_kwargs, context_hints=context_hints)
    if candidate is not None:
        request_text = _extract_request_text(request_kwargs)
        answer = _canonical_knowledge_answer(
            str(candidate["kind"]),
            answer_text,
            brief_answer=bool(candidate.get("brief_answer", False)),
        )
        if not answer:
            answer = " ".join(str(answer_text or "").split()).strip()
        if not answer:
            return None
        return {
            "kind": str(candidate["kind"]),
            "key": str(candidate["key"]),
            "answer": answer,
            "metadata": {
                "reason": str(candidate.get("reason", "reasoning_memory_seed") or "reasoning_memory_seed"),
                "constraint": str(candidate.get("constraint", "knowledge_reuse") or "knowledge_reuse"),
                "parameter_sensitive": bool(candidate.get("parameter_sensitive", False)),
                "promotion_required": bool(candidate.get("promotion_required", False)),
                "prompt_signature": _prompt_signature(request_text),
                "prompt_signatures": _prompt_module_signatures(request_kwargs),
                "promotion_group": str(candidate.get("promotion_group", "") or candidate.get("key", "")),
                "promotion_state": "near_threshold_shadow"
                if bool(candidate.get("promotion_required", False))
                else "dynamic_verified",
            },
        }
    return None


def capital_query_key(text: Any) -> str:
    candidate = _capital_query_entity(text)
    if not candidate:
        return ""
    return candidate["key"]


def _deterministic_shortcut(
    request_kwargs: dict[str, Any] | None,
) -> ReasoningShortcut | None:
    request_text = _extract_request_text(request_kwargs)
    normalized = normalize_text(request_text)
    if not normalized or not _prefers_brief_answer(normalized):
        return None
    profit_answer = _solve_profit_margin(request_text, normalized)
    if profit_answer:
        candidate = _profit_margin_memory_candidate(request_text)
        return ReasoningShortcut(
            kind="profit_margin",
            answer=profit_answer,
            confidence=0.995,
            reason="profit_margin_formula",
            byte_reason="deterministic_reasoning",
            key=str(candidate.get("key", "") or stable_digest({"kind": "profit_margin", "text": normalized})),
            source="deterministic",
            constraint="numeric_answer",
            promotion_state="guarded",
        )
    policy_answer = _solve_refund_policy_label(request_text, normalized)
    if policy_answer:
        candidate = _refund_policy_memory_candidate(request_text)
        return ReasoningShortcut(
            kind="policy_label",
            answer=policy_answer,
            confidence=0.99,
            reason="refund_policy_rule",
            byte_reason="deterministic_reasoning",
            key=str(candidate.get("key", "") or stable_digest({"kind": "policy_label", "text": normalized})),
            source="deterministic",
            constraint="label_set",
            promotion_state="guarded",
        )
    return None


def _knowledge_memory_candidate(
    request_kwargs: dict[str, Any] | None,
) -> dict[str, str] | None:
    request_text = _extract_request_text(request_kwargs)
    capital = _capital_query_entity(request_text)
    if capital is not None:
        return capital
    return None


def _reasoning_memory_candidate(
    request_kwargs: dict[str, Any] | None,
    *,
    context_hints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    request_text = _extract_request_text(request_kwargs)
    candidate = _profit_margin_memory_candidate(request_text)
    if candidate is not None:
        return candidate
    candidate = _refund_policy_memory_candidate(request_text)
    if candidate is not None:
        return candidate
    candidate = _grounded_memory_candidate(request_kwargs, context_hints=context_hints)
    if candidate is not None:
        return candidate
    candidate = _knowledge_memory_candidate(request_kwargs)
    if candidate is not None:
        candidate["constraint"] = "knowledge_reuse"
        candidate["reason"] = "knowledge_memory_seed"
        candidate["allow_related"] = str(candidate.get("kind", "") or "") != "capital_city"
        candidate["query_text"] = request_text
        candidate["min_related_score"] = 0.94
        return candidate
    return None


def _profit_margin_memory_candidate(request_text: str) -> dict[str, Any] | None:
    normalized = normalize_text(request_text)
    if "profit margin" not in normalized and "margin percentage" not in normalized:
        return None
    price = None
    for pattern in _PRICE_PATTERNS:
        match = pattern.search(request_text)
        if match:
            price = _number(match.group("value"))
            break
    if price is None or price <= 0:
        return None
    costs = []
    for pattern in _COST_PATTERNS:
        for match in pattern.finditer(request_text):
            value = _number(match.group("value"))
            if value is not None:
                costs.append(round(float(value), 4))
    if not costs:
        return None
    key = stable_digest(
        {
            "kind": "profit_margin",
            "price": round(float(price), 4),
            "costs": costs,
        }
    )
    return {
        "kind": "profit_margin",
        "key": key,
        "constraint": "numeric_answer",
        "reason": "profit_margin_formula",
        "parameter_sensitive": True,
        "promotion_required": True,
        "promotion_group": key,
    }


def _refund_policy_memory_candidate(request_text: str) -> dict[str, Any] | None:
    normalized = normalize_text(request_text)
    if "refund" not in normalized or "day" not in normalized:
        return None
    labels = _extract_labels(request_text)
    if not labels:
        return None
    window_match = _POLICY_WINDOW_PATTERN.search(request_text)
    if not window_match:
        return None
    days = [int(match.group("day")) for match in _POLICY_DAY_PATTERN.finditer(request_text)]
    if not days:
        return None
    key = stable_digest(
        {
            "kind": "policy_label",
            "window": int(window_match.group("days")),
            "day": int(days[-1]),
            "labels": [label.upper() for label in labels],
        }
    )
    return {
        "kind": "policy_label",
        "key": key,
        "constraint": "label_set",
        "reason": "refund_policy_rule",
        "parameter_sensitive": True,
        "promotion_required": True,
        "promotion_group": key,
    }


def _grounded_memory_candidate(
    request_kwargs: dict[str, Any] | None,
    *,
    context_hints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    raw_aux = _raw_aux_context(request_kwargs, context_hints)
    if not raw_aux:
        return None
    request_text = _extract_request_text(request_kwargs)
    target = _requested_grounded_target(request_text)
    if not target:
        return None
    return {
        "kind": f"grounded_{target}",
        "key": stable_digest(
            {
                "kind": f"grounded_{target}",
                "target": target,
                "context": stable_digest(raw_aux),
            }
        ),
        "constraint": "grounded_value",
        "reason": f"grounded_context_{target}",
    }


def _capital_query_entity(text: Any) -> dict[str, str] | None:
    normalized = normalize_text(text)
    if "capital" not in normalized:
        return None
    for pattern in _CAPITAL_QUERY_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        entity = _cleanup_entity(match.group("entity"))
        if not entity:
            continue
        return {
            "kind": "capital_city",
            "entity": entity,
            "key": stable_digest({"kind": "capital_city", "entity": entity}),
            "brief_answer": _capital_prefers_brief_answer(normalized),
        }
    return None


def _curated_knowledge_shortcut(
    request_kwargs: dict[str, Any] | None,
) -> ReasoningShortcut | None:
    request_text = _extract_request_text(request_kwargs)
    candidate = _capital_query_entity(request_text)
    if candidate is None or not bool(candidate.get("brief_answer", False)):
        return None
    entity = str(candidate.get("entity", "") or "").strip()
    capital = _CURATED_CAPITAL_CITIES.get(entity)
    if not capital:
        return None
    return ReasoningShortcut(
        kind="capital_city",
        answer=capital,
        confidence=0.999,
        reason="verified_capital_city_reference",
        byte_reason="deterministic_reasoning",
        key=str(candidate.get("key", "") or stable_digest({"kind": "capital_city", "entity": entity})),
        source="deterministic",
        constraint="knowledge_fact",
        promotion_state="guarded",
    )


def _exact_contract_shortcut(
    request_kwargs: dict[str, Any] | None,
    *,
    context_hints: dict[str, Any] | None = None,
) -> ReasoningShortcut | None:
    intent = extract_request_intent(request_kwargs or {})
    if intent.category != "exact_answer":
        return None
    request_text = _extract_request_text(request_kwargs)
    normalized = normalize_text(request_text)
    if "exactly" not in normalized:
        return None
    if "nothing else" not in normalized and "only" not in normalized:
        return None
    token = _extract_exact_token(request_text)
    if not token:
        return None
    return ReasoningShortcut(
        kind="exact_contract",
        answer=token,
        confidence=1.0,
        reason="explicit_exact_output_contract",
        byte_reason="contract_shortcut",
        key=stable_digest({"kind": "exact_contract", "token": normalize_text(token)}),
        source="contract",
        constraint="exact_token",
    )


def _grounded_context_shortcut(
    request_kwargs: dict[str, Any] | None,
    *,
    context_hints: dict[str, Any] | None = None,
) -> ReasoningShortcut | None:
    raw_aux = _raw_aux_context(request_kwargs, context_hints)
    if not raw_aux:
        return None
    request_text = _extract_request_text(request_kwargs)
    target = _requested_grounded_target(request_text)
    if not target:
        return None
    answer = _resolve_grounded_target_value(target, raw_aux)
    if not answer:
        return None
    return ReasoningShortcut(
        kind=f"grounded_{target}",
        answer=answer,
        confidence=0.995,
        reason=f"grounded_context_{target}",
        byte_reason="grounded_context_shortcut",
        key=stable_digest({"kind": target, "answer": normalize_text(answer)}),
        source="grounded_context",
        constraint="grounded_value",
    )


def _coding_shortcut(request_kwargs: dict[str, Any] | None) -> ReasoningShortcut | None:
    if request_kwargs is None:
        return None
    intent = extract_request_intent(request_kwargs or {})
    category = str(getattr(intent, "category", "") or "")
    if category not in {
        "code_fix",
        "code_explanation",
        "test_generation",
        "documentation",
        "code_refactor",
    }:
        return None
    request_text = _extract_request_text(request_kwargs)
    normalized = normalize_text(request_text)
    token = _extract_exact_token(request_text)
    if (
        token
        and "exactly" in normalized
        and ("nothing else" in normalized or "only" in normalized)
        and supports_coding_exact_contract(category, token)
    ):
        return ReasoningShortcut(
            kind=f"{category}_contract",
            answer=token,
            confidence=0.995,
            reason="coding_exact_output_contract",
            byte_reason="coding_contract_shortcut",
            key=stable_digest({"kind": category, "token": normalize_text(token)}),
            source="contract",
            constraint="exact_token",
        )
    labels = extract_label_candidates(request_text)
    if not labels:
        return None
    inferred = infer_coding_label_from_request(
        category=category,
        request_text=request_text,
        labels=labels,
        style=str(getattr(intent, "slots", {}).get("style") or ""),
    )
    if not inferred:
        return None
    return ReasoningShortcut(
        kind=f"{category}_analysis",
        answer=inferred,
        confidence=0.985,
        reason="static_code_analysis",
        byte_reason="coding_analysis_shortcut",
        key=stable_digest({"kind": category, "answer": normalize_text(inferred)}),
        source="deterministic",
        constraint="label_set" if labels else "exact_token",
    )


__all__ = [
    "ReasoningMemoryStore",
    "ReasoningShortcut",
    "assess_reasoning_answer",
    "capital_query_key",
    "derive_reasoning_memory_record",
    "resolve_reasoning_shortcut",
]
