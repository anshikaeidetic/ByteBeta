"""Output-contract extraction and structured constraint helpers."""

from __future__ import annotations

import ast
import csv
import json
import re
from csv import DictReader
from io import StringIO
from typing import Any

from byte.processor._quality_models import OutputContract, ResponseAssessment
from byte.processor.coding_analysis import (
    extract_code_block as coding_extract_code_block,
)
from byte.processor.coding_analysis import (
    infer_bug_label_from_request as coding_infer_bug_label_from_request,
)
from byte.processor.coding_analysis import (
    infer_complexity_label_from_request as coding_infer_complexity_label_from_request,
)
from byte.processor.coding_analysis import (
    infer_framework_label_from_request as coding_infer_framework_label_from_request,
)
from byte.processor.intent import extract_request_intent
from byte.processor.pre import normalize_text


def extract_output_contract(request_kwargs: dict[str, Any]) -> OutputContract:
    request_kwargs = request_kwargs or {}
    request_text = _extract_request_text(request_kwargs)
    intent = extract_request_intent(request_kwargs)
    exact_token = _extract_exact_token(request_text)
    labels = tuple(_labels_from_slots(intent.slots) or _extract_label_candidates(request_text))
    structured_format = _expected_structured_format(intent, request_text)
    strict = bool(exact_token or labels or structured_format)
    return OutputContract(
        category=str(getattr(intent, "category", "") or ""),
        exact_token=exact_token,
        labels=labels,
        structured_format=structured_format,
        strict=strict,
    )


_EXACT_TOKEN_PATTERNS = (
    re.compile(
        r"(?is)(?:return|reply|respond|answer)\s+with\s+exactly\s+(?P<token>[A-Za-z0-9_.:-]+)"
    ),
    re.compile(r"(?is)exactly\s+(?P<token>[A-Za-z0-9_.:-]+)\s+and\s+nothing\s+else"),
)

_LABEL_PATTERNS = (
    re.compile(r"(?is)(?:labels?|classes?)\s*:\s*(?P<labels>[^\n]+)"),
    re.compile(r"(?is)(?:framework|complexity)?\s*label\s+from\s*\{(?P<labels>[^}]+)\}"),
    re.compile(r"(?is)\bfrom\s*\{(?P<labels>[^}]+)\}"),
)

_GENERIC_EXACT_TOKENS = {"one", "label", "labels", "token", "word", "json", "yaml", "csv"}
_GENERIC_EXACT_TOKEN_PHRASES = {
    "the city",
    "city",
    "the city name",
    "city name",
    "the label",
    "label",
    "the final action label",
    "final action label",
    "action label",
    "the final answer",
    "final answer",
    "the percentage",
    "percentage",
}
_EVIDENCE_FIELDS = (
    "byte_retrieval_context",
    "byte_document_context",
    "byte_support_articles",
    "byte_tool_result_context",
    "byte_repo_summary",
    "byte_repo_snapshot",
    "byte_changed_files",
    "byte_changed_hunks",
)
_EVIDENCE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "their",
    "there",
    "have",
    "has",
    "been",
    "were",
    "was",
    "will",
    "would",
    "should",
    "could",
    "only",
    "exactly",
    "answer",
    "reply",
    "respond",
    "return",
    "label",
    "json",
    "yaml",
    "csv",
}


def _extract_request_text(request_kwargs: dict[str, Any]) -> str:
    messages = request_kwargs.get("messages") or []
    if messages:
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "") or item.get("content", "") or ""))
                else:
                    parts.append(str(item or ""))
            return " ".join(part for part in parts if part)
        return str(content or "")
    if request_kwargs.get("prompt") is not None:
        return str(request_kwargs.get("prompt") or "")
    if request_kwargs.get("input") is not None:
        return str(request_kwargs.get("input") or "")
    return ""


def _extract_exact_token(request_text: str) -> str:
    for pattern in _EXACT_TOKEN_PATTERNS:
        match = pattern.search(request_text or "")
        if match:
            token = match.group("token").strip()
            if _is_generic_exact_token_candidate(token):
                continue
            return token
    return ""


def _is_generic_exact_token_candidate(token: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(token or "").strip().lower())
    if not normalized:
        return True
    if normalized in _GENERIC_EXACT_TOKENS or normalized in _GENERIC_EXACT_TOKEN_PHRASES:
        return True
    if normalized.startswith("the ") and normalized[4:] in _GENERIC_EXACT_TOKEN_PHRASES:
        return True
    if " " in normalized and not re.search(r"[A-Z0-9_:-]", token or ""):
        return True
    return False


def _extract_label_candidates(request_text: str) -> list[str]:
    labels: list[str] = []
    for pattern in _LABEL_PATTERNS:
        match = pattern.search(request_text or "")
        if not match:
            continue
        labels.extend(_split_label_values(match.group("labels")))
    ordered = []
    seen = set()
    for item in labels:
        key = item.upper()
        if key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def _split_label_values(raw: str) -> list[str]:
    parts = re.split(r"[,/|;\n]", raw or "")
    values = []
    for part in parts:
        candidate = part.strip().strip("{}[]()")
        candidate = re.sub(
            r"\b(?:and|or|only|exactly|reply|respond|answer|return|with|one)\b",
            " ",
            candidate,
            flags=re.I,
        )
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate:
            continue
        token_match = re.findall(r"[A-Za-z0-9_./:-]+", candidate)
        if not token_match:
            continue
        token = token_match[0]
        if any(ch.isalpha() for ch in token):
            values.append(token)
    return values


def _contains_token(answer_text: str, token: str) -> bool:
    if not token:
        return False
    if (
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", answer_text, flags=re.I)
        is not None
    ):
        return True
    normalized_answer = normalize_text(answer_text)
    for alias in _label_aliases(token):
        if _contains_normalized_phrase(normalized_answer, alias):
            return True
    return False


def _match_best_label(answer_text: str, labels: list[str]) -> str:
    normalized_answer = normalize_text(answer_text)
    for label in sorted(labels, key=len, reverse=True):
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])",
            answer_text,
            flags=re.I,
        )
        if match is not None:
            return match.group(0)
        for alias in _label_aliases(label):
            if _contains_normalized_phrase(normalized_answer, alias):
                return label
    return ""


def _label_aliases(label: str) -> list[str]:
    canonical = str(label or "").strip()
    if not canonical:
        return []
    aliases = {
        normalize_text(canonical),
        normalize_text(canonical.replace("_", " ")),
        normalize_text(canonical.replace("-", " ")),
        normalize_text(canonical.replace("/", " ")),
        normalize_text(canonical.replace(".", " ")),
    }
    special_aliases = {
        "MUTABLE_DEFAULT": {"mutable default", "mutable default argument"},
        "OFF_BY_ONE": {"off by one", "off by one error", "off by one bug"},
        "SYNTAX_ERROR": {"syntax error", "invalid syntax"},
        "BROAD_EXCEPTION": {"broad exception", "bare except"},
        "DOCSTRING_READY": {"docstring ready", "documentation ready"},
        "DOCSTRING_DONE": {"docstring done", "documentation done"},
        "READABILITY_REFACTOR": {"readability refactor", "readability improvement"},
        "O_1": {"o 1", "o one", "constant", "constant time", "constant complexity"},
        "O_N": {"o n", "linear", "linear time", "linear complexity"},
        "O_N_SQUARED": {
            "o n squared",
            "o n 2",
            "quadratic",
            "quadratic time",
            "nested loop",
            "nested loops",
        },
    }
    aliases.update(special_aliases.get(canonical.upper(), set()))
    return [alias for alias in aliases if alias]


def _contains_normalized_phrase(normalized_text: str, phrase: str) -> bool:
    text = f" {str(normalized_text or '').strip()} "
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in text


def _repair_exact_token_from_task_output(
    intent: Any, request_text: str, answer_text: str, exact_token: str
) -> ResponseAssessment | None:
    category = str(getattr(intent, "category", "") or "")
    request_symbols = _code_symbol_names(request_text)
    answer_symbols = _code_symbol_names(answer_text)
    shared_symbol = bool(request_symbols and answer_symbols and request_symbols & answer_symbols)
    if (
        category == "documentation"
        and request_symbols
        and _looks_like_docstring_artifact(answer_text)
    ):
        return ResponseAssessment(
            score=0.74,
            accepted=True,
            repaired_answer=exact_token,
            reason="exact_token_docstring_contract_repaired",
            constraint="exact_token",
        )
    if category == "code_refactor" and shared_symbol and _looks_like_code_artifact(answer_text):
        return ResponseAssessment(
            score=0.7,
            accepted=True,
            repaired_answer=exact_token,
            reason="exact_token_refactor_contract_repaired",
            constraint="exact_token",
        )
    return None


def _semantic_label_from_request(intent: Any, request_text: str, labels: list[str]) -> str:
    category = str(getattr(intent, "category", "") or "")
    if (
        category == "code_explanation"
        and str(getattr(intent, "slots", {}).get("style") or "") == "complexity"
    ):
        return _infer_complexity_label_from_request(request_text, labels)
    if category == "test_generation":
        return _infer_framework_label_from_request(request_text, labels)
    if category == "code_fix":
        return _infer_bug_label_from_request(request_text, labels)
    return ""


def _answer_matches_reference(reference: Any, answer_text: str) -> bool:
    answer = " ".join(str(answer_text or "").split()).strip()
    expected = " ".join(str(getattr(reference, "answer", "") or "").split()).strip()
    constraint = str(getattr(reference, "constraint", "") or "")
    if not answer or not expected:
        return False
    if constraint == "json":
        parsed_expected = _extract_json_payload(expected)
        parsed_answer = _extract_json_payload(answer)
        return parsed_expected is not None and parsed_expected == parsed_answer
    if constraint == "label_set":
        return normalize_text(answer).strip(". ") == normalize_text(expected)
    return normalize_text(answer) == normalize_text(expected)


def _infer_complexity_label_from_request(request_text: str, labels: list[str]) -> str:
    return coding_infer_complexity_label_from_request(request_text, labels)


def _infer_framework_label_from_request(request_text: str, labels: list[str]) -> str:
    return coding_infer_framework_label_from_request(request_text, labels)


def _infer_bug_label_from_request(request_text: str, labels: list[str]) -> str:
    return coding_infer_bug_label_from_request(request_text, labels)


def _extract_code_block(request_text: str) -> str:
    return coding_extract_code_block(request_text)


def _infer_complexity_label_from_code_fallback(code: str) -> str:
    lines = [line.rstrip() for line in str(code or "").splitlines() if line.strip()]
    max_depth = 0
    stack: list[int] = []
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1]:
            stack.pop()
        if re.match(r"^\s*(for|while)\b", line):
            stack.append(indent)
            max_depth = max(max_depth, len(stack))
    if max_depth >= 2:
        return "O_N_SQUARED"
    if max_depth == 1:
        return "O_N"
    return "O_1"


def _max_iteration_depth(node: ast.AST, depth: int = 0) -> int:
    current_depth = depth
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        current_depth = depth + 1
    best = current_depth
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        best = max(best, depth + len(getattr(node, "generators", [])))
    for child in ast.iter_child_nodes(node):
        child_depth = (
            current_depth if isinstance(node, (ast.For, ast.AsyncFor, ast.While)) else depth
        )
        best = max(best, _max_iteration_depth(child, child_depth))
    return best


def _pick_label(labels: list[str], canonical_label: str) -> str:
    wanted = str(canonical_label or "").strip().upper()
    for label in labels:
        if str(label or "").strip().upper() == wanted:
            return label
    return ""


def _code_symbol_names(text: str) -> set:
    return {
        match.group("name")
        for match in re.finditer(
            r"(?m)^\s*(?:def|class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)", text or ""
        )
    }


def _looks_like_docstring_artifact(answer_text: str) -> bool:
    lowered = str(answer_text or "").lower()
    if not lowered:
        return False
    if '"""' in lowered or "'''" in lowered:
        return True
    if lowered.strip().startswith(('"', "'", "#")):
        return True
    return any(marker in lowered for marker in ("args:", "returns:", "param", "parameter"))


def _looks_like_code_artifact(answer_text: str) -> bool:
    text = str(answer_text or "")
    lowered = text.lower()
    if "```" in text:
        return True
    if len(text.splitlines()) >= 2 and re.search(
        r"(?m)^\s*(?:def|class|return|if|for|while|import)\b", lowered
    ):
        return True
    return bool(re.search(r"(?m)^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", text))


def _labels_from_slots(slots: dict[str, Any]) -> list[str]:
    labels = str((slots or {}).get("labels") or "")
    if not labels:
        return []
    return [item for item in labels.split("|") if item]


def _fields_from_slots(slots: dict[str, Any]) -> list[str]:
    fields = str((slots or {}).get("fields") or "")
    if not fields:
        return []
    return [item for item in fields.split("|") if item]


def _expected_structured_format(intent, request_text: str) -> str:
    slot_format = str((getattr(intent, "slots", {}) or {}).get("format") or "").strip().lower()
    if slot_format in {"json", "yaml", "csv"}:
        return slot_format
    normalized = (request_text or "").lower()
    if "json" in normalized:
        return "json"
    if "yaml" in normalized:
        return "yaml"
    if "csv" in normalized:
        return "csv"
    return ""


def _extract_json_payload(answer_text: str) -> Any | None:
    candidates = [answer_text.strip()]
    start = answer_text.find("{")
    end = answer_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(answer_text[start : end + 1])
    start = answer_text.find("[")
    end = answer_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(answer_text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_structured_payload(
    answer_text: str,
    structured_format: str,
    *,
    fields: list[str] | None = None,
) -> tuple[Any | None, str | None, float | None]:
    fields = [str(item).strip() for item in (fields or []) if str(item).strip()]
    if structured_format == "json":
        parsed = _extract_json_payload(answer_text)
        if parsed is None:
            return None, None, None
        return (
            parsed,
            json.dumps(parsed, sort_keys=True, separators=(",", ":")),
            _field_coverage(parsed, fields),
        )
    if structured_format == "yaml":
        parsed = _extract_yaml_payload(answer_text)
        if parsed is None:
            return None, None, None
        return parsed, answer_text.strip(), _field_coverage(parsed, fields)
    if structured_format == "csv":
        parsed = _extract_csv_payload(answer_text)
        if parsed is None:
            return None, None, None
        return parsed, answer_text.strip(), _field_coverage(parsed, fields)
    return None, None, None


def _extract_yaml_payload(answer_text: str) -> Any | None:
    try:
        import yaml
    except ImportError:
        return None

    candidates = [answer_text.strip()]
    fence_match = re.search(r"```(?:yaml|yml)?\s*(?P<body>.+?)```", answer_text, flags=re.I | re.S)
    if fence_match:
        candidates.insert(0, fence_match.group("body").strip())
    for candidate in candidates:
        try:
            parsed = yaml.safe_load(candidate)
        except yaml.YAMLError:
            continue
        if parsed not in (None, "", [], {}):
            return parsed
    return None


def _extract_csv_payload(answer_text: str) -> list[dict[str, Any]] | None:
    text = answer_text.strip()
    if not text:
        return None
    fence_match = re.search(r"```(?:csv)?\s*(?P<body>.+?)```", answer_text, flags=re.I | re.S)
    if fence_match:
        text = fence_match.group("body").strip()
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    try:
        reader = DictReader(StringIO("\n".join(lines)))
        rows = [dict(row or {}) for row in reader]
    except csv.Error:
        return None
    if not rows or not reader.fieldnames:
        return None
    return rows


def _field_coverage(parsed: Any, fields: list[str]) -> float | None:
    if not fields:
        return None
    normalized_required = {_normalize_field_name(field) for field in fields if field}
    if not normalized_required:
        return None
    available = _structured_field_names(parsed)
    if not available:
        return 0.0
    matches = sum(1 for field in normalized_required if field in available)
    return matches / max(1, len(normalized_required))


def _structured_field_names(parsed: Any) -> set:
    if isinstance(parsed, dict):
        return {_normalize_field_name(key) for key in parsed.keys()}
    if isinstance(parsed, list) and parsed:
        field_names = set()
        for item in parsed[:5]:
            if isinstance(item, dict):
                field_names.update(_normalize_field_name(key) for key in item.keys())
        return field_names
    return set()


def _normalize_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


__all__ = [
    "OutputContract",
    "_answer_matches_reference",
    "_code_symbol_names",
    "_contains_token",
    "_expected_structured_format",
    "_extract_code_block",
    "_extract_exact_token",
    "_extract_json_payload",
    "_extract_label_candidates",
    "_extract_request_text",
    "_extract_structured_payload",
    "_fields_from_slots",
    "_is_generic_exact_token_candidate",
    "_labels_from_slots",
    "_match_best_label",
    "_repair_exact_token_from_task_output",
    "_semantic_label_from_request",
    "extract_output_contract",
]
