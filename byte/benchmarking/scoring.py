from __future__ import annotations

import json
import math
import re
from typing import Any

from byte.benchmarking.contracts import BenchmarkItem, OutputContract


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def score_output(
    item: BenchmarkItem,
    response_text: str,
    *,
    fallback_taken: bool = False,
) -> bool:
    if item.output_contract is OutputContract.EXACT_TEXT:
        return normalize_text(response_text) == normalize_text(item.expected_value)
    if item.output_contract in {OutputContract.ENUM_LABEL, OutputContract.WORKFLOW_ACTION}:
        return normalize_text(response_text) == normalize_text(item.expected_value)
    if item.output_contract is OutputContract.NUMERIC_TOLERANCE:
        return _score_numeric(item, response_text)
    if item.output_contract is OutputContract.JSON_SCHEMA:
        return _score_json_schema(item, response_text)
    if item.output_contract is OutputContract.FALLBACK_EXPECTED:
        return bool(fallback_taken) == bool(item.expected_value)
    return False


def score_policy_adherence(item: BenchmarkItem, response_text: str) -> bool:
    if item.output_contract is OutputContract.JSON_SCHEMA:
        try:
            value = json.loads(response_text)
        except json.JSONDecodeError:
            return False
        return isinstance(value, dict)
    if item.output_contract in {
        OutputContract.EXACT_TEXT,
        OutputContract.ENUM_LABEL,
        OutputContract.WORKFLOW_ACTION,
    }:
        return normalize_text(response_text) == normalize_text(item.expected_value)
    if item.output_contract is OutputContract.NUMERIC_TOLERANCE:
        return _extract_number(response_text) is not None
    if item.output_contract is OutputContract.FALLBACK_EXPECTED:
        return True
    return False


def canonical_output(item: BenchmarkItem, response_text: str) -> str:
    if item.output_contract is OutputContract.JSON_SCHEMA:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return normalize_text(response_text)
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    if item.output_contract is OutputContract.NUMERIC_TOLERANCE:
        value = _extract_number(response_text)
        return "" if value is None else f"{value:.4f}"
    return normalize_text(response_text)


def _score_numeric(item: BenchmarkItem, response_text: str) -> bool:
    actual = _extract_number(response_text)
    expected = _extract_number(item.expected_value)
    if actual is None or expected is None:
        return False
    tolerance = float(item.tolerance if item.tolerance is not None else 0.0)
    return math.isclose(actual, expected, abs_tol=tolerance)


def _score_json_schema(item: BenchmarkItem, response_text: str) -> bool:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return False
    schema = item.expected_value if isinstance(item.expected_value, dict) else {}
    return _basic_json_schema_match(payload, schema)


def _basic_json_schema_match(value: Any, schema: dict[str, Any]) -> bool:
    if not schema:
        return isinstance(value, dict)
    schema_type = str(schema.get("type", "") or "")
    if schema_type == "object":
        if not isinstance(value, dict):
            return False
        required = list(schema.get("required", []) or [])
        properties = dict(schema.get("properties", {}) or {})
        if any(key not in value for key in required):
            return False
        for key, property_schema in properties.items():
            if key in value and not _basic_json_schema_match(value[key], property_schema):
                return False
        return True
    if schema_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = dict(schema.get("items", {}) or {})
        return all(_basic_json_schema_match(item, item_schema) for item in value)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    return True


def _extract_number(value: Any) -> float | None:
    text = str(value or "")
    match = re.search(r"(-?\d+(?:\.\d+)?)", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
