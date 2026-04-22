from __future__ import annotations

import json
import math
import re
from typing import Any

from byte.benchmarking.contracts import BenchmarkItem, OutputContract

BENCHMARK_SCHEMA_VERSION = "2.0"
BENCHMARK_CORPUS_VERSION = "byte-corpus-v2"
BENCHMARK_REPORT_VERSION = "2.0"
BENCHMARK_CONTRACT_VERSION = "byte-benchmark-v2"
BENCHMARK_SCORING_VERSION = "byte-score-v3"
BENCHMARK_TRUST_POLICY_VERSION = "byte-trust-v2"

_NUMBER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")


def validate_item_contract(item: BenchmarkItem) -> dict[str, Any]:
    metadata = dict(item.metadata or {})
    contract_version = str(metadata.get("contract_version", "") or "")
    if contract_version != BENCHMARK_CONTRACT_VERSION:
        raise ValueError(
            f"{item.item_id} has contract version {contract_version or '<missing>'}; "
            f"expected {BENCHMARK_CONTRACT_VERSION}."
        )
    derived_expected = derive_expected_value(item)
    if derived_expected is None:
        raise ValueError(f"{item.item_id} is missing a supported contract recipe.")
    if not expected_values_match(item, derived_expected):
        raise ValueError(
            f"{item.item_id} expected value drifted from contract recipe: "
            f"{item.expected_value!r} != {derived_expected!r}."
        )
    return {
        "contract_version": contract_version,
        "contract_recipe": str(metadata.get("contract_recipe", "") or ""),
        "derived_expected": derived_expected,
    }


def derive_expected_value(item: BenchmarkItem) -> Any:
    metadata = dict(item.metadata or {})
    recipe = str(metadata.get("contract_recipe", "") or "").strip().lower()
    if recipe == "profit_margin":
        return _profit_margin_expected(metadata)
    if recipe == "refund_policy":
        return _refund_policy_expected(metadata)
    if recipe == "json_schema":
        return _json_schema_expected(metadata)
    if recipe == "literal":
        return metadata.get("literal")
    return None


def expected_values_match(item: BenchmarkItem, derived_expected: Any) -> bool:
    if item.output_contract is OutputContract.NUMERIC_TOLERANCE:
        actual = _extract_number(item.expected_value)
        expected = _extract_number(derived_expected)
        if actual is None or expected is None:
            return False
        return math.isclose(actual, expected, abs_tol=0.005)
    if item.output_contract is OutputContract.JSON_SCHEMA:
        return json.dumps(item.expected_value, sort_keys=True) == json.dumps(
            derived_expected, sort_keys=True
        )
    return _normalize_text(item.expected_value) == _normalize_text(derived_expected)


def _profit_margin_expected(metadata: dict[str, Any]) -> str | None:
    try:
        price = float(metadata["price"])
        production = float(metadata["production"])
        marketing = float(metadata["marketing"])
        shipping = float(metadata["shipping"])
    except (KeyError, TypeError, ValueError):
        return None
    if price <= 0:
        return None
    value = ((price - production - marketing - shipping) / price) * 100.0
    rounded = round(float(value or 0.0), 2)
    if abs(rounded - round(rounded)) < 0.005:
        return f"{int(round(rounded))}%"
    return f"{rounded:.2f}".rstrip("0").rstrip(".") + "%"


def _refund_policy_expected(metadata: dict[str, Any]) -> str | None:
    try:
        window = int(metadata["window"])
        day = int(metadata["day"])
    except (KeyError, TypeError, ValueError):
        return None
    approve_label = str(metadata.get("approve_label", "REFUND_APPROVE") or "REFUND_APPROVE")
    deny_label = str(metadata.get("deny_label", "REFUND_DENY") or "REFUND_DENY")
    return approve_label if day <= window else deny_label


def _json_schema_expected(metadata: dict[str, Any]) -> dict[str, Any] | None:
    payload = metadata.get("schema")
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def _extract_number(value: Any) -> float | None:
    match = _NUMBER_PATTERN.search(str(value or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()
