from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputContract(str, Enum):
    EXACT_TEXT = "EXACT_TEXT"
    ENUM_LABEL = "ENUM_LABEL"
    NUMERIC_TOLERANCE = "NUMERIC_TOLERANCE"
    JSON_SCHEMA = "JSON_SCHEMA"
    WORKFLOW_ACTION = "WORKFLOW_ACTION"
    FALLBACK_EXPECTED = "FALLBACK_EXPECTED"


class RunPhase(str, Enum):
    COLD = "cold"
    WARM_100 = "warm_100"
    WARM_1000 = "warm_1000"


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    provider_track: str
    family: str
    scenario: str
    seed_id: str
    variant_id: str
    input_payload: dict[str, Any]
    output_contract: OutputContract
    expected_value: Any
    tolerance: float | None = None
    reuse_safe: bool = False
    must_fallback: bool = False
    tags: tuple[str, ...] = ()
    deterministic_expected: bool = False
    workflow_total_steps: int = 0
    model_hint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, payload: dict[str, Any]) -> BenchmarkItem:
        return cls(
            item_id=str(payload["item_id"]),
            provider_track=str(payload["provider_track"]),
            family=str(payload["family"]),
            scenario=str(payload["scenario"]),
            seed_id=str(payload["seed_id"]),
            variant_id=str(payload["variant_id"]),
            input_payload=dict(payload.get("input_payload", {}) or {}),
            output_contract=OutputContract(str(payload["output_contract"])),
            expected_value=payload.get("expected_value"),
            tolerance=(
                None
                if payload.get("tolerance") in (None, "")
                else float(payload.get("tolerance") or 0.0)
            ),
            reuse_safe=bool(payload.get("reuse_safe", False)),
            must_fallback=bool(payload.get("must_fallback", False)),
            tags=tuple(str(tag) for tag in (payload.get("tags") or [])),
            deterministic_expected=bool(payload.get("deterministic_expected", False)),
            workflow_total_steps=int(payload.get("workflow_total_steps", 0) or 0),
            model_hint=str(payload.get("model_hint", "") or ""),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "provider_track": self.provider_track,
            "family": self.family,
            "scenario": self.scenario,
            "seed_id": self.seed_id,
            "variant_id": self.variant_id,
            "input_payload": dict(self.input_payload),
            "output_contract": self.output_contract.value,
            "expected_value": self.expected_value,
            "tolerance": self.tolerance,
            "reuse_safe": self.reuse_safe,
            "must_fallback": self.must_fallback,
            "tags": list(self.tags),
            "deterministic_expected": self.deterministic_expected,
            "workflow_total_steps": self.workflow_total_steps,
            "model_hint": self.model_hint,
            "metadata": dict(self.metadata),
        }
