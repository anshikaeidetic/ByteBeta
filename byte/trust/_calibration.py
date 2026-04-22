"""Versioned calibration artifact loading for Byte trust decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

DEFAULT_CALIBRATION_VERSION = "byte-trust-v2"


@dataclass(frozen=True)
class CalibrationBucket:
    """A monotonic raw-score to calibrated-score mapping bucket."""

    threshold: float
    mapped: float


@dataclass(frozen=True)
class TrustCalibration:
    """Loaded trust calibration data with validated provenance metadata."""

    version: str
    status: str
    method: str
    source_dataset: str
    source_status: str
    public_proof_status: str
    public_proof_manifest: str
    checksum: str
    computed_checksum: str
    buckets: tuple[CalibrationBucket, ...]
    confidence_scores: dict[str, float]
    confidence_thresholds: dict[str, float]
    confidence_adjustments: dict[str, float]
    risk_scores: dict[str, float]
    risk_thresholds: dict[str, float]
    reference_scores: dict[str, float]
    reference_thresholds: dict[str, float]
    validation_metrics: dict[str, float]

    @property
    def checksum_valid(self) -> bool:
        """Return whether the artifact checksum matches its canonical payload."""

        return self.checksum == self.computed_checksum

    def confidence_score(self, name: str) -> float:
        """Return a named confidence score from the calibration artifact."""

        return _lookup_float(self.confidence_scores, name, group="confidence_scores")

    def confidence_threshold(self, name: str) -> float:
        """Return a named confidence threshold from the calibration artifact."""

        return _lookup_float(self.confidence_thresholds, name, group="confidence_thresholds")

    def confidence_adjustment(self, name: str) -> float:
        """Return a named confidence adjustment from the calibration artifact."""

        return _lookup_float(self.confidence_adjustments, name, group="confidence_adjustments")

    def risk_score(self, name: str) -> float:
        """Return a named risk score from the calibration artifact."""

        return _lookup_float(self.risk_scores, name, group="risk_scores")

    def risk_threshold(self, name: str) -> float:
        """Return a named risk threshold from the calibration artifact."""

        return _lookup_float(self.risk_thresholds, name, group="risk_thresholds")

    def reference_score(self, name: str) -> float:
        """Return a named deterministic-reference score from the artifact."""

        return _lookup_float(self.reference_scores, name, group="reference_scores")

    def reference_threshold(self, name: str) -> float:
        """Return a named deterministic-reference threshold from the artifact."""

        return _lookup_float(self.reference_thresholds, name, group="reference_thresholds")


def load_trust_calibration(version: str | None = None) -> TrustCalibration:
    """Load the requested trust calibration artifact."""

    requested = str(version or DEFAULT_CALIBRATION_VERSION).strip() or DEFAULT_CALIBRATION_VERSION
    if requested != DEFAULT_CALIBRATION_VERSION:
        requested = DEFAULT_CALIBRATION_VERSION
    return _load_default_calibration(requested)


@lru_cache(maxsize=4)
def _load_default_calibration(version: str) -> TrustCalibration:
    payload = json.loads(_artifact_text(version))
    computed_checksum = calibration_checksum(payload)
    buckets = tuple(
        CalibrationBucket(
            threshold=float(item["threshold"]),
            mapped=float(item["mapped"]),
        )
        for item in payload.get("buckets", [])
    )
    return TrustCalibration(
        version=str(payload["version"]),
        status=str(payload["status"]),
        method=str(payload["method"]),
        source_dataset=str(payload["source_dataset"]),
        source_status=str(payload.get("source_status", "")),
        public_proof_status=str(payload.get("public_proof_status", "")),
        public_proof_manifest=str(payload.get("public_proof_manifest", "")),
        checksum=str(payload.get("checksum", "")),
        computed_checksum=computed_checksum,
        buckets=buckets,
        confidence_scores=_float_mapping(payload.get("confidence_scores", {})),
        confidence_thresholds=_float_mapping(payload.get("confidence_thresholds", {})),
        confidence_adjustments=_float_mapping(payload.get("confidence_adjustments", {})),
        risk_scores=_float_mapping(payload.get("risk_scores", {})),
        risk_thresholds=_float_mapping(payload.get("risk_thresholds", {})),
        reference_scores=_float_mapping(payload.get("reference_scores", {})),
        reference_thresholds=_float_mapping(payload.get("reference_thresholds", {})),
        validation_metrics=_float_mapping(payload.get("validation_metrics", {})),
    )


def calibration_checksum(payload: dict[str, Any]) -> str:
    """Return the canonical SHA-256 checksum for a calibration payload."""

    canonical_payload = dict(payload)
    canonical_payload.pop("checksum", None)
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_text(version: str) -> str:
    artifact = resources.files("byte.trust").joinpath("calibration").joinpath(f"{version}.json")
    return artifact.read_text(encoding="utf-8")


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _lookup_float(mapping: dict[str, float], name: str, *, group: str) -> float:
    try:
        return mapping[name]
    except KeyError as exc:
        raise KeyError(f"missing trust calibration value {group}.{name}") from exc


__all__ = [
    "DEFAULT_CALIBRATION_VERSION",
    "CalibrationBucket",
    "TrustCalibration",
    "calibration_checksum",
    "load_trust_calibration",
]
