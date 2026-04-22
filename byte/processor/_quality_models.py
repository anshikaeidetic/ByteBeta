"""Shared dataclasses for quality scoring and validation."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResponseAssessment:
    score: float
    accepted: bool
    repaired_answer: str | None
    reason: str
    constraint: str


@dataclass(frozen=True)
class EvidenceAssessment:
    score: float
    accepted: bool
    reason: str
    constraint: str


@dataclass(frozen=True)
class OutputContract:
    category: str
    exact_token: str = ""
    labels: tuple[str, ...] = ()
    structured_format: str = ""
    strict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "exact_token": self.exact_token,
            "labels": list(self.labels),
            "structured_format": self.structured_format,
            "strict": self.strict,
        }


__all__ = ["EvidenceAssessment", "OutputContract", "ResponseAssessment"]
