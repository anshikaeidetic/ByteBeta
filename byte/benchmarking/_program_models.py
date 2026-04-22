"""Typed primitives shared by benchmark entrypoints and executors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkProgramSpec:
    """Describe a benchmark program without importing its execution stack."""

    name: str
    implementation_module: str
    description: str


@dataclass(frozen=True)
class RequestCase:
    """Provider-agnostic request payload used by benchmark scenario builders."""

    case_id: str
    prompt: str
    expected: str
    group: str
    variant: str
    kind: str
    max_tokens: int = 12
    request_overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkScenario:
    """Deterministic benchmark scenario and the requests that belong to it."""

    scenario_id: str
    name: str
    provider: str
    cases: tuple[RequestCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime execution settings shared across live benchmark runners."""

    provider: str
    model: str
    concurrency: int = 1
    attempts: int = 3
    base_sleep_seconds: float = 1.0
    phase: str = "default"


@dataclass(frozen=True)
class ExecutionRecord:
    """Normalized record emitted by provider and Byte execution paths."""

    case_id: str
    provider: str
    phase: str
    ok: bool
    latency_ms: float
    response_text: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgramReport:
    """In-memory program report before serialization to benchmark artifacts."""

    program: str
    generated_at: str
    records: tuple[ExecutionRecord, ...]
    summary: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "BenchmarkProgramSpec",
    "BenchmarkScenario",
    "ExecutionConfig",
    "ExecutionRecord",
    "ProgramReport",
    "RequestCase",
]
