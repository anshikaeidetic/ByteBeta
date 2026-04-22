"""Pydantic request and response models for Byte server routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CacheData(BaseModel):
    prompt: str
    answer: str | None = ""


class MCPToolRegistration(BaseModel):
    server_name: str
    tool_name: str
    endpoint: str
    method: str = "POST"
    input_schema: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    cache_policy: str = "read_only"
    ttl: float | None = None
    headers: dict[str, Any] = Field(default_factory=dict)


class MCPToolCall(BaseModel):
    server_name: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    scope: str = ""
    timeout_s: float = 30.0


class WarmData(BaseModel):
    data: list[Any]


class FeedbackData(BaseModel):
    query: str
    thumbs_up: bool


class MemoryImportData(BaseModel):
    snapshot: dict[str, Any]


class MemoryArtifactExportData(BaseModel):
    path: str
    format: str | None = None
    limit: int | None = None


class MemoryArtifactImportData(BaseModel):
    path: str
    format: str | None = None


class ScopeData(BaseModel):
    tenant: str
    session: str
    workflow: str
    source: str = "derived"

    @property
    def scope_key(self) -> str:
        return f"{self.tenant}:{self.session}:{self.workflow}"


class WorkerHeartbeat(BaseModel):
    worker_id: str
    url: str = ""
    status: str = "ready"
    queue_depth: int = 0
    free_vram_gb: float = 0.0
    model_inventory: list[str] = Field(default_factory=list)
    health_score: float = 1.0
    last_error: str = ""
    last_heartbeat: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerLease(BaseModel):
    tenant: str
    session_id: str
    model_family: str
    worker_id: str
    updated_at: float


class KVResidencyRecord(BaseModel):
    tenant: str
    session_id: str
    model_name: str
    compression_profile: str
    worker_id: str
    prompt_digest: str
    updated_at: float


class ReplayJob(BaseModel):
    id: int
    tenant: str
    session_id: str
    workflow_id: str
    route_key: str
    feature: str
    status: str
    request_digest: str
    projected_savings: float
    created_at: float


class ReplayOutcome(BaseModel):
    job_id: int
    deterministic_score: float
    evidence_score: float
    error_rate: float
    error_rate_upper: float
    judge_required: bool
    projected_savings: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float


class ThresholdRecommendation(BaseModel):
    tenant: str
    route_key: str
    feature: str
    suggested_threshold: float
    sample_count: int
    projected_savings: float
    error_rate: float
    error_rate_upper: float
    auto_apply_eligible: bool
    details: dict[str, Any] = Field(default_factory=dict)
    updated_at: float


class CacheInspectionEvent(BaseModel):
    tenant: str
    session_id: str
    workflow_id: str
    route_key: str
    event_type: str
    reason: str
    cache_hit: bool
    latency_ms: float
    worker_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float


class IntentGraphEdge(BaseModel):
    tenant: str
    source_intent: str
    target_intent: str
    edge_count: int
    updated_at: float


class ControlPlaneSettingsUpdate(BaseModel):
    worker_urls: list[str] | None = None
    memory_service_url: str | None = None
    replay_enabled: bool | None = None
    replay_sample_rate: float | None = None


class InferenceGenerateRequest(BaseModel):
    scope: ScopeData
    request: dict[str, Any] = Field(default_factory=dict)
    selection_source: str = ""


class InferencePrefillRequest(BaseModel):
    scope: ScopeData
    request: dict[str, Any] = Field(default_factory=dict)


class KVEventData(BaseModel):
    worker_id: str
    event_type: str
    scope: ScopeData | None = None
    model_name: str = ""
    compression_profile: str = ""
    prompt_digest: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResolveRequest(BaseModel):
    scope: ScopeData
    request: dict[str, Any] = Field(default_factory=dict)
    provider_mode: str = "hosted"


class MemoryRememberRequest(BaseModel):
    scope: ScopeData
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] | str | None = None
    provider_mode: str = "hosted"
    worker_id: str = ""
