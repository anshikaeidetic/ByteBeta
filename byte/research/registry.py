"""Machine-readable registry of research references that Byte maps to product features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchArtifact:
    artifact_id: str
    title: str
    source_url: str
    area: str
    claimed_gains: list[str]
    supported_backends: list[str]
    implementation_status: str
    reproducibility_status: str
    benchmark_parity: str
    production_risks: list[str]
    license_review: str = "reference_only_pending_review"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "title": self.title,
            "source_url": self.source_url,
            "area": self.area,
            "claimed_gains": list(self.claimed_gains),
            "supported_backends": list(self.supported_backends),
            "implementation_status": self.implementation_status,
            "reproducibility_status": self.reproducibility_status,
            "benchmark_parity": self.benchmark_parity,
            "production_risks": list(self.production_risks),
            "license_review": self.license_review,
        }


_REGISTRY = [
    ResearchArtifact(
        artifact_id="prompt-cache-2311.04934",
        title="Prompt Cache: Modular Attention Reuse for Low-Latency Inference",
        source_url="https://arxiv.org/abs/2311.04934",
        area="prompt_distillation",
        claimed_gains=["modular prompt reuse", "prefill reuse", "lower time-to-first-token"],
        supported_backends=["byte_gateway", "huggingface_local", "hosted_provider_bridge"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_adaptation_complete",
        benchmark_parity="pending_prompt_distillation_matrix",
        production_risks=["module boundary drift", "cache-key instability without versioning"],
    ),
    ResearchArtifact(
        artifact_id="longllmlingua-2310.06839",
        title="LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression",
        source_url="https://arxiv.org/abs/2310.06839",
        area="prompt_distillation",
        claimed_gains=["long-context prompt compression", "query-aware pruning", "token budget control"],
        supported_backends=["byte_gateway", "hosted_provider_bridge"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_adaptation_complete",
        benchmark_parity="pending_prompt_distillation_matrix",
        production_risks=["compression faithfulness drift", "structured contract loss"],
    ),
    ResearchArtifact(
        artifact_id="llmlingua2-2403.12968",
        title="LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression",
        source_url="https://arxiv.org/abs/2403.12968",
        area="prompt_distillation",
        claimed_gains=["task-agnostic prompt compression", "faithful token pruning", "distilled compressor models"],
        supported_backends=["byte_gateway", "huggingface_local"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_adaptation_complete",
        benchmark_parity="pending_prompt_distillation_matrix",
        production_risks=["compressor artifact calibration", "fallback thresholds on unseen tasks"],
    ),
    ResearchArtifact(
        artifact_id="selective-context-2310.06201",
        title="Compressing Context to Enhance Inference Efficiency of Large Language Models",
        source_url="https://arxiv.org/abs/2310.06201",
        area="prompt_distillation",
        claimed_gains=["selective context pruning", "lower inference memory", "lower latency"],
        supported_backends=["byte_gateway", "hosted_provider_bridge"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_adaptation_complete",
        benchmark_parity="pending_prompt_distillation_matrix",
        production_risks=["dropping rare but critical evidence"],
    ),
    ResearchArtifact(
        artifact_id="recomp-2310.04408",
        title="RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation",
        source_url="https://arxiv.org/abs/2310.04408",
        area="retrieval_compression",
        claimed_gains=["retrieval compression", "selective augmentation", "smaller prompt payloads"],
        supported_backends=["byte_gateway", "hosted_provider_bridge"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_adaptation_complete",
        benchmark_parity="pending_prompt_distillation_matrix",
        production_risks=["evidence omission under weak retrieval ranking"],
    ),
    ResearchArtifact(
        artifact_id="vcache-2502.03771",
        title="vCache: Verified Semantic Prompt Caching",
        source_url="https://arxiv.org/abs/2502.03771",
        area="semantic_caching",
        claimed_gains=[
            "per-prompt adaptive thresholds",
            "verified semantic caching with user-defined error guarantees",
            "higher cache hit rates at bounded error",
        ],
        supported_backends=["byte_gateway", "hosted_provider_bridge"],
        implementation_status="implemented",
        reproducibility_status="source_audited_byte_owned_adaptation_needs_public_release_lane",
        benchmark_parity="pending_verified_semantic_cache_matrix",
        production_risks=[
            "adaptive threshold drift by workload family",
            "guarantee quality depends on calibration data quality",
        ],
    ),
    ResearchArtifact(
        artifact_id="promptintern-2407.02211",
        title="PromptIntern: Saving Inference Costs by Internalizing Recurrent Prompt during Large Language Model Fine-tuning",
        source_url="https://arxiv.org/abs/2407.02211",
        area="prompt_distillation",
        claimed_gains=["recurrent prompt internalization", "offline prompt absorption", "lower input-token cost"],
        supported_backends=["huggingface_local"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_adaptation_complete",
        benchmark_parity="pending_prompt_distillation_matrix",
        production_risks=["artifact drift across model families", "offline-only portability limits"],
    ),
    ResearchArtifact(
        artifact_id="qjl-2406.03482",
        title="QJL: 1-Bit Quantized JL Transform for KV Cache Quantization",
        source_url="https://arxiv.org/abs/2406.03482",
        area="kv_quantization",
        claimed_gains=["1-bit sign sketches", "inner-product preservation", "projection-based compression"],
        supported_backends=["huggingface_local", "byte_vector_sidecar"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_reference_complete",
        benchmark_parity="pending_full_matrix",
        production_risks=["approximate reconstruction quality", "calibration under long tails"],
    ),
    ResearchArtifact(
        artifact_id="polarquant-2502.02617",
        title="PolarQuant",
        source_url="https://arxiv.org/abs/2502.02617",
        area="kv_quantization",
        claimed_gains=["polar transform quantization", "long-context compression", "low-bit angular coding"],
        supported_backends=["huggingface_local"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_reference_complete",
        benchmark_parity="pending_full_matrix",
        production_risks=["high-dimensional angle stability", "decode overhead"],
    ),
    ResearchArtifact(
        artifact_id="turboquant-2504.19874",
        title="TurboQuant",
        source_url="https://arxiv.org/abs/2504.19874",
        area="vector_and_kv_quantization",
        claimed_gains=["online vector quantization", "near-optimal distortion rate", "residual quantization"],
        supported_backends=["huggingface_local", "byte_vector_sidecar", "byte_memory_indices"],
        implementation_status="implemented",
        reproducibility_status="byte_owned_reference_complete",
        benchmark_parity="pending_full_matrix",
        production_risks=["residual drift under repeated updates"],
    ),
    ResearchArtifact(
        artifact_id="mlsys-kv-survey-2025",
        title="Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving",
        source_url="https://proceedings.mlsys.org/paper_files/paper/2025/file/26289c647c6828e862e271ca3c490486-Paper-Conference.pdf",
        area="survey",
        claimed_gains=["taxonomy for kv compression", "comparative tradeoff framing", "production evaluation axes"],
        supported_backends=["huggingface_local", "byte_gateway"],
        implementation_status="registered",
        reproducibility_status="survey",
        benchmark_parity="not_applicable",
        production_risks=["survey guidance must be validated per backend"],
    ),
]


def research_registry() -> list[dict[str, Any]]:
    """Return the audited registry as a JSON-serializable payload."""
    return [item.to_dict() for item in _REGISTRY]


def research_registry_summary() -> dict[str, Any]:
    """Return registry counts plus the current production-governance policy."""
    implemented = sum(1 for item in _REGISTRY if item.implementation_status == "implemented")
    return {
        "artifacts": research_registry(),
        "total_artifacts": len(_REGISTRY),
        "implemented_artifacts": implemented,
        "production_gate": "shadow_then_guarded_then_enabled",
        "license_policy": "byte_owned_implementation_reference_validation_only",
    }
