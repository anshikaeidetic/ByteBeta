from dataclasses import dataclass, field
from threading import Lock

from byte.processor.ai_memory import AIMemoryStore
from byte.processor.execution import ExecutionMemoryStore, FailureMemoryStore, PatchPatternStore
from byte.processor.intent import IntentGraph
from byte.processor.optimization_memory import (
    ArtifactMemoryStore,
    PromptPieceStore,
    SessionDeltaStore,
    WorkflowPlanStore,
)
from byte.processor.reasoning_reuse import ReasoningMemoryStore
from byte.processor.tool_result import ToolResultStore
from byte.prompt_distillation import PromptModuleRegistry


@dataclass
class SharedMemoryBundle:
    intent_graph: IntentGraph = field(default_factory=IntentGraph)
    tool_result_store: ToolResultStore = field(default_factory=ToolResultStore)
    ai_memory_store: AIMemoryStore = field(default_factory=AIMemoryStore)
    execution_memory_store: ExecutionMemoryStore = field(default_factory=ExecutionMemoryStore)
    failure_memory_store: FailureMemoryStore = field(default_factory=FailureMemoryStore)
    patch_pattern_store: PatchPatternStore = field(default_factory=PatchPatternStore)
    prompt_piece_store: PromptPieceStore = field(default_factory=PromptPieceStore)
    prompt_module_registry: PromptModuleRegistry = field(default_factory=PromptModuleRegistry)
    artifact_memory_store: ArtifactMemoryStore = field(default_factory=ArtifactMemoryStore)
    workflow_plan_store: WorkflowPlanStore = field(default_factory=WorkflowPlanStore)
    session_delta_store: SessionDeltaStore = field(default_factory=SessionDeltaStore)
    reasoning_memory_store: ReasoningMemoryStore = field(default_factory=ReasoningMemoryStore)


_REGISTRY: dict[str, SharedMemoryBundle] = {}
_REGISTRY_LOCK = Lock()


def get_shared_memory(scope: str | None) -> SharedMemoryBundle | None:
    if not scope:
        return None
    with _REGISTRY_LOCK:
        bundle = _REGISTRY.get(scope)
        if bundle is None:
            bundle = SharedMemoryBundle()
            _REGISTRY[scope] = bundle
        return bundle


def clear_shared_memory(scope: str | None = None) -> None:
    with _REGISTRY_LOCK:
        if scope is None:
            _REGISTRY.clear()
        else:
            _REGISTRY.pop(scope, None)
