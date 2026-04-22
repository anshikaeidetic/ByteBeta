import ast
import importlib
from pathlib import Path
from unittest.mock import patch

from byte import Cache, Config
from byte._devtools.verification_targets import CORE_HOTSPOT_TARGETS
from byte.config import CacheConfig, RoutingConfig
from byte.manager import manager_factory
from byte.manager.data_manager import DataManager
from byte.manager.scalar_data.base import CacheData
from byte.mcp_gateway import MCPGateway
from byte.similarity_evaluation.numpy_similarity import NumpyNormEvaluation

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVIDER_WRAPPERS = (
    "anthropic",
    "bedrock",
    "cohere",
    "deepseek",
    "gemini",
    "groq",
    "huggingface",
    "llama_cpp",
    "mistral",
    "ollama",
    "openai",
    "openrouter",
)
MODULE_LIMITS = {
    REPO_ROOT / "byte" / "adapter" / "adapter.py": 15,
    REPO_ROOT / "byte" / "adapter" / "runtime.py": 400,
    REPO_ROOT / "byte" / "adapter" / "_runtime_support.py": 900,
    REPO_ROOT / "byte" / "adapter" / "_runtime_response.py": 900,
    REPO_ROOT / "byte" / "adapter" / "_runtime_verification.py": 900,
    REPO_ROOT / "byte" / "adapter" / "_runtime_memory.py": 900,
    REPO_ROOT / "byte" / "adapter" / "_runtime_sync.py": 900,
    REPO_ROOT / "byte" / "adapter" / "_runtime_async.py": 900,
    REPO_ROOT / "byte" / "processor" / "pre.py": 300,
    REPO_ROOT / "byte" / "processor" / "_pre_canonicalize.py": 900,
    REPO_ROOT / "byte" / "processor" / "_pre_context.py": 900,
    REPO_ROOT / "byte" / "processor" / "_pre_accessors.py": 900,
    REPO_ROOT / "byte" / "processor" / "_pre_relevance.py": 900,
    REPO_ROOT / "byte" / "processor" / "_pre_selection.py": 900,
    REPO_ROOT / "byte" / "processor" / "optimization_memory.py": 300,
    REPO_ROOT / "byte" / "processor" / "_optimization_summary.py": 250,
    REPO_ROOT / "byte" / "processor" / "_optimization_stores.py": 250,
    REPO_ROOT / "byte" / "processor" / "_optimization_common.py": 900,
    REPO_ROOT / "byte" / "processor" / "quality.py": 300,
    REPO_ROOT / "byte" / "processor" / "_quality_contracts.py": 900,
    REPO_ROOT / "byte" / "processor" / "_quality_evidence.py": 900,
    REPO_ROOT / "byte" / "processor" / "_quality_models.py": 900,
    REPO_ROOT / "byte" / "processor" / "_quality_scorer.py": 900,
    REPO_ROOT / "byte" / "processor" / "reasoning_reuse.py": 300,
    REPO_ROOT / "byte" / "processor" / "_reasoning_patterns.py": 900,
    REPO_ROOT / "byte" / "processor" / "_reasoning_shortcuts.py": 900,
    REPO_ROOT / "byte" / "processor" / "_reasoning_store.py": 900,
    REPO_ROOT / "byte" / "h2o" / "runtime.py": 300,
    REPO_ROOT / "byte" / "h2o" / "_runtime_common.py": 900,
    REPO_ROOT / "byte" / "h2o" / "_runtime_engine.py": 250,
    REPO_ROOT / "byte" / "h2o" / "_runtime_kv.py": 900,
    REPO_ROOT / "byte" / "prompt_distillation" / "core.py": 300,
    REPO_ROOT / "byte" / "prompt_distillation" / "_distillation_common.py": 900,
    REPO_ROOT / "byte" / "prompt_distillation" / "_distillation_engine.py": 900,
    REPO_ROOT / "byte" / "prompt_distillation" / "_distillation_faithfulness.py": 900,
    REPO_ROOT / "byte" / "prompt_distillation" / "_distillation_measure.py": 900,
    REPO_ROOT / "byte_server" / "server.py": 600,
    REPO_ROOT / "byte_server" / "_server_security.py": 900,
    REPO_ROOT / "byte_server" / "_server_gateway.py": 900,
    REPO_ROOT / "byte_server" / "_server_routes_cache.py": 900,
    REPO_ROOT / "byte_server" / "_server_routes_proxy.py": 900,
    REPO_ROOT / "byte_server" / "_server_routes_chat.py": 900,
    REPO_ROOT / "byte_server" / "_server_routes_mcp.py": 900,
}
MODULE_LIMITS.update(
    {
        REPO_ROOT / "byte" / "_backends" / "gemini.py": 250,
        REPO_ROOT / "byte" / "_backends" / "gemini_audio.py": 400,
        REPO_ROOT / "byte" / "_backends" / "gemini_chat.py": 400,
        REPO_ROOT / "byte" / "_backends" / "gemini_clients.py": 250,
        REPO_ROOT / "byte" / "_backends" / "gemini_images.py": 250,
        REPO_ROOT / "byte" / "_backends" / "gemini_messages.py": 250,
        REPO_ROOT / "byte" / "_backends" / "gemini_responses.py": 400,
        REPO_ROOT / "byte" / "_core_memory.py": 250,
        REPO_ROOT / "byte" / "_core_memory_execution.py": 250,
        REPO_ROOT / "byte" / "_core_memory_interactions.py": 250,
        REPO_ROOT / "byte" / "_core_memory_optimization.py": 250,
        REPO_ROOT / "byte" / "_core_memory_patterns.py": 250,
        REPO_ROOT / "byte" / "_core_memory_reasoning.py": 250,
        REPO_ROOT / "byte" / "_core_memory_snapshot.py": 250,
        REPO_ROOT / "byte" / "_core_memory_tools.py": 250,
        REPO_ROOT / "byte" / "adapter" / "pipeline" / "_response_assessment.py": 300,
        REPO_ROOT / "byte" / "adapter" / "pipeline" / "_response_commit.py": 250,
        REPO_ROOT / "byte" / "adapter" / "pipeline" / "_response_finalize.py": 250,
        REPO_ROOT / "byte" / "adapter" / "pipeline" / "_response_streaming.py": 350,
        REPO_ROOT / "byte" / "h2o" / "_runtime_decode.py": 250,
        REPO_ROOT / "byte" / "h2o" / "_runtime_factory.py": 250,
        REPO_ROOT / "byte" / "h2o" / "_runtime_generation.py": 400,
        REPO_ROOT / "byte" / "h2o" / "_runtime_runtime.py": 250,
        REPO_ROOT / "byte" / "h2o" / "_runtime_tokens.py": 250,
        REPO_ROOT / "byte" / "processor" / "_model_router_policy.py": 450,
        REPO_ROOT / "byte" / "processor" / "_model_router_signals.py": 250,
        REPO_ROOT / "byte" / "processor" / "_model_router_tracker.py": 250,
        REPO_ROOT / "byte" / "processor" / "_model_router_types.py": 250,
        REPO_ROOT / "byte" / "processor" / "_optimization_artifact_store.py": 400,
        REPO_ROOT / "byte" / "processor" / "_optimization_artifacts.py": 400,
        REPO_ROOT / "byte" / "processor" / "_optimization_focus.py": 400,
        REPO_ROOT / "byte" / "processor" / "_optimization_prompt_pieces.py": 250,
        REPO_ROOT / "byte" / "processor" / "_optimization_prompt_store.py": 250,
        REPO_ROOT / "byte" / "processor" / "_optimization_public.py": 250,
        REPO_ROOT / "byte" / "processor" / "_optimization_session_store.py": 250,
        REPO_ROOT / "byte" / "processor" / "_optimization_text.py": 250,
        REPO_ROOT / "byte" / "processor" / "_optimization_workflow_store.py": 400,
        REPO_ROOT / "byte" / "processor" / "model_router.py": 250,
    }
)
PUBLIC_FACADE_MODULES = {
    REPO_ROOT / "byte" / "adapter" / "adapter.py",
    REPO_ROOT / "byte" / "adapter" / "runtime.py",
    REPO_ROOT / "byte" / "processor" / "pre.py",
    REPO_ROOT / "byte" / "processor" / "optimization_memory.py",
    REPO_ROOT / "byte" / "processor" / "quality.py",
    REPO_ROOT / "byte" / "processor" / "reasoning_reuse.py",
    REPO_ROOT / "byte" / "h2o" / "runtime.py",
    REPO_ROOT / "byte" / "prompt_distillation" / "core.py",
    REPO_ROOT / "byte_server" / "server.py",
}
RUNTIME_HELPER_FACADES = {
    REPO_ROOT / "byte" / "adapter" / "_runtime_async.py",
    REPO_ROOT / "byte" / "adapter" / "_runtime_memory.py",
    REPO_ROOT / "byte" / "adapter" / "_runtime_response.py",
    REPO_ROOT / "byte" / "adapter" / "_runtime_support.py",
    REPO_ROOT / "byte" / "adapter" / "_runtime_sync.py",
    REPO_ROOT / "byte" / "adapter" / "_runtime_verification.py",
}
WRAPPER_MODULE_PATHS = {
    REPO_ROOT / "byte" / "adapter" / f"{provider_name}.py"
    for provider_name in PROVIDER_WRAPPERS
}
STRICT_EXPORT_MODULES = PUBLIC_FACADE_MODULES | WRAPPER_MODULE_PATHS
EXPLICIT_FACADE_MODULES = STRICT_EXPORT_MODULES | RUNTIME_HELPER_FACADES


class _InvalidateTrackingManager(DataManager):
    def __init__(self) -> None:
        self.invalidated = []

    def save(self, question, answer, embedding_data, **kwargs) -> None:
        return None

    def import_data(self, questions, answers, embedding_datas, session_ids) -> None:
        return None

    def get_scalar_data(self, res_data, **kwargs) -> object:
        return CacheData(question="q", answers=["a"])

    def search(self, embedding_data, **kwargs) -> object:
        return []

    def invalidate_by_query(self, query: str, *, embedding_func=None) -> bool:
        self.invalidated.append((query, embedding_func("probe")))
        return query == "remove-me"

    def add_session(self, res_data, session_id, pre_embedding_data) -> None:
        return None

    def list_sessions(self, session_id, key) -> object:
        return []

    def delete_session(self, session_id) -> None:
        return None

    def close(self) -> None:
        return None


class _HTTPResponse:
    def __init__(self, payload) -> None:
        self._payload = payload
        self.headers = {"content-type": "application/json"}
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def _module_tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_dunder_all(path: Path) -> list[str]:
    module_tree = _module_tree(path)
    for node in module_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        assert isinstance(value, (list, tuple)), path.as_posix()
        assert all(isinstance(name, str) for name in value), path.as_posix()
        return list(value)
    raise AssertionError(f"{path.as_posix()} must define a literal __all__")


def test_config_supports_group_sections_and_flat_fields() -> None:
    cfg = Config(
        cache=CacheConfig(similarity_threshold=0.91),
        routing=RoutingConfig(model_routing=True, routing_cheap_model="cheap"),
        context_compiler=False,
        context_compiler_keep_last_messages=3,
        mcp_timeout_s=12.5,
    )

    assert cfg.similarity_threshold == 0.91
    assert cfg.cache.similarity_threshold == 0.91
    assert cfg.context_compiler is False
    assert cfg.context_compiler_config.context_compiler is False
    assert cfg.context_compiler_config.context_compiler_keep_last_messages == 3
    assert cfg.routing.model_routing is True
    assert cfg.routing.routing_cheap_model == "cheap"
    assert cfg.mcp_timeout_s == 12.5
    assert cfg.integrations.mcp_timeout_s == 12.5


def test_config_from_env_loads_grouped_overrides(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_MCP_TIMEOUT_S", "14.5")
    monkeypatch.setenv("BYTE_SKIP_LIST", '["system"]')
    monkeypatch.setenv("BYTE_ROUTING_FALLBACKS", '{"primary":["fallback-a","fallback-b"]}')

    cfg = Config.from_env()

    assert cfg.mcp_timeout_s == 14.5
    assert cfg.skip_list == ["system"]
    assert cfg.routing_fallbacks == {"primary": ["fallback-a", "fallback-b"]}


def test_cache_stays_lazy_until_memory_features_are_used() -> None:
    cache_obj = Cache()

    assert cache_obj.intent_graph is None
    assert cache_obj.tool_result_store is None

    cache_obj.record_intent({"messages": [{"role": "user", "content": "hello"}]})

    assert cache_obj.intent_graph is not None
    assert cache_obj.tool_result_store is not None


def test_cache_invalidate_by_query_uses_data_manager_contract() -> None:
    manager = _InvalidateTrackingManager()
    cache_obj = Cache()
    cache_obj.init(
        data_manager=manager,
        embedding_func=lambda query, **_: f"emb::{query}",
    )

    assert cache_obj.invalidate_by_query("remove-me") is True
    assert manager.invalidated == [("remove-me", "emb::probe")]


def test_clear_resets_intent_graph_and_memory_stores(tmp_path) -> None:
    cache_obj = Cache()
    cache_obj.init(data_manager=manager_factory("map", data_dir=str(tmp_path)))
    cache_obj.record_intent({"messages": [{"role": "user", "content": "hello"}]})
    cache_obj.remember_tool_result("tool", {"x": 1}, {"ok": True})

    assert cache_obj.intent_stats()["total_records"] == 1
    assert cache_obj.tool_memory_stats()["total_entries"] == 1

    cache_obj.clear()

    assert cache_obj.intent_stats()["total_records"] == 0
    assert cache_obj.tool_memory_stats()["total_entries"] == 0


def test_mcp_gateway_uses_cache_config_timeout() -> None:
    gateway = MCPGateway()
    gateway.register_tool(
        server_name="docs",
        tool_name="search",
        endpoint="https://example.com/search",
    )
    cache_obj = Cache()
    cache_obj.config = Config(mcp_timeout_s=12.5)

    with patch(
        "byte.mcp_gateway.requests.request", return_value=_HTTPResponse({"ok": True})
    ) as mock_request:
        gateway.call_tool("docs", "search", {"q": "byte"}, cache_obj=cache_obj)

    assert mock_request.call_args.kwargs["timeout"] == 12.5


def test_similarity_numpy_module_alias_remains_compatible() -> None:
    legacy_module = importlib.import_module("byte.similarity_evaluation.np")

    assert legacy_module.NumpyNormEvaluation is NumpyNormEvaluation


def test_adapter_facade_stays_thin() -> None:
    for module_path, line_limit in MODULE_LIMITS.items():
        lines = module_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= line_limit, module_path.as_posix()


def test_core_hotspot_targets_are_architecture_ratcheted() -> None:
    for rel_path in CORE_HOTSPOT_TARGETS:
        assert REPO_ROOT / rel_path in MODULE_LIMITS, rel_path


def test_public_facades_and_wrappers_do_not_use_import_star() -> None:
    for module_path in EXPLICIT_FACADE_MODULES:
        module_tree = _module_tree(module_path)
        import_star = [
            node for node in ast.walk(module_tree) if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
        ]
        assert not import_star, module_path.as_posix()


def test_public_facades_and_wrappers_define_literal_dunder_all() -> None:
    for module_path in EXPLICIT_FACADE_MODULES:
        literal_exports = _literal_dunder_all(module_path)
        assert literal_exports, module_path.as_posix()


def test_public_facades_and_wrappers_do_not_export_private_names() -> None:
    for module_path in STRICT_EXPORT_MODULES:
        for export_name in _literal_dunder_all(module_path):
            assert not export_name.startswith("_"), f"{module_path.as_posix()} exports {export_name}"


def test_runtime_public_facade_exports_only_supported_surface() -> None:
    runtime_exports = _literal_dunder_all(REPO_ROOT / "byte" / "adapter" / "runtime.py")

    assert runtime_exports == [
        "adapt",
        "aadapt",
        "get_budget_tracker",
        "get_quality_scorer",
        "cache_health_check",
        "acache_health_check",
    ]


def test_server_internal_modules_do_not_import_server_facade() -> None:
    for module_path in sorted((REPO_ROOT / "byte_server").glob("_server_*.py")):
        module_tree = _module_tree(module_path)
        offending_nodes = []
        for node in ast.walk(module_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "byte_server.server":
                        offending_nodes.append(node)
            if isinstance(node, ast.ImportFrom) and node.module in {"byte_server.server", "byte_server"}:
                if node.module == "byte_server.server":
                    offending_nodes.append(node)
                if node.module == "byte_server" and any(alias.name == "server" for alias in node.names):
                    offending_nodes.append(node)
        assert not offending_nodes, module_path.as_posix()


def test_provider_wrappers_remain_compatible_without_module_aliasing() -> None:
    for provider_name in ("openai", "anthropic", "openrouter"):
        wrapper_module = importlib.import_module(f"byte.adapter.{provider_name}")
        backend_module = importlib.import_module(f"byte._backends.{provider_name}")
        wrapper_path = REPO_ROOT / "byte" / "adapter" / f"{provider_name}.py"

        assert wrapper_module is not backend_module
        assert getattr(wrapper_module, "ChatCompletion", None) is getattr(
            backend_module, "ChatCompletion", None
        )
        assert wrapper_module.__all__ == _literal_dunder_all(wrapper_path)

        wrapper_source = wrapper_path.read_text(encoding="utf-8")
        assert "sys.modules[__name__]" not in wrapper_source


def test_server_request_models_live_in_dedicated_module() -> None:
    import byte_server.models as server_models
    import byte_server.server as server_module

    assert server_module.CacheData is server_models.CacheData
    assert server_module.MCPToolCall is server_models.MCPToolCall


def test_server_create_app_returns_singleton() -> None:
    import byte_server.server as server_module

    assert server_module.create_app() is server_module.app
    assert server_module.__all__ == [
        "CacheData",
        "FeedbackData",
        "MCPToolCall",
        "MCPToolRegistration",
        "MemoryArtifactExportData",
        "MemoryArtifactImportData",
        "MemoryImportData",
        "WarmData",
        "app",
        "create_app",
        "main",
    ]
