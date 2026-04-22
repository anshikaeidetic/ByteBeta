"""Shared optional-feature metadata for runtime loaders, tests, and CI."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

PROJECT_DISTRIBUTION_NAME = "byteai-cache"


@dataclass(frozen=True)
class OptionalFeature:
    """Describe an optional Byte capability and the modules it depends on."""

    extra: str | None
    module_groups: tuple[tuple[str, ...], ...]
    install_packages: tuple[str, ...] = ()
    service_hint: str | None = None

    def install_hint(self, *, project_name: str = PROJECT_DISTRIBUTION_NAME) -> str:
        hints: list[str] = []
        if self.extra:
            hints.append(f'`pip install "{project_name}[{self.extra}]"`')
            hints.append(f'`pip install -e ".[{self.extra}]"`')
        if self.install_packages:
            hints.append("`pip install " + " ".join(self.install_packages) + "`")

        if not hints:
            return "Install the required dependency before using this feature."

        if len(hints) == 1:
            guidance = f"Install it with {hints[0]}."
        else:
            guidance = f"Install it with {hints[0]} or {hints[1]}."
            for hint in hints[2:]:
                guidance += f" If you only need the underlying SDKs, {hint} also works."
        if self.service_hint:
            guidance += f" {self.service_hint}"
        return guidance


OPTIONAL_FEATURES: dict[str, OptionalFeature] = {
    "openai": OptionalFeature(extra="openai", module_groups=(("openai",),)),
    "groq": OptionalFeature(extra="groq", module_groups=(("groq",),)),
    "sqlalchemy": OptionalFeature(extra="sql", module_groups=(("sqlalchemy",),)),
    "transformers": OptionalFeature(
        extra="huggingface",
        module_groups=(("torch", "transformers"),),
    ),
    "onnx": OptionalFeature(
        extra="onnx",
        module_groups=(("onnxruntime", "transformers", "huggingface_hub", "sentencepiece"),),
    ),
    "langchain": OptionalFeature(
        extra="langchain",
        module_groups=(("langchain", "langchain_core", "langchain_openai"),),
    ),
    "faiss": OptionalFeature(extra="faiss", module_groups=(("faiss",),)),
    "redis": OptionalFeature(
        extra=None,
        module_groups=(("redis", "redis_om"),),
        install_packages=("redis", "redis-om"),
    ),
    "mongo": OptionalFeature(
        extra=None,
        module_groups=(("pymongo", "mongoengine"),),
        install_packages=("pymongo", "mongoengine"),
    ),
    "dynamo": OptionalFeature(
        extra=None,
        module_groups=(("boto3",),),
        install_packages=("boto3",),
    ),
    "pgvector": OptionalFeature(
        extra=None,
        module_groups=(("sqlalchemy", "psycopg2"),),
        install_packages=("sqlalchemy", "psycopg2-binary"),
        service_hint="A reachable PostgreSQL server is also required for live tests.",
    ),
    "qdrant": OptionalFeature(
        extra=None,
        module_groups=(("qdrant_client",),),
        install_packages=("qdrant-client",),
    ),
    "cohere": OptionalFeature(
        extra=None,
        module_groups=(("cohere",),),
        install_packages=("cohere",),
    ),
    "llama_cpp": OptionalFeature(
        extra=None,
        module_groups=(("llama_cpp",),),
        install_packages=("llama-cpp-python",),
    ),
    "sbert": OptionalFeature(
        extra=None,
        module_groups=(("sentence_transformers",),),
        install_packages=("sentence-transformers",),
    ),
    "milvus_sbert": OptionalFeature(
        extra=None,
        module_groups=(
            ("sentence_transformers", "pymilvus"),
            ("sentence_transformers", "chromadb"),
        ),
        install_packages=("sentence-transformers", "pymilvus"),
        service_hint="A supported Milvus or Chroma-compatible vector store is also required.",
    ),
    "paddle": OptionalFeature(
        extra=None,
        module_groups=(("google.protobuf", "paddle", "paddlenlp"),),
        install_packages=("protobuf==3.20.0", "paddlepaddle", "paddlenlp"),
    ),
    "timm": OptionalFeature(
        extra=None,
        module_groups=(("timm",),),
        install_packages=("timm",),
    ),
    "uform": OptionalFeature(
        extra=None,
        module_groups=(("uform",),),
        install_packages=("uform",),
    ),
    "vit": OptionalFeature(
        extra=None,
        module_groups=(("vit",),),
        install_packages=("vit",),
    ),
    "pillow": OptionalFeature(
        extra=None,
        module_groups=(("PIL",),),
        install_packages=("pillow",),
    ),
    "usearch": OptionalFeature(
        extra=None,
        module_groups=(("usearch",),),
        install_packages=("usearch",),
    ),
    "docarray": OptionalFeature(
        extra=None,
        module_groups=(("docarray",),),
        install_packages=("docarray",),
    ),
    "telemetry": OptionalFeature(
        extra="observability",
        module_groups=(("opentelemetry.sdk",),),
    ),
    "hypothesis": OptionalFeature(
        extra=None,
        module_groups=(("hypothesis",),),
        install_packages=("hypothesis",),
    ),
}

FEATURE_ALIASES = {
    "huggingface": "transformers",
    "sql": "sqlalchemy",
    "torch_transformers": "transformers",
}


class MissingOptionalDependencyError(ModuleNotFoundError):
    """Raised when a Byte optional feature is used without its dependency stack."""


def normalize_feature_name(name: str) -> str:
    return FEATURE_ALIASES.get(name, name)


def feature_spec(name: str) -> OptionalFeature:
    canonical = normalize_feature_name(name)
    try:
        return OPTIONAL_FEATURES[canonical]
    except KeyError as exc:  # pragma: no cover - defensive caller contract
        raise KeyError(f"Unknown optional feature: {name}") from exc


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def feature_available(name: str) -> bool:
    spec = feature_spec(name)
    return any(
        all(_module_available(module_name) for module_name in module_group)
        for module_group in spec.module_groups
    )


def feature_for_module(module_name: str) -> str | None:
    matched_feature: str | None = None
    matched_prefix_length = -1
    for feature_name, spec in OPTIONAL_FEATURES.items():
        for module_group in spec.module_groups:
            for registered_module in module_group:
                if module_name == registered_module or module_name.startswith(f"{registered_module}."):
                    if len(registered_module) > matched_prefix_length:
                        matched_feature = feature_name
                        matched_prefix_length = len(registered_module)
    return matched_feature


def _package_install_hint(package_name: str) -> str:
    return f"Install it with `pip install {package_name}`."


def missing_dependency_error_for_module(
    module_name: str,
    *,
    package: str | None = None,
) -> MissingOptionalDependencyError:
    package_name = str(package or module_name).strip().strip("'\"")
    feature_name = feature_for_module(module_name)
    if feature_name is None:
        message = (
            f"Optional dependency '{package_name}' is not installed. "
            f"{_package_install_hint(package_name)}"
        )
        return MissingOptionalDependencyError(message)

    spec = feature_spec(feature_name)
    message = (
        f"Optional feature '{feature_name}' is not installed. "
        f"{spec.install_hint()}"
    )
    return MissingOptionalDependencyError(message)


def missing_dependency_error_for_feature(name: str) -> MissingOptionalDependencyError:
    feature_name = normalize_feature_name(name)
    spec = feature_spec(feature_name)
    return MissingOptionalDependencyError(
        f"Optional feature '{feature_name}' is not installed. {spec.install_hint()}"
    )


def require_feature(name: str) -> None:
    if feature_available(name):
        return
    raise missing_dependency_error_for_feature(name)


__all__ = [
    "FEATURE_ALIASES",
    "OPTIONAL_FEATURES",
    "MissingOptionalDependencyError",
    "OptionalFeature",
    "feature_available",
    "feature_for_module",
    "feature_spec",
    "missing_dependency_error_for_feature",
    "missing_dependency_error_for_module",
    "normalize_feature_name",
    "require_feature",
]
