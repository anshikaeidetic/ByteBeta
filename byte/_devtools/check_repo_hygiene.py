"""Enforce repository hygiene rules for maintained source and test surfaces."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".editorconfig",
    ".gitignore",
    ".gitattributes",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".rst",
    ".sh",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    "Makefile",
}
FALLBACK_IGNORED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_PATHS = {"requirements.txt", "tests/requirements.txt"}
LEGACY_TEXT_PATTERNS = {
    "github.com/byte-ai/byte": "legacy repository URL",
    "ChatGPT": "legacy model wording",
    "AI-generated code": "legacy validation wording",
    "Google Colab": "legacy tutorial copy",
    "origin notebook": "legacy tutorial copy",
    "pip install byte": "stale distribution guidance",
}
LEGACY_TEXT_EXEMPT_PATHS = {
    "byte/_devtools/check_repo_hygiene.py",
    "scripts/check_repo_hygiene.py",
}
LEGACY_TEXT_EXEMPT_PREFIXES = ("tests/",)
BRAND_SURFACE_PREFIXES = ("docs/", "examples/")
BRAND_SURFACE_PATHS = {
    ".env.example",
    "CHANGELOG.md",
    "README.md",
    "byte/__init__.py",
    "byte/cli.py",
    "byte/client.py",
    "byte_inference/server.py",
    "byte_memory/server.py",
    "byte_server/server.py",
    "pyproject.toml",
}
PATH_COUPLING_EXEMPT_PATHS: set[str] = set()
FORBIDDEN_TEST_IMPORT_PATTERNS = {
    "from scripts import": "test module imports repo-root scripts package directly",
    "from scripts.": "test module imports repo-root scripts module directly",
    "import benchmark as": "test module imports the repo-root benchmark shim directly",
    "from tests._": "test module imports private helpers from the tests package",
}
COMPATIBILITY_BRAND_PATTERNS = {
    "ByteAI Cache": {
        "reason": "stale public product name",
        "allowlist": {
            "byte/_devtools/check_repo_hygiene.py",
            "scripts/check_repo_hygiene.py",
        },
    },
    "ByteNew": {
        "reason": "transitional repository naming in public surface",
        "allowlist": {
            "README.md",
            "byte/__init__.py",
            "byte/_devtools/check_repo_hygiene.py",
            "docs/conf.py",
            "pyproject.toml",
            "scripts/check_repo_hygiene.py",
        },
    },
}
MAX_TRACKED_FILE_SIZE_BYTES = 800_000
SIZE_ALLOWLIST_PATHS = {
    "docs/Byte-Local-Search.png",
    "docs/Byte-Multinode.png",
    "docs/Byte.png",
    "docs/ByteStructure.png",
}
SIZE_ALLOWLIST_PREFIXES = ("byte/benchmarking/workloads/",)
MAX_BENCHMARK_ENTRY_LINES = 400
BENCHMARK_ENTRY_SIZE_ALLOWLIST = {
    "byte/benchmarking/programs/advanced_openai_cost_patterns.py": (
        "legacy benchmark orchestrator pending modular split"
    ),
    "byte/benchmarking/programs/deep_deepseek_reasoning_reuse_benchmark.py": (
        "legacy benchmark orchestrator pending modular split"
    ),
    "byte/benchmarking/programs/deep_openai_comprehensive_workload_benchmark.py": (
        "legacy benchmark orchestrator pending modular split"
    ),
    "byte/benchmarking/programs/deep_openai_unified_router_benchmark.py": (
        "legacy benchmark orchestrator pending modular split"
    ),
    "byte/benchmarking/programs/deep_openai_100_request_mixed_benchmark.py": (
        "legacy benchmark orchestrator pending modular split"
    ),
    "byte/benchmarking/programs/deep_openai_1000_request_mixed_benchmark.py": (
        "legacy benchmark orchestrator pending modular split"
    ),
    "byte/benchmarking/programs/deep_openai_prompt_stress_benchmark.py": (
        "legacy benchmark orchestrator pending modular split"
    ),
}
ALLOWED_EXAMPLE_PATHS = {
    "examples/README.md",
    "examples/adapter/api.py",
}
ALLOWED_EXAMPLE_PREFIXES = (
    "examples/context_process/",
    "examples/kubernetes/",
    "examples/processor/",
    "examples/session/",
)


def _git_executable() -> str | None:
    git_path = shutil.which("git")
    if git_path:
        return git_path

    local_app_data = os.environ.get("LOCALAPPDATA")
    candidates = [
        (
            Path(local_app_data) / "ByteNewTools" / "git" / "cmd" / "git.exe"
            if local_app_data
            else None
        ),
        Path(os.environ.get("PROGRAMFILES", "")) / "Git" / "cmd" / "git.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Git" / "bin" / "git.exe",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate)
    return None


def _tracked_paths() -> list[Path]:
    git_executable = _git_executable()
    if git_executable is None:
        return [
            path
            for path in REPO_ROOT.rglob("*")
            if path.is_file()
            and not any(part in FALLBACK_IGNORED_PARTS for part in path.parts)
            and not any(part.endswith(".egg-info") for part in path.parts)
        ]
    try:
        result = subprocess.run(
            [git_executable, "ls-files", "-z"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=False,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [
            path
            for path in REPO_ROOT.rglob("*")
            if path.is_file()
            and not any(part in FALLBACK_IGNORED_PARTS for part in path.parts)
            and not any(part.endswith(".egg-info") for part in path.parts)
        ]
    return [
        REPO_ROOT / rel.decode("utf-8")
        for rel in result.stdout.split(b"\x00")
        if rel and (REPO_ROOT / rel.decode("utf-8")).is_file()
    ]


def _is_text_file(path: Path) -> bool:
    if path.name in TEXT_FILENAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return True
    return path.name.startswith(".") and path.suffix.lower() in TEXT_SUFFIXES


def _benchmark_entry_size_problem(rel_path: str, line_count: int) -> str | None:
    if not rel_path.endswith(".py"):
        return None
    if not (
        rel_path.startswith("byte/benchmarking/programs/")
        or rel_path == "byte/benchmarking/workload_generator.py"
    ):
        return None
    if line_count <= MAX_BENCHMARK_ENTRY_LINES:
        return None
    rationale = BENCHMARK_ENTRY_SIZE_ALLOWLIST.get(rel_path)
    if rationale:
        return None
    return (
        f"{rel_path}: benchmark module exceeds {MAX_BENCHMARK_ENTRY_LINES} lines without an "
        "explicit hygiene allowlist rationale"
    )


def _path_coupling_problems(rel_path: str, text: str) -> list[str]:
    if not rel_path.startswith("tests/") or rel_path in PATH_COUPLING_EXEMPT_PATHS:
        return []

    problems: list[str] = []
    for pattern, reason in FORBIDDEN_TEST_IMPORT_PATTERNS.items():
        if pattern in text:
            problems.append(f"{rel_path}: {reason}")
    return problems


def _lint_path(path: Path) -> list[str]:
    rel_path = path.relative_to(REPO_ROOT).as_posix()
    problems: list[str] = []
    if rel_path in FORBIDDEN_PATHS:
        problems.append(f"{rel_path}: tracked legacy dependency manifest")
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        problems.append(f"{rel_path}: tracked cache artifact directory")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        problems.append(f"{rel_path}: tracked Python bytecode artifact")
    if rel_path.startswith("docs/bootcamp/"):
        problems.append(f"{rel_path}: bootcamp docs are not part of the maintained docs surface")
    if (
        rel_path.startswith("examples/")
        and rel_path not in ALLOWED_EXAMPLE_PATHS
        and not any(rel_path.startswith(prefix) for prefix in ALLOWED_EXAMPLE_PREFIXES)
    ):
        problems.append(f"{rel_path}: non-canonical example tracked")
    if (
        path.stat().st_size > MAX_TRACKED_FILE_SIZE_BYTES
        and rel_path not in SIZE_ALLOWLIST_PATHS
        and not any(rel_path.startswith(prefix) for prefix in SIZE_ALLOWLIST_PREFIXES)
    ):
        problems.append(f"{rel_path}: tracked file exceeds size budget")
    if not _is_text_file(path):
        return problems

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append(f"{rel_path}: UTF-8 BOM present")
    if b"\r\n" in raw:
        problems.append(f"{rel_path}: CRLF line endings present")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        problems.append(f"{rel_path}: tracked text file is not valid UTF-8")
        return problems
    benchmark_size_problem = _benchmark_entry_size_problem(rel_path, len(text.splitlines()))
    if benchmark_size_problem is not None:
        problems.append(benchmark_size_problem)
    problems.extend(_path_coupling_problems(rel_path, text))
    if rel_path not in LEGACY_TEXT_EXEMPT_PATHS and not any(
        rel_path.startswith(prefix) for prefix in LEGACY_TEXT_EXEMPT_PREFIXES
    ):
        for pattern, reason in LEGACY_TEXT_PATTERNS.items():
            if pattern in text:
                problems.append(f"{rel_path}: {reason}")
    if rel_path in BRAND_SURFACE_PATHS or any(
        rel_path.startswith(prefix) for prefix in BRAND_SURFACE_PREFIXES
    ):
        for pattern, metadata in COMPATIBILITY_BRAND_PATTERNS.items():
            allowlist = set(metadata.get("allowlist", set()))
            if rel_path in allowlist:
                continue
            if pattern in text:
                problems.append(f"{rel_path}: {metadata['reason']}")
    return problems


def main() -> int:
    problems: list[str] = []
    for path in _tracked_paths():
        problems.extend(_lint_path(path))

    if problems:
        print("Repo hygiene check failed:")
        for problem in sorted(problems):
            print(f" - {problem}")
        return 1

    print("Repo hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
