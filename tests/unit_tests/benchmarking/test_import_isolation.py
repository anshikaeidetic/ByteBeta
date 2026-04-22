from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

_BLOCKER = textwrap.dedent(
    """
    import importlib.abc
    import os
    import sys

    _blocked_modules = tuple(
        module.strip()
        for module in os.environ.get("BYTE_BLOCK_OPTIONAL_MODULES", "").split(",")
        if module.strip()
    )


    class _BlockOptionalModules(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            del path, target
            for module in _blocked_modules:
                if fullname == module or fullname.startswith(module + "."):
                    raise ModuleNotFoundError(
                        f"blocked optional dependency during import-isolation test: {module}"
                    )
            return None


    if _blocked_modules:
        sys.meta_path.insert(0, _BlockOptionalModules())
    """
)


def _blocked_optional_env(tmp_path: Path, *modules: str) -> dict[str, str]:
    blocker_dir = tmp_path / "blocked_optional_sitecustomize"
    blocker_dir.mkdir()
    (blocker_dir / "sitecustomize.py").write_text(_BLOCKER, encoding="utf-8")
    env = os.environ.copy()
    env["BYTE_BLOCK_OPTIONAL_MODULES"] = ",".join(modules)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(blocker_dir), existing_pythonpath] if existing_pythonpath else [str(blocker_dir)]
    )
    return env


def test_benchmark_plan_modules_import_without_openai_sdk(tmp_path) -> None:
    command = [
        sys.executable,
        "-c",
        textwrap.dedent(
            """
            import importlib

            modules = [
                "byte.benchmarking._program_entry",
                "byte.benchmarking._program_execution",
                "byte.benchmarking._program_models",
                "byte.benchmarking.plans.deepseek_runtime_optimization",
                "byte.benchmarking.plans.deepseek_reasoning_reuse",
                "byte.benchmarking.plans.openai_mixed_100",
                "byte.benchmarking.plans.openai_mixed_1000",
                "byte.benchmarking.quickstart",
                "byte.benchmarking.workload_families.registry",
            ]
            for name in modules:
                importlib.import_module(name)
            """
        ),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=_blocked_optional_env(tmp_path, "anthropic", "groq", "langchain", "openai", "redis"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_benchmark_collection_succeeds_without_openai_sdk(tmp_path) -> None:
    collect_targets = [
        "tests/unit_tests/benchmarking",
        "tests/unit_tests/adapter/test_deepseek_runtime_benchmark_plan.py",
        "tests/unit_tests/adapter/test_deepseek_reasoning_reuse_benchmark_plan.py",
        "tests/unit_tests/adapter/test_mixed_benchmark_100_plan.py",
        "tests/unit_tests/adapter/test_mixed_benchmark_plan.py",
    ]
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", *collect_targets]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=_blocked_optional_env(tmp_path, "anthropic", "groq", "langchain", "openai", "redis"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
