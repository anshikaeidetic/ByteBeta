from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from byte.benchmarking import workload_generator
from byte.benchmarking._program_execution import execute_case_with_retries
from byte.benchmarking._program_models import ExecutionConfig, RequestCase
from byte.benchmarking.workload_families.registry import FAMILY_MODULES, iter_family_builders

ROOT = Path(__file__).resolve().parents[3]

THIN_ENTRYPOINTS = (
    "byte/benchmarking/programs/deep_deepseek_runtime_optimization_benchmark.py",
    "byte/benchmarking/programs/deep_multi_provider_routing_memory.py",
    "byte/benchmarking/programs/deep_openai_coding_benchmark.py",
    "byte/benchmarking/programs/deep_openai_cost_levers.py",
    "byte/benchmarking/programs/deep_openai_surface_benchmark.py",
    "byte/benchmarking/workload_generator.py",
)


def test_refactored_benchmark_entrypoints_stay_thin() -> None:
    for relative_path in THIN_ENTRYPOINTS:
        line_count = len((ROOT / relative_path).read_text(encoding="utf-8").splitlines())

        assert line_count <= 80, relative_path


def test_program_entrypoints_do_not_import_private_implementations_at_import_time() -> None:
    command = [
        sys.executable,
        "-c",
        textwrap.dedent(
            """
            import importlib
            import sys

            wrappers = [
                "byte.benchmarking.programs.deep_deepseek_runtime_optimization_benchmark",
                "byte.benchmarking.programs.deep_multi_provider_routing_memory",
                "byte.benchmarking.programs.deep_openai_coding_benchmark",
                "byte.benchmarking.programs.deep_openai_cost_levers",
                "byte.benchmarking.programs.deep_openai_surface_benchmark",
            ]
            for name in wrappers:
                importlib.import_module(name)
            loaded = [
                name for name in sys.modules
                if name.startswith("byte.benchmarking._program_impl.")
            ]
            if loaded:
                raise SystemExit("private benchmark implementations imported: " + ", ".join(loaded))
            """
        ),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_workload_family_registry_matches_public_generator_families() -> None:
    registry_names = [family_name for family_name, _ in FAMILY_MODULES]

    assert registry_names == list(workload_generator.FAMILY_LANES)


@pytest.mark.parametrize("family_name,builder", list(iter_family_builders()))
def test_workload_family_builders_are_deterministic(family_name, builder) -> None:
    first = builder()
    second = builder()

    assert first, family_name
    assert [item.item_id for item in first] == [item.item_id for item in second]
    assert {item.family for item in first} == {family_name}


def test_execute_case_with_retries_returns_success_record() -> None:
    case = RequestCase(
        case_id="case.ok",
        prompt="Return OK",
        expected="OK",
        group="smoke",
        variant="v1",
        kind="exact",
    )
    config = ExecutionConfig(provider="fake", model="fake-model", phase="cold")

    record = execute_case_with_retries(case, config, lambda: ("OK", {"prompt_tokens": 2}))

    assert record.ok is True
    assert record.case_id == "case.ok"
    assert record.provider == "fake"
    assert record.phase == "cold"
    assert record.usage["prompt_tokens"] == 2


def test_execute_case_with_retries_returns_structured_failure_record() -> None:
    case = RequestCase(
        case_id="case.fail",
        prompt="Return OK",
        expected="OK",
        group="smoke",
        variant="v1",
        kind="exact",
    )
    config = ExecutionConfig(
        provider="fake",
        model="fake-model",
        phase="warm",
        attempts=2,
        base_sleep_seconds=0,
    )

    def fail() -> tuple[str, dict[str, object]]:
        raise RuntimeError("provider unavailable")

    record = execute_case_with_retries(case, config, fail)

    assert record.ok is False
    assert record.error == {
        "provider": "fake",
        "case_id": "case.fail",
        "phase": "warm",
        "attempts": 2,
        "type": "RuntimeError",
        "message": "provider unavailable",
    }
