from __future__ import annotations

import subprocess

from byte._devtools import check_repo_hygiene


def test_git_executable_prefers_path(monkeypatch) -> None:
    monkeypatch.setattr(check_repo_hygiene.shutil, "which", lambda name: "git-from-path.exe")

    assert check_repo_hygiene._git_executable() == "git-from-path.exe"


def test_git_executable_uses_portable_install(monkeypatch, tmp_path) -> None:
    portable_git = tmp_path / "ByteNewTools" / "git" / "cmd" / "git.exe"
    portable_git.parent.mkdir(parents=True)
    portable_git.write_text("", encoding="utf-8", newline="\n")

    monkeypatch.setattr(check_repo_hygiene.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "missing-program-files"))

    assert check_repo_hygiene._git_executable() == str(portable_git)


def test_tracked_paths_fallback_ignores_local_env_and_build_dirs(tmp_path, monkeypatch) -> None:
    project_file = tmp_path / "byte" / "module.py"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("print('ok')\n", encoding="utf-8")

    ignored_python = tmp_path / ".venv" / "Lib" / "site-packages" / "ignored.py"
    ignored_python.parent.mkdir(parents=True)
    ignored_python.write_text("print('ignore')\n", encoding="utf-8")

    ignored_wheel = tmp_path / "dist" / "artifact.whl"
    ignored_wheel.parent.mkdir(parents=True)
    ignored_wheel.write_bytes(b"wheel")

    ignored_cache = tmp_path / "tests" / ".pytest_cache" / "README.md"
    ignored_cache.parent.mkdir(parents=True)
    ignored_cache.write_text("cache\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_repo_hygiene, "_git_executable", lambda: None)

    tracked_paths = check_repo_hygiene._tracked_paths()

    assert project_file in tracked_paths
    assert ignored_python not in tracked_paths
    assert ignored_wheel not in tracked_paths
    assert ignored_cache not in tracked_paths


def test_tracked_paths_skips_deleted_git_entries(tmp_path, monkeypatch) -> object:
    existing = tmp_path / "README.md"
    existing.write_text("ok\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_repo_hygiene, "_git_executable", lambda: "git.exe")

    def fake_git(*args, **kwargs) -> object:
        return subprocess.CompletedProcess(
            args[0],
            0,
            b"README.md\x00docs/bootcamp/index.rst\x00",
            b"",
        )

    monkeypatch.setattr(check_repo_hygiene.subprocess, "run", fake_git)

    assert check_repo_hygiene._tracked_paths() == [existing]


def test_lint_path_flags_legacy_repo_url(tmp_path, monkeypatch) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "See https://github.com/byte-ai/byte for details.\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(readme) == ["README.md: legacy repository URL"]


def test_lint_path_flags_legacy_validation_wording(tmp_path, monkeypatch) -> None:
    validation_doc = tmp_path / "docs" / "code-validation.md"
    validation_doc.parent.mkdir(parents=True)
    validation_doc.write_text(
        "AI-generated code is untrusted until reviewed.\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(validation_doc) == [
        "docs/code-validation.md: legacy validation wording"
    ]


def test_lint_path_flags_stale_distribution_guidance(tmp_path, monkeypatch) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("Run `pip install byte`.\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(readme) == ["README.md: stale distribution guidance"]


def test_lint_path_flags_stale_public_product_name(tmp_path, monkeypatch) -> None:
    usage = tmp_path / "docs" / "usage.md"
    usage.parent.mkdir(parents=True)
    usage.write_text("ByteAI Cache is here.\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(usage) == ["docs/usage.md: stale public product name"]


def test_lint_path_flags_transitional_repo_name_on_brand_surface(tmp_path, monkeypatch) -> None:
    architecture = tmp_path / "docs" / "architecture.md"
    architecture.parent.mkdir(parents=True)
    architecture.write_text("ByteNew is the public product.\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(architecture) == [
        "docs/architecture.md: transitional repository naming in public surface"
    ]


def test_lint_path_allows_repo_url_in_readme(tmp_path, monkeypatch) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "git clone https://github.com/anshikaeidetic/ByteNew.git Byte\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(readme) == []


def test_lint_path_skips_legacy_copy_checks_for_hygiene_tests(tmp_path, monkeypatch) -> None:
    test_file = tmp_path / "tests" / "unit_tests" / "test_repo_hygiene.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "legacy = 'https://github.com/byte-ai/byte'\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(test_file) == []


def test_lint_path_flags_non_canonical_examples(tmp_path, monkeypatch) -> None:
    legacy_example = tmp_path / "examples" / "integrate" / "openai" / "basic_usage.py"
    legacy_example.parent.mkdir(parents=True)
    legacy_example.write_text("print('legacy')\n", encoding="utf-8", newline="\n")

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(legacy_example) == [
        "examples/integrate/openai/basic_usage.py: non-canonical example tracked"
    ]


def test_lint_path_flags_files_over_size_budget(tmp_path, monkeypatch) -> None:
    oversized = tmp_path / "examples" / "benchmark" / "mock_data.json"
    oversized.parent.mkdir(parents=True)
    oversized.write_bytes(b"x" * (check_repo_hygiene.MAX_TRACKED_FILE_SIZE_BYTES + 1))

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(oversized) == [
        "examples/benchmark/mock_data.json: non-canonical example tracked",
        "examples/benchmark/mock_data.json: tracked file exceeds size budget",
    ]


def test_lint_path_allows_large_benchmark_workloads(tmp_path, monkeypatch) -> None:
    workload = tmp_path / "byte" / "benchmarking" / "workloads" / "large.json"
    workload.parent.mkdir(parents=True)
    workload.write_bytes(b"x" * (check_repo_hygiene.MAX_TRACKED_FILE_SIZE_BYTES + 1))

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(workload) == []


def test_benchmark_entry_size_problem_flags_new_oversized_module() -> None:
    problem = check_repo_hygiene._benchmark_entry_size_problem(
        "byte/benchmarking/programs/new_entry.py",
        check_repo_hygiene.MAX_BENCHMARK_ENTRY_LINES + 1,
    )

    assert problem == (
        "byte/benchmarking/programs/new_entry.py: benchmark module exceeds "
        f"{check_repo_hygiene.MAX_BENCHMARK_ENTRY_LINES} lines without an explicit hygiene "
        "allowlist rationale"
    )


def test_benchmark_entry_size_problem_allows_documented_exception() -> None:
    assert (
        check_repo_hygiene._benchmark_entry_size_problem(
            "byte/benchmarking/programs/deep_openai_prompt_stress_benchmark.py",
            check_repo_hygiene.MAX_BENCHMARK_ENTRY_LINES + 100,
        )
        is None
    )


def test_refactored_benchmark_entries_are_not_allowlisted() -> None:
    refactored_paths = {
        "byte/benchmarking/programs/deep_deepseek_runtime_optimization_benchmark.py",
        "byte/benchmarking/programs/deep_multi_provider_routing_memory.py",
        "byte/benchmarking/programs/deep_openai_coding_benchmark.py",
        "byte/benchmarking/programs/deep_openai_cost_levers.py",
        "byte/benchmarking/programs/deep_openai_surface_benchmark.py",
        "byte/benchmarking/workload_generator.py",
    }

    assert refactored_paths.isdisjoint(check_repo_hygiene.BENCHMARK_ENTRY_SIZE_ALLOWLIST)


def test_lint_path_flags_direct_scripts_import_in_tests(tmp_path, monkeypatch) -> None:
    test_file = tmp_path / "tests" / "unit_tests" / "test_bad_import.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "from " "scripts import bootstrap_dev\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(test_file) == [
        "tests/unit_tests/test_bad_import.py: test module imports repo-root scripts package directly"
    ]


def test_lint_path_flags_direct_scripts_module_import_in_tests(tmp_path, monkeypatch) -> None:
    test_file = tmp_path / "tests" / "unit_tests" / "test_bad_module_import.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "from " "scripts.check_repo_hygiene import main\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(test_file) == [
        "tests/unit_tests/test_bad_module_import.py: test module imports repo-root scripts module directly"
    ]


def test_lint_path_flags_benchmark_shim_import_in_tests(tmp_path, monkeypatch) -> None:
    test_file = tmp_path / "tests" / "unit_tests" / "test_bad_benchmark_import.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "import " "benchmark as benchmark_entry\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(test_file) == [
        "tests/unit_tests/test_bad_benchmark_import.py: test module imports the repo-root benchmark shim directly"
    ]


def test_lint_path_flags_private_tests_helper_import_in_tests(tmp_path, monkeypatch) -> None:
    test_file = tmp_path / "tests" / "unit_tests" / "test_bad_tests_import.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "from " "tests._optional_deps import feature_available\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(check_repo_hygiene, "REPO_ROOT", tmp_path)

    assert check_repo_hygiene._lint_path(test_file) == [
        "tests/unit_tests/test_bad_tests_import.py: test module imports private helpers from the tests package"
    ]
