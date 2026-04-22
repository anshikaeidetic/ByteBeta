from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from byte.benchmarking.quickstart import run_local_comparison, run_product_benchmark
from byte.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[3]


def _fake_results() -> dict:
    return {
        "providers": {
            "openai": {
                "systems": {
                    "direct": {
                        "phases": {
                            "cold": {
                                "summary": {
                                    "request_count": 6,
                                    "actual_reuse_rate": 0.0,
                                    "p50_latency_ms": 120.0,
                                    "p95_latency_ms": 180.0,
                                    "safe_answer_rate": 1.0,
                                },
                            }
                        }
                    },
                    "byte": {
                        "phases": {
                            "cold": {
                                "summary": {
                                    "request_count": 6,
                                    "actual_reuse_rate": 0.5,
                                    "p50_latency_ms": 40.0,
                                    "p95_latency_ms": 75.0,
                                    "safe_answer_rate": 1.0,
                                },
                                "comparison_to_direct": {"cost_reduction_ratio": 0.65},
                            }
                        }
                    },
                }
            }
        },
        "artifacts": {
            "engineering_json": "local/report.json",
            "engineering_markdown": "local/report.md",
        },
    }


@pytest.mark.requires_feature("onnx", "sqlalchemy", "faiss")
def test_run_local_comparison_runs_without_provider_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    results = run_local_comparison(
        provider="openai",
        profile="tier1",
        systems=["direct", "native_cache", "byte"],
        phases=["cold"],
        out_dir=tmp_path,
        max_items_per_family=1,
        concurrency=1,
    )

    assert "openai" in results["providers"]
    assert sorted(results["systems_executed"]) == ["byte", "direct", "native_cache"]


def test_run_product_benchmark_writes_summary_and_skips_live(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("byte.benchmarking.quickstart.run_local_comparison", lambda **kwargs: _fake_results())
    monkeypatch.setattr("byte.benchmarking.quickstart._provider_key_available", lambda provider: False)

    summary = run_product_benchmark(out_dir=str(tmp_path))

    assert Path(summary["artifacts"]["summary_json"]).exists()
    assert Path(summary["artifacts"]["summary_markdown"]).exists()
    assert summary["live_status"].startswith("skipped:")
    output = capsys.readouterr().out
    assert "LOCAL comparison" in output
    assert "live skipped" in output


def test_benchmark_shim_invokes_quickstart_runner(monkeypatch, tmp_path) -> object:
    calls: list[list[str]] = []

    def _capture() -> int:
        calls.append(list(sys.argv))
        return 0

    monkeypatch.setattr("byte.benchmarking.quickstart.main", _capture)
    monkeypatch.setattr("sys.argv", ["benchmark.py", "--out-dir", str(tmp_path)])

    try:
        runpy.run_path(str(ROOT / "benchmark.py"), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("benchmark.py should exit through SystemExit when run as a script.")

    assert calls


def test_cli_benchmark_subcommand_uses_quickstart_runner(monkeypatch, tmp_path) -> object:
    calls: list[dict] = []

    def _capture(**kwargs) -> object:
        calls.append(kwargs)
        return {"rows": [], "artifacts": {"root": str(tmp_path)}}

    monkeypatch.setattr("byte.cli.run_product_benchmark", _capture)
    monkeypatch.setattr("sys.argv", ["byte", "benchmark", "--out-dir", str(tmp_path)])

    cli_main()

    assert calls
